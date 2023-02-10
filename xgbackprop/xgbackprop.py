from enum import IntEnum, unique
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ParamT = Dict[str, str]


def to_integers(data: Union[bytes, List[int]]) -> List[int]:
    """Convert a sequence of bytes to a list of Python integer"""
    return [v for v in data]


@unique
class SplitType(IntEnum):
    numerical = 0
    categorical = 1

class Leaf(nn.Module):
    def __init__(
        self,
        node_id: int,
        base_weight: float,
        gain: float,
        cover: float) -> None:
        
        super().__init__()
        
        self.node_id = node_id
        # self.register_buffer("gain", torch.Tensor([gain]))
        self.gain = torch.nn.Parameter(torch.Tensor([gain]))
        
        # statistic
        self.base_weight = base_weight
        self.cover = cover

    # @torch.jit.script
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gain * torch.ones_like(input[:, 0])

class Node(nn.Module):
    def __init__(
        self,
        node_id: int,
        split_idx: int,
        split_cond: float,
        default_left: bool,
        split_type: SplitType,
        categories: List[int],
        base_weight: float,
        gain: float,
        cover: float) -> None:
        
        super().__init__()
        
        self.node_id = node_id
        
        self.left: Node = None
        self.right: Node = None
        
        self.register_buffer("split_idx", torch.LongTensor([split_idx]))
        # self.register_buffer("split_cond", torch.Tensor([split_cond]))
        self.split_cond = torch.nn.Parameter(torch.Tensor([split_cond]))
        self.register_buffer("gain", torch.Tensor([gain]))
        
        self.default_left = default_left
        self.split_type = split_type
        self.categories = categories
        
        # statistic
        self.base_weight = base_weight
        self.cover = cover

    # @torch.jit.script
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        condition = input[:, self.split_idx].flatten() - self.split_cond
        if self.training:
            condition = torch.cat([condition[:, None], torch.zeros_like(condition[:, None])], -1)
            weight = F.gumbel_softmax(condition, hard=False)[:, 0]            
        else:
            weight = torch.heaviside(condition, torch.Tensor([0]))
            
        # left is yes, right is no
        out = (1.0 - weight) * self.left(input) + weight * self.right(input)
        return out

class XGBackpropLayer(nn.Module):
    """Gradient boosted tree model."""

    def __init__(self, model: dict) -> None:
        """Construct the Model from a JSON object.
        parameters
        ----------
         model : A dictionary loaded by json representing a XGBoost boosted tree model.
        """
        super().__init__()
        
        # Basic properties of a model
        self.learner_model_shape: ParamT = model["learner"]["learner_model_param"]
        self.num_output_group = int(self.learner_model_shape["num_class"])
        self.num_feature = int(self.learner_model_shape["num_feature"])
        
        self.register_buffer("base_score", torch.Tensor([float(self.learner_model_shape["base_score"])]))
        # self.base_score = float(self.learner_model_shape["base_score"])
        
        # A field encoding which output group a tree belongs
        self.tree_info = model["learner"]["gradient_booster"]["model"]["tree_info"]

        model_shape: ParamT = model["learner"]["gradient_booster"]["model"][
            "gbtree_model_param"
        ]

        # JSON representation of trees
        j_trees = model["learner"]["gradient_booster"]["model"]["trees"]

        # Load the trees
        self.num_trees = int(model_shape["num_trees"])
        self.leaf_size = int(model_shape["size_leaf_vector"])
        # Right now XGBoost doesn't support vector leaf yet
        assert self.leaf_size == 0, str(self.leaf_size)

        trees: List[Node] = []
        for i in range(self.num_trees):
            tree: Dict[str, Any] = j_trees[i]
            tree_id = int(tree["id"])
            assert tree_id == i, (tree_id, i)
            # - properties
            left_children: List[int] = tree["left_children"]
            right_children: List[int] = tree["right_children"]
            parents: List[int] = tree["parents"]
            split_conditions: List[float] = tree["split_conditions"]
            split_indices: List[int] = tree["split_indices"]
            # when ubjson is used, this is a byte array with each element as uint8
            default_left = to_integers(tree["default_left"])

            # - categorical features
            # when ubjson is used, this is a byte array with each element as uint8
            split_types = to_integers(tree["split_type"])
            # categories for each node is stored in a CSR style storage with segment as
            # the begin ptr and the `categories' as values.
            cat_segments: List[int] = tree["categories_segments"]
            cat_sizes: List[int] = tree["categories_sizes"]
            # node index for categorical nodes
            cat_nodes: List[int] = tree["categories_nodes"]
            assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
            cats = tree["categories"]
            assert len(left_children) == len(split_types)

            # The storage for categories is only defined for categorical nodes to
            # prevent unnecessary overhead for numerical splits, we track the
            # categorical node that are processed using a counter.
            cat_cnt = 0
            if cat_nodes:
                last_cat_node = cat_nodes[cat_cnt]
            else:
                last_cat_node = -1
            node_categories: List[List[int]] = []
            for node_id in range(len(left_children)):
                if node_id == last_cat_node:
                    beg = cat_segments[cat_cnt]
                    size = cat_sizes[cat_cnt]
                    end = beg + size
                    node_cats = cats[beg:end]
                    # categories are unique for each node
                    assert len(set(node_cats)) == len(node_cats)
                    cat_cnt += 1
                    if cat_cnt == len(cat_nodes):
                        last_cat_node = -1  # continue to process the rest of the nodes
                    else:
                        last_cat_node = cat_nodes[cat_cnt]
                    assert node_cats
                    node_categories.append(node_cats)
                else:
                    # append an empty node, it's either a numerical node or a leaf.
                    node_categories.append([])

            # Stats
            base_weights: List[float] = tree["base_weights"]
            gain: List[float] = tree["loss_changes"]
            cover: List[float] = tree["sum_hessian"]

            # Construct a list of nodes that have complete information
            node_id = 0
            if left_children[node_id] > -1:
                root = Node(
                    node_id = node_id,
                    split_idx = split_indices[node_id],
                    split_cond = split_conditions[node_id],
                    default_left = (default_left[node_id] == 1),
                    split_type = SplitType(split_types[node_id]),
                    categories = node_categories[node_id],
                    base_weight = base_weights[node_id],
                    gain = gain[node_id],
                    cover = cover[node_id]
                )
            else:
                root = Leaf(
                    node_id = node_id,
                    base_weight = base_weights[node_id],
                    gain = split_conditions[node_id],
                    cover = cover[node_id]
                )
            
            frontier = [root]
            while frontier:
                node = frontier.pop()
                node_id = node.node_id
                if left_children[node_id] > -1:
                    n_id = left_children[node_id]
                    if left_children[n_id] > -1:
                        left_node = Node(
                            node_id = n_id,
                            split_idx = split_indices[n_id],
                            split_cond = split_conditions[n_id],
                            default_left = (default_left[n_id] == 1),
                            split_type = SplitType(split_types[n_id]),
                            categories = node_categories[n_id],
                            base_weight = base_weights[n_id],
                            gain = gain[n_id],
                            cover = cover[n_id]
                        )
                        node.left = left_node
                        frontier.append(left_node)
                    else:
                        left_node = Leaf(
                            node_id = n_id,
                            base_weight = base_weights[n_id],
                            gain = split_conditions[n_id],
                            cover = cover[n_id]
                        )
                        node.left = left_node
                if right_children[node_id] > -1:
                    n_id = right_children[node_id]
                    if right_children[n_id] > -1:
                        right_node = Node(
                            node_id = n_id,
                            split_idx = split_indices[n_id],
                            split_cond = split_conditions[n_id],
                            default_left = (default_left[n_id] == 1),
                            split_type = SplitType(split_types[n_id]),
                            categories = node_categories[n_id],
                            base_weight = base_weights[n_id],
                            gain = gain[n_id],
                            cover = cover[n_id]
                        )
                        node.right = right_node
                        frontier.append(right_node)
                    else:
                        right_node = Leaf(
                            node_id = n_id,
                            base_weight = base_weights[n_id],
                            gain = split_conditions[n_id],
                            cover = cover[n_id]
                        )
                        node.right = right_node
            trees.append(root)
        self.trees = nn.ModuleList(trees)
    
    # @torch.jit.script
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([tree(input) for tree in self.trees], dim=-1)
        out = torch.sum(preds, dim=-1) + self.base_score
        return out
