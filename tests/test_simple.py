import json
import os
import sys

import numpy as np
import torch
import ubjson
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgbackprop.xgbackprop import XGBackpropLayer

def test_simple():
    torch.cuda.is_available()

    # load model from file
    model_name = '/Users/randyardywibowo/Github/xgbackprop/models/model_simple.json'
    with open(model_name, "r") as fd:
        model = json.load(fd)

    device = 'cuda'
    model = XGBackpropLayer(model)

    x_matrix = np.array(
            [
                [1, 1],
                [0, 0]
            ]
        )
    x_tensor = torch.tensor(x_matrix, dtype=torch.float32)
    out = model(x_tensor)

test_simple()
