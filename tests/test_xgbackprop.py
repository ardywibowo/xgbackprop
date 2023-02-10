import json
import os
import sys

import numpy as np
import torch
import ubjson
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgbackprop.xgbackprop import XGBackpropLayer


def test_xgbackprop():
    # load model from file
    model = '/Users/randyardywibowo/Github/xgbackprop/models/model.json'
    
    if model.endswith("json"):
        # use json format
        with open(model, "r") as fd:
            model = json.load(fd)
    elif model.endswith("ubj"):
        if ubjson is None:
            raise ImportError("ubjson is not installed.")
        # use ubjson format
        with open(model, "rb") as bfd:
            model = ubjson.load(bfd)
    else:
        raise ValueError(
            "Unexpected file extension. Supported file extension are json and ubj."
        )
    
    model = XGBackpropLayer(model)
    # model = torch.jit.script(model)
    # model = torch.compile(model)
    input = torch.nn.Embedding(2, 8)
    optimizer = torch.optim.Adam(input.parameters())
    
    outputs = []
    for i in range(1000):
        encoder_output = input(torch.tensor([0, 1]))
        output = model(encoder_output)
        if (i // 100) % 2 == 0:
            print("Target: -10")
            loss = torch.square(output)
            outputs.append(output.detach().numpy())
        else:
            print("Target: +10")
            loss = torch.square(output - 1)
            outputs.append(output.detach().numpy())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    outputs = np.array(outputs)
    plt.plot(outputs)
    
test_xgbackprop()
