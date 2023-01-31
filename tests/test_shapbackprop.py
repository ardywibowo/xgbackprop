import os
import pickle
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgbackprop.SHAPBackpropLayer import SHAPBackpropLayer

def test_backprop_model():
    # load model from file
    model = pickle.load(open("models/model.xgb", "rb"))
    
    layer = SHAPBackpropLayer(model)
    
    input = torch.nn.Embedding(1, 8)
    optimizer = torch.optim.Adam(input.parameters())
    
    outputs = []
    for i in range(1000):
        encoder_output = input(torch.tensor([0])) + 0.1 * torch.randn(1, 8)
        
        output = layer(encoder_output)
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
    
test_backprop_model()
