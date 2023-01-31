import torch
import torch.nn as nn
import torch.nn.functional as F

class XGBackpropLayer(nn.Module):
    def __init__(self, xgboost_model):
        super(XGBackpropLayer, self).__init__()
        
        self.xgboost_model = xgboost_model
        if torch.cuda.is_available():
            self.xgboost_model.set_param({"predictor": "gpu_predictor"})
        self.xgb_unrolled = XGBackpropLayer.parse_xgb_model(xgboost_model)
    
    @staticmethod
    def parse_xgb_model(xgboost_model):
        return 0

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
