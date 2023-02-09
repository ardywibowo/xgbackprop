import torch
import torch.nn as nn
import xgboost as xgb


class SHAPBackprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, xgboost_model: xgb.core.Booster):
        # TODO: Convert to numpy array and back to tensor
        x = xgb.DMatrix(x.numpy())
        shap_contribs = xgboost_model.predict(x, pred_contribs=True)
        ctx.save_for_backward(torch.Tensor(shap_contribs))
        
        preds = xgboost_model.predict(x)
        return torch.Tensor(preds)

    @staticmethod
    def backward(ctx, grad_output):
        shap_contribs = ctx.saved_tensors[0][:, :-1]
        return shap_contribs * grad_output, None

class SHAPBackpropLayer(nn.Module):
    def __init__(self, xgboost_model: xgb.core.Booster):
        super(SHAPBackpropLayer, self).__init__()
        self.shap_backprop = SHAPBackprop.apply
        self.xgboost_model = xgboost_model
        # if torch.cuda.is_available():
        #     self.xgboost_model.set_param({"predictor": "gpu_predictor"})

    def forward(self, x: torch.Tensor):
        return self.shap_backprop(x, self.xgboost_model)    
