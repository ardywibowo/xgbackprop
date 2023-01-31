# xgbackprop
PyTorch implementation of backpropagating through decision trees

## Overview

This package implements two flavors of backpropagation through XGBoost decision trees. `SHAPBackpropLayer` and `XGBackpropLayer`. 

The `SHAPBackpropLayer` approximates the gradients of a decision tree by Shapley values. the Shapley values of the input features are treated as the gradient `df(x) / dx`. 

The `XGBackpropLayer` approximates the gradients by relaxing the decision tree branches into sigmoids and samples paths through the decision tree in the style of Straight-Through (ST) approximators for traditional Deep Neural Network (DNN) architectures.
