import pickle

import numpy as np
import xgboost as xgb
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_train_model():
    # load data
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=",")
    
    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    X_train = np.array(
        [
            [1, 1],
            [0, 0]
        ]
    )
    y_train = np.array(
        [1, 0]
    )
    
    # fit model no training data
    # Fetch dataset using sklearn
    num_round = 1
    param = {
        "eta": 0.05,
        "max_depth": 10,
    }

    # GPU accelerated training
    dtrain = xgb.DMatrix(X_train, label = y_train)
    model = xgb.train(param, dtrain, num_round)
    
    print(model)
    pickle.dump(model, open("models/model_simple.xgb", "wb"))
    model.save_model('models/model_simple.json')

test_train_model()
