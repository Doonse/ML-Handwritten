from Imports import *

# Funcitons
def sigmoid(x): # sigmoid function
    return 1 / (1 + np.exp(-x)) 

def d_sigmoid(x):# derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x)) 

def error(out, l): # Error function (MSE)
    return 1 / len(out) * np.sum((out - l) ** 2, axis=0)