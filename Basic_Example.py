import numpy as np
from scipy.special import expit
import math


# class NeuralNetwork:
#     def __init__(self, layer_structure, input_layer, bias=0):
#         self.input_layer = input_layer
#         self.layer_structure = [np.zeros(size=(layer,)).T for layer in layer_structure]
#         self.num_layers = len(layer_structure) - 1
#
#         print(self.layer_structure)
#
#         self.weight_structure = [np.random.random() for _ in range(self.num_layers)]
#
#
#     def sigmoid(self, x):
#         return expit(x)
#
#     def sigmoid_derivative(self, x):
#         return sigmoid(x) * (1 - sigmoid(x))
#
#     def forward_propagation(self, weights, bias):
#         for weights
#
#
#         output = sigmoid(np.dot(self.input_layer, weights) - bias)
#
#         return output



# The goal is to make a 3 to 1 neural net that can detect the ORing of two of the inputs states
def sigmoid(x):
    return expit(x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward_propagation(input_layer, weights, bias):
    output = sigmoid(np.dot(input_layer, weights) - bias)

    return output


def learn(num):
    bias = 5
    weights = np.random.random(size=(3,)).T
    file = open("out.txt", 'w')
    for i in range(num):
        np.savetxt(file, weights)
        file.write('\n')
        input_layer = np.random.randint(low=0, high=2, size=(3,)).T
        output_layer = forward_propagation(input_layer, weights, bias)
        correct_output = input_layer[0] | input_layer[2]
        error = correct_output - output_layer
        delta_weights = np.dot(input_layer, error*2*sigmoid_derivative(output_layer))
        weights += delta_weights
    file.close()
    return weights


def evaluate(weights, bias, num):
    for i in range(num):
        input_layer = np.random.randint(low=0, high=2, size=(3,)).T
        output_layer = forward_propagation(input_layer, weights, bias)
        correct_output = input_layer[0] | input_layer[2]
        error = abs(correct_output - output_layer)
        if error > 0.001:
            print(input_layer)
            print(output_layer)
            print(error)
            print("")


# weights = [18.32235212, -3.82071533, 18.31964812]
# bias = 5
# evaluate(weights, bias, 100)
weights = learn(100000000)
print(weights)
#
# net = NeuralNetwork([3,4,4,1], np.array([1,0,1]).T)
#
