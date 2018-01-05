#Ari Klau (Jan 5th, 2018)
#Steps (and some code) from https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
#Neural Network for taking a matrix derived from a Fast Fourier Transform and determining whether it
#is a snare, kick, or hi-hat

import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1]])

#Output
y=np.array([[1],[1]])

#Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
cycles = 10000 #Setting training iterations
lr = 50 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
weight_hidden=np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bias_hidden=np.random.uniform(size=(1, hiddenlayer_neurons))
weight_out=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bias_out=np.random.uniform(size=(1,output_neurons))

for i in range(cycles):

    #Forward Propogation
    hidden_layer_input1 = np.dot(X,weight_hidden)
    hidden_layer_input = hidden_layer_input1 + bias_hidden
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations,weight_out)
    output_layer_input = output_layer_input1 + bias_out
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(weight_out.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    weight_out += hiddenlayer_activations.T.dot(d_output) *lr
    bias_out += np.sum(d_output, axis=0,keepdims=True) *lr
    weight_hidden += X.T.dot(d_hiddenlayer) *lr
    bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print(output)