# Ari Klau (Jan 5th, 2018)
# Steps (and some code) from https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
# Neural Network for taking a matrix derived from a Fast Fourier Transform and determining whether it
# is a snare, kick, or hi-hat

import numpy as np

# Input array
# This will be changed based on how many discrete frequency ranges Grant's FFT code returns.
# Most likely, the number of frequencies will be the number of subarrays, with the frequency's intensity
# represented as a binary number in the subarray.
X=np.array([[1,0,1,0],[1,0,1,1]])

# Output array
# This will also likely need to be changed so that the number of subarrays is the same as the input, but only the
# last two numbers will be used to form a binary number (either 00, 01, or 10; 11 unused) representing snare, kick
# or hi-hat
y=np.array([[1],[1]])

# Sigmoid Function
# May be worth messing around and trying different functions for the backpropagation to see if it yields better results.
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid Function
# Will have to change this if I decide to change the sigmoid function to something else.
def derivatives_sigmoid(x):
    return x * (1 - x)

# Variable initialization
# More cycles is obviously more accurate but also more time consuming. Will have to try a bunch of different orders of magnitude and 
# to optimize time vs. accuracy. If the FFT yields significantly different results for the different types of sounds, it may not be all
# that hard for the neural network to tell which type it is.
cycles = 10000 #Setting training iterations
lr = 50 #Setting learning rate --> messing around and higher seemed to be better, but haven't tried it with any actual data yet
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layer neurons --> Haven't messed with this number yet
output_neurons = 1 #number of neurons at output layer

# Weight and bias initialization
# All random as of now but as I do more research, perhaps I'll find better values to set them to initially
weight_hidden=np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bias_hidden=np.random.uniform(size=(1, hiddenlayer_neurons))
weight_out=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bias_out=np.random.uniform(size=(1,output_neurons))

for i in range(cycles):

    # Forward Propogation
    hidden_layer_input1 = np.dot(X,weight_hidden) #matrix multiplication of input values with hidden layer perceptron weights
    hidden_layer_input = hidden_layer_input1 + bias_hidden #add hidden bias to the input
    hiddenlayer_activations = sigmoid(hidden_layer_input) #apply function
    output_layer_input1 = np.dot(hiddenlayer_activations,weight_out) #matrix multiplication of function output and output perceptron weights
    output_layer_input = output_layer_input1 + bias_out #add output bias to output
    output = sigmoid(output_layer_input) #function again

    # Backpropagation
    # Need to really hammer home the concept of what's going on here... as of now I've just followed online resources.
    # Will definitely be helpful to really understand it if I want to improve it.
    error = y-output #how far NN output was from target output
    slope_output_layer = derivatives_sigmoid(output) 
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = error * slope_output_layer
    Error_at_hidden_layer = d_output.dot(weight_out.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    weight_out += hiddenlayer_activations.T.dot(d_output) * lr
    bias_out += np.sum(d_output, axis=0,keepdims=True) * lr
    weight_hidden += X.T.dot(d_hiddenlayer) * lr
    bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) * lr

print(output)
