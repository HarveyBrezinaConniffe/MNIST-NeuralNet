# Written by Harvey Brezina Conniffe in August 2019
# Import neccesary libraries
import numpy as np
from matplotlib.pyplot import imread
import os
import math

import sys
np.set_printoptions(threshold=sys.maxsize)

# -- LOAD DATA -- 

loaded = 0

labelsp = []
imagesp = []
# Loop through directory
for x in os.walk("trainingSet"):
    # Get the last character of the subdirectory, This is the class( A number from 0-9 )
    num = x[0][-1]
    # Ensure it is a number indicating class
    try:
        num = int(num)
    except:
        continue
    # Loop over images in directory
    for y in x[2]:
        if loaded%500 == 0:
            print("LOADED "+str(loaded))
        # Append the label
        labelsp.append(num)
        # Read the image and flatten it
        img = imread(x[0]+"/"+y)
        img = img.reshape([img.size,])
        # Append the image to the overall array.
        imagesp.append(img)
        loaded += 1

labels = np.array(labelsp)
images = np.array(imagesp)

# Normalize the images by subtracting the mean and dividing by the standard deviation
def normalize(arr):
    mean = np.sum(arr)/arr.size
    arr = arr-mean
    arr = arr/np.std(arr)
    return arr

images = normalize(images)

# One-Hot encode the labels, Turn them from a single integer to arrays with a 1 in the position where the class is positive and 0 otherwise. E.G. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
labelsonehot = np.zeros([labels.size, 10])
for x in range(0, labels.size):
    labelsonehot[x][int(labels[x])] = 1

# -- INITIALIZE NEURAL NETWORK --
# Architecture is a fully connected network with 1 hidden layer containing 32 neurons

# The extra weight is for the bias( It will be multiplied by 1 ). A bias allows us to 'shift' the center of the activation rather than just change its shape.
# Weights for input to first hidden layer 0->1
weights0 = np.random.random_sample([32, (28*28)+1])
# Weights for first hidden layer to output 1->2
weights1 = np.random.random_sample([10, 32+1])

# -- DEFINE FUNCTIONS FOR TRAINING --

# Sigmoid function( Makes a very negative X closer and closer to zero, Makes a very positive X closer and closer to 1 )
# When the number gets very negative the exponent becomes positive, this causes e to the x to become very big, thus returning ~1/(VERY LARGE NUMBER) == 0 
# When the number gets vey positive the exponent becomes negative, this causes e to the x to become very small, thus returning ~1/1 == 1
def sigmoid(num):
    return 1/(1+math.exp(-num))
# Vectorize it so it runs on every element of an array
sigmoid = np.vectorize(sigmoid)

# Get the gradient of the sigmoid function -- See backpropagation section below.
def sigmoidgradient(z):
    grad = z*(1-z)
    return grad

# Loss function - Log Loss
def logloss(predictions):
    # Compute the loss as if they were all meant to be positive
    predictionspositive = -np.log(predictions)
    # Compute the loss as if they were all ment to be negative
    predictionsnegative = -np.log(1-predictions)
    # Remove the loss calculated for the positive cases where it was actually negative and vice versa
    predictionspositive = predictionspositive*labelsonehot
    predictionsnegative = predictionsnegative*(1-labelsonehot)
    # Add the two arrays together to get the loss for each prediction
    tloss = predictionspositive+predictionsnegative
    # Return the overall loss
    return np.sum(tloss)

# Accuracy measure, This gives us a % accuracy of the neural network's predictions
def accuracy(predictions):
    c = np.argmax(predictions, axis=1)
    totalmatches = 0
    for i in range(0, len(c)):
        if c[i] == labels[i]:
            totalmatches += 1
    return totalmatches/len(c)

# Run an example through the network to get probabilities for each class.
def forwardpropagation(inp):
    # Add placeholder 1s to the input to multiply the bias weight by.
    inp = np.append(np.ones([inp.shape[0], 1]), inp, axis=1)
    # Multiply each neurons weights by the relevant features to get the values of each neuron in layer 1. NOTE( IF THE INPUT IS A SINGLE EXAMPLE ONLY ): If this was Octave/Matlab we would have to transpose the second 'vector' since they don't have real vectors( It always has 2 dimensions. ), Numpy does however so we don't need to transpose.
    l1 = np.dot(weights0, np.transpose(inp))
    # Get the activation of the neuron by running it through the activation function, Which in this case is the sigmoid function.
    a1 = sigmoid(l1)
    # Add placeholder 1s to the last layer to multiply the bias
    a1 = np.append(np.ones([1, a1.shape[1]]), a1, axis=0)
    # Multiply each neurons weights by the relevant inputs from the first layer to get the values of the output layer.
    l2 = np.dot(weights1, a1)
    # Again we get the activation, This is the final output probabilities for each class
    a2 = sigmoid(l2)
    return [l1, a1, l2, a2, inp]

# Using the error of the output layer, Propagate the error back through the network by multiplying it by the weights of the layer before it to work out how "wrong" each unit in the hidden layer is. From this we work out the gradient updates, Which is the direction and magnitude by which we need to multiply each weight in order to minimize the cost function.
# A lot of this function is derived from the cost function. While I do know the general mechanism of this and how to implement it. I do not have the neccesary calculus knowledge to prove the derivation myself.
def backprop(fprop):
    # Compute the error of the final layer.
    delta2 = np.transpose(np.transpose(fprop[3])-labelsonehot)
    # Propagate it back to the previous layer.
    delta1 = np.dot(np.transpose(weights1), delta2)
    # Remove the placeholder bias 1s as we don't want to update them.
    delta1 = np.delete(delta1, (0), axis=0)
    # Compute the gradient of the activation function.
    gprime = sigmoidgradient(fprop[1])
    # Remove the bias placeholders again.
    gprime = np.delete(gprime, (0), axis=0)
    # Multiply the weight updates by the gradient of the sigmoid function
    delta1 = delta1*gprime
    # Matrix multiply the weight updates by the activation values. 
    D1 = np.dot(delta1, fprop[4])
    D2 = np.dot(delta2, np.transpose(fprop[1]))
    # Scale by 1/DATASET SIZE to get the average gradient
    D1 *= 1/labels.size
    D2 *= 1/labels.size
    # Return gradients for optimization function.
    return [D1, D2]

# The gradient descent optimization function. This updates the weights to minimize the cost function.
def gradientDescent(iterations, learningrate):
    global weights0, weights1
    for i in range(0, iterations):
        # Compute the output of the neural network on the training set.
        fprop = forwardpropagation(images)
        # Run backpropagation on the output to work out the gradient updates.
        backp = backprop(fprop)
        # Update the weights. We multiply the updates by the learning rate to multiply how fast it learns. This will naturally slow as we get closer to the minimum as the gradients magnitudes get smaller.
        weights0 = weights0-(learningrate*backp[0])
        weights1 = weights1-(learningrate*backp[1])
        # Calculate the loss and accuracy on the training set.
        acc = accuracy(np.transpose(fprop[3]))
        loss = logloss(np.transpose(fprop[3]))
        # Print information about training.
        print("Iteration {}, Accuracy: {:.2f}%, Loss: {:.2f}".format(i, acc*100, loss))

gradientDescent(5000, 3)
