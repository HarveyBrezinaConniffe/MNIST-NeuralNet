# Introduction
This is a implementation of a simple Neural Network in vanilla python( Well, I used numpy cause I'm not insane. ). I did this mostly as an excercise to improve my understanding of neural networks. It trains an artificial neural network to classify handwritten digets using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
# How to run it
If you would like to run this yourself:
1. Ensure you have Python3 and numpy installed.
2. Clone this repository.
3. Download the [MNIST as JPG sample dataset](https://www.kaggle.com/scolianni/mnistasjpg) and unzip into the directory.
4. Run AI.py
5. You can adjust the learning rate to see the changes in training accuracy.
# Technical details of the Neural Network
* This is a feedforward dense neural network.
* It has one hidden layer with 32 neurons.
* All neurons use the sigmoid activation function.
* The loss function is logloss.
* I use batch gradient descent to update the weights.
# Is this a state of the art implementation with no flaws?
No, Not at all. In fact, Here is a list of all the flaws:
* I use batch gradient descent. This means I accumulate the gradient updates over the whole training set before preforming a single update. This won't scale well, This is why I am using the sample of MNIST and not the whole thing.
* It's probably wasteful to use MNIST as JPGs rather than their original format but I was more comfortable with it.
* I have no regularization so there is probably overfitting happening.
* Sigmoid is probably not the best activation to use but it is nice and simple.
* You would probably be better off using a convolutional neural network but this was mostly just for educational purposes so I went with a simpler structure.
* I mostly arbitrarily chose the hyperparameters( No. of neurons in the hidden layer. No. hidden layers. Etc. )
