# Introduction
This is a implementation of a simple Neural Network in vanilla python( Well, I used numpy cause I'm not insane. ). I did this mostly as an excercise to improve my understanding of neural networks. It trains an artificial neural network to classify handwritten digets using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
# How to run it
If you would like to run this yourself:
1. Ensure you have Python3 and numpy installed.
2. Clone this repository.
3. Download the [MNIST as JPG training dataset](https://www.kaggle.com/scolianni/mnistasjpg) and unzip into the directory.
4. Run AI.py
5. You can adjust the learning rate to see the changes in training accuracy.
# Technical details of the Neural Network
* This is a feedforward dense neural network.
* It has one hidden layer with 32 neurons.
* All neurons use the sigmoid activation function.
* The loss function is logloss.
* I use batch gradient descent to update the weights.
# TODO
Some of the features I plan on implementing:
* Add a different activation( Tanh next probably )
* Add a better optimiser( Adam next probably )
* Add some other fun stuff( Batchnorm and droput next )
