# Assignment 3

## Part 1: Pre-processing
I’d like to emphasize a fundamental concept in data science: pre-processing.

Machine learning is not magic, as you may have learnt during lecture. Most of the time your algorithm will perform not as well as it should (or even horribly) if you mindlessly throw your training data into it. You may have heard of a phrase similar to “feed a neural network garbage, and you will get a garbage neural network.” A lot of what makes a good neural network is not the neural network architecture itself -- it’s the data that you use to train it. In simpler terms, we need to make our data as “intuitive” as possible for the machine to learn by removing things that are unnecessary to a prediction. This entails determining what part of the data you would consider important, and what part of the data is not.

Let’s take a very simple example: I would like to create a machine learning algorithm to detect the leafy part of a strawberry.

![img1](https://i.imgur.com/WnsH1fm.jpg)


Now I would be able to just throw this exact image above into a model, telling the model “this is the leafy part of a strawberry. However, I can do a simple threshold of the image to only keep certain colors within the image, and get something like this:

As you can see, I’ve thresholded colors in a way that everything except the greens of the image are simply turned black. Let’s assume that through the feature extraction, what the neural network ends up “seeing” is the edges of the picture:

Unprocessed | Processed
------------ | -------------
![img1](https://i.imgur.com/O16cN9k.png) | ![img1](https://i.imgur.com/0zOprgZ.png)

Unprocessed, there seems to be a lot of edges that are not even part of the leafy part of the strawberry. But on the right, we can see that most of that noise is gone. Just by thresholding certain colors, we were able to get an image without unnecessary data (such as edges that aren’t part of the leaves of a strawberry).

Removing unnecessary data can be very beneficial for how your model performs. This example is very simple, but I’d like to emphasize this so you may keep this in mind whenever you’re training your models.

## Part 2: Hyperparameters
You may have noticed word “parameters” occasionally been thrown left and right in class (don’t quote me on that). While internal parameters are parameters that are set during training (such that their values change as the neural network trains), hyperparameters are parameters that are set before training. 

In the tutorial homework 2, you may have noticed that I used a learning rate of 1 for the simple gradient descent optimizer I showed. Learning rate in this scenario is a hyperparameter, as it doesn’t change during training. In optimizers the have variable learning rate values during training, the initial learning rate value is also a hyperparameter, as it is set by you and while learning rate changes during training, the initial learning rate value doesn’t change.

Some common hyperparameters you may encounter during coding:
Learning rate
Number of neurons per layer
Filter size (CNNs)
Activation function per layer
Number of convolutions
Dimensionality of input

Why is this important? Depending on the hyperparameters you choose before even training your neural network may drastically improve (or degrade) the performance of your neural network. That being said, you should pick hyperparameters that make sense for the goal that you’re trying to achieve. This should go without saying, but sometimes this is not something completely intuitive to think about.

For example, a look at the sigmoid activation function below. It is apparent that the steepness of the curve between -2 and 2 are much steeper relative to the steepness from -inf to 4, and 4 to inf.

When using this in a classification neural network, where you’re trying to output categories of data, this is great! This looks like a smooth step-function, and when data is passed through this activation function, data in the middle gets pushed closer to the extremities, in this case closer to 1 or 0. This makes the data less ambiguous when categorizing, for example.

You can kind of visualize what a sigmoid does to data through these images*:

*Note: Not completely accurate, but is used as a visual analogy of what sigmoid does to a data.

As you can see, the pixels that are already very dark change value a lot, but the pixels within the image that are roughly halfway in the middle of black and white in the original image get pushed to the extremities (in this case, they turn whiter).

Using sigmoids is good in this scenario, but you do not always want to do this. If you’re not predicting classifications but rather regressions, using sigmoids may actually cause your model perform worse.

For example, sigmoids may not be a wise choice for sequence data. Take my research that focuses on human motion data. If I pass the cartesian coordinate x into a sigmoid, I may get something like this**:

Unprocessed | Sigmoided
------------ | -------------
![norm](https://i.imgur.com/HTLcaSJ.gif) | ![img1](https://i.imgur.com/6TWh0QF.gif)

** <sub> Do NOT @ me about my legs </sub>

Again, this is not totally accurate, but this should give you an idea as to why you wouldn't want to sigmoid this sequential data. You wouldn't want to push the x-values of the data to its extremities, as this could cause very unnatural movement (as seen above). Whilst pushing data closer to a step function is helpful for categorization/classification of data, it is not always helpful for sequential data. 

## Part 3: Homework Tutorial

Okay, let's do some coding! We're going to be making an MLP neural network in Keras, and doing hyperparameter grid search. We aren't going to be doing pre-processing today, but we will in the next homework.

### Defining the neural network

For people who have coded in Keras before, this may be a little different. We'll be wrapping our model into a function because of the hyperparameter grid search. As you may know, this is not necessary normally (though recommended). The code may also be a little different from what you may already be doing.

#### Imports

Let's import what we need

#### Dataset

We'll be using MNIST for our dataset. This is similar to homework 2, where the dataset is downloaded through the code itself (meaning you do not need to provide your own images or dataset).


### Doing hyperparameter grid search

We'll be using Talos for our grid search.
