# Assignment 3

## Part 1: Pre-processing
I’d like to emphasize a fundamental concept in data science: pre-processing.

Machine learning is not magic, as most of you already know. Your neural network will perform not as well as it should if you mindlessly throw your training data into it (it will probably perform quite poorly). Many of you may have heard of a phrase similar to “feed a neural network garbage, and you will get a garbage neural network.” A lot of what makes a good neural network is not the neural network architecture itself -- it’s the data that you use to train it. In simpler terms, we need to make our data as “intuitive” as possible for the machine to learn by removing things that are unnecessary to a prediction. This entails determining what part of the data you would consider important, and what part of the data is not.

Let’s take a very simple example: I would like to create a machine learning algorithm to detect the leafy part of a strawberry. Let's assume that the images I will test/validate on *only* has strawberries.

![strawberry leaf](https://i.imgur.com/WnsH1fm.jpg)

Now I would be able to just throw this exact image above into a model, telling the model “this is the leafy part of a strawberry." However, I can do a simple threshold of the image to only keep certain colors within the image, and get something like this:

![thresholded strawberry leaf]https://i.imgur.com/rb9n4fM.png

As you can see, I’ve thresholded colors in a way that everything except the greens of the image are simply turned black. Let’s assume that through the feature extraction, what the neural network ends up “seeing” is the edges of the picture:

Unprocessed | Processed
------------ | -------------
![unprocessed edge](https://i.imgur.com/O16cN9k.png) | ![processed edge](https://i.imgur.com/0zOprgZ.png)

Unprocessed, there seems to be a lot of edges that are not even part of the leafy part of the strawberry. But on the right, we can see that most of that noise is gone. Just by thresholding certain colors, we were able to get an image without unnecessary data (such as edges that aren’t part of the leaves of a strawberry).

Removing unnecessary data can be very beneficial for how your model performs. This example is very simple, but I’d like to emphasize this so you may keep this in mind whenever you’re training your models.

## Part 2: Hyperparameters
You may have noticed word “parameters” occasionally been thrown left and right in class (don’t quote me on that). While internal parameters are parameters that are set during training (such that their values change as the neural network trains), hyperparameters are parameters that are set before training. 

In the tutorial homework 2, you may have noticed that I used a learning rate of 1 for the simple gradient descent optimizer I showed. Learning rate in this scenario is a hyperparameter, as it doesn’t change during training. In optimizers the have variable learning rate values during training, the initial learning rate value is also a hyperparameter, as it is set by you and while learning rate changes during training, the initial learning rate value doesn’t change.

Some common hyperparameters you may encounter during coding:
* Learning rate
* Number of neurons per layer
* Filter size (convolutions)
* Activation function per layer
* Number of convolutions
* Dimensionality of data

Why is this important? Depending on the hyperparameters you choose before even training your neural network may drastically improve (or degrade) the performance of your neural network. That being said, you should pick hyperparameters that make sense for the goal that you’re trying to achieve. This should go without saying, but sometimes this is not something completely intuitive to think about.

For example, a look at the sigmoid activation function below. It is apparent that the steepness of the curve between -2 and 2 are much steeper relative to the steepness from -inf to 4, and 4 to inf.

When using this in binary classification, where you’re trying to output to -1 or 1, this is great! This looks like a smooth step-function, and when data is passed through this activation function, data in the middle having steep differences in value compared to the extremities, in this case closer to 1 or 0. This makes the data less ambiguous when categorizing, for example.

You can kind of visualize what a sigmoid does to data through these images**:

![sig-images](https://i.imgur.com/IKoJG8I.png)


<sub> ** Note: Not really accurate, but is used as a visual analogy. Image from http://ccis2k.org/iajit/PDF/vol.1,no.2/10-nagla.pdf </sub>

As you can see, the pixels that are already very dark *don't* change value a lot, but the pixels within the image that are roughly halfway in the middle of black and white in the original image get pushed closer to the extremities (in this case, they turn whiter).

Using sigmoids is good in this scenario, but you do not always want to do this. If you’re not predicting binary classifications but rather regressions, using sigmoids doesn't make a lot of sense.

Take my one of my research projects that focuses on human motion data. Let's assume that in the GIFs you see below, the pixels from the left to right represent some x-values. We are then taking the x-values of my foot and putting it through a hypothetical sigmoid:

"Linear" Activation | "Sigmoid" Activation
------------ | -------------
![norm](https://i.imgur.com/HTLcaSJ.gif) | ![sig](https://i.imgur.com/6TWh0QF.gif)

<sub> Note: Do not @ me about my legs or leg day </sub>

Again, this is not really accurate to what would actually happen to the data, but hopefully this is a good enough example as to why you wouldn't want to use something like a sigmoid for this type of data. You're not trying to classify between two things when generating a sequence of motion data. In "sigmoid" GIF, you may notice how my foot motion ends up on either the very left or the very right most of the time, but motion in the middle is quick, almost non-existent. As you can see, the "sigmoid" motion looks a lot less natural than the unprocessed/"linear" version. If your goal is to generate natural walking data, then applying this kind of activation function to your data is like giving your neural network bad information to learn from, degrading the performance. 

## Part 3: Homework Tutorial

Okay, let's do some coding! We're going to be making an MLP neural network in Keras, and doing hyperparameter grid search using Talos. We aren't going to be doing pre-processing to our data today, but we will in the next homework.

### Required library
Before we start, please install talos. This is the library that we will be using in order to do hyperparameter grid search. It has a lot of features, such as plotting of neural network performance for each change of a hyperparamter. More information can be found here: ![Talos](https://github.com/autonomio/talos)

Using Anaconda Prompt (Windows), or terminal (macOS or Linux), activate your python environment and type this to install this library to your python interpreter. If you're lost, you may refer to the machine setup guide where you do roughly the same thing.

```
pip install talos
```

### Defining the neural network

For people who have experience coding using Keras, this may be a little different. We'll be wrapping our model into a function because of how Talos does its hyperparameter grid search. As you may know, this is not necessary normally (though may be recommended). The code may also be a little different from what you may already be doing.

#### Imports

Let's import what we need

```python
import sys
import talos
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.activations import sigmoid, softmax, relu, elu, linear
from keras.datasets import mnist
from keras import backend
```

#### Dataset

We'll be using MNIST for our dataset. This is similar to homework 2, where the dataset is downloaded through the code itself (meaning you do not need to provide your own images or dataset). 

Thus, let's go get the data (taken from the Keras example):
```python
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```
#### Building the neural network

Now that we have our data, let's build a neural network using Keras. We're going to be making a serial model today, and the easiest way to do that in Keras is using Sequential().

Sequential works in this manner:

First, instantiate the model. We can call this easily as we've imported it above.

```python
model = Sequential()
```

Then, you can easily add layers to the model by calling model.add(). Some examples of layers are
* Dense (the simplest type of layer, essentially a neuron with a weight, bias, and activation attached)
* LSTM (classified as an recurrent neural layer with memory for temporal data)
* Conv2D (classified as a convolutional neural layer, in which CNNs excel in spatial data)
* MaxPooling (commonly used in CNNs)

In this tutorial, we're going to be making an MLP neural network, the simplest kind. Thus, we will be adding a Dense layers in our neural network.

Before we do that, let's add a Flatten() layer, as Dense can only take vectors/one-dimensional arrays. Right now, through the code we used to fetch and process the MNIST dataset, each image in the dataset is actually a 3D array of this format: (28, 28, 1), where
* 28, - represents the row of pixels
* 28, - represents the column of pixels
* 1 - represents the single integer that denotes the luminosity of the pixel (0 for black, 255 for white, and gray in between)

The flatten layer just flattens this 3-dimensional array representing an image into a single dimension or a vector, So instead of a 28 x 28 x 1 array, you have an array of length 768, with each element in that array being the luminosity of the pixel. The reason why we're doing this is because Dense layers can only take a vector or single dimension array as input. 

```python
model.add(Flatten(input_shape=(28, 28, 1)))
```
Inside our Flatten(), we add input_shape. input_shape is a parameter you need to set in the first layer that you add(). It should represent the shape that the training data has.

Other layers found in Keras, such as Conv2D, can take a 3D array (2D for the row x column, 1D for the values). I will explain that more in-depth in the next homework. 

Now, let's add the Dense layers:

```python
model.add(Dense(units=24, activation='softmax'))
model.add(Dense(units=10, activation='softmax'))
```
Where:
* Units represents the amount of neurons per layer
* Activation represents the activation function used in that layer

After adding all of your desired layers, let's compile the model using model.compile().
```python
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
```
Where:
* Loss represents the loss used by the model
* Optimizer represents the optimizer used by the model
* Metrics specifies the metrics

Lastly, let's fit our training and testing data into the model we compiled using model.fit().
```python
out = model.fit(X=x_train, y=y_train, batch_size=2000, epochs=100, verbose=0)
```
Where:
* X represents training data
* y represents your label
* Batch size represents the size of your mini-batch
* Epochs represents the number of iterations you are doing for training

We've built the neural network! Your code finished should look something like this:
```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(params['first_neuron'], activation=params['activation']))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X=x_train, y=y_train, batch_size=2000, epochs=100)
```
Running this code should entail your neural network training on the MNIST dataset.

### Doing hyperparameter grid search
We've discussed a few hyperparamters in the previous section, namely:
* number of units
* Activation function
* Loss
* Optimizer
* Batch Size
* number of epochs

And for each, there are many choices. Let's say you don't have any intuition for what activation function between softmax, sigmoid, and ReLU will perform best on your dataset. What you can do is a hyperparameter grid search. This is an automated way of trying every combination of loss, optimizer, # units, etc. that you specify. 

```python
p = {
    'first_neuron': [12, 24],
    'activation': ['softmax', 'sigmoid', 'relu'],
    'loss': ['mse', 'mae', 'binary_crossentropy']
    'optimizer': ['adam', 'adagrad']
}
```

As you can see, we set first_neuron number, activation, loss, and optimizer params. This gives us 2 x 3 x 3 x 2 = 36 different combinations of these hyperparamters.
