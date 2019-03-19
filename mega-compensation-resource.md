# Compensation Resource
My deep apologies.

## Short Reading
### What is a convolutional neural network?
Generally, a supervised convolutional neural network comprises of one or more convolution layers added before a set of fully connected (Dense) layers for classification. Convolutions are in charge of taking features (feature extraction) and new representations of them in order to help the output. 

### How do convolutional neural networks work?
Instead of re-writing everything that I know, here is a fantastic resource (with images) on how CNNs work. The following is both a video and text resource, so if you'd rather read (or vice versa) then you can choose whatever you'd prefer.
https://brohrer.github.io/how_convolutional_neural_networks_work.html

### What does data look like when it propagates through the neural network?
Probably the most helpful resource I can give you to visualize the effects of CNNs is just through this cool visualization:
https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
I very much encourage you to look at the demo above, scrolling down into "network visualization."

### Dropout
In this homework, in addition to our convolution layers (which will be explained below), we're going to be using a Dropout layer in addition to those convolution layers. 

#### What is dropout? 
Simply put, Dropout is a way to reduce overfitting by "dropping out" a random set of neurons during training. This means that the connections to certain neurons are multiplied by 0, rending the connection for that epoch "useless."

#### Why does this work? 
If say, a neural network is stuck within a local minima, it allows the neural network to escape that local minima in hopes that it will find another minima closer to the global minima. In a sense, this makes the input "noisy," when a layer passes it's data onto the next with dropout, an input may look closer to the image on the right rather than the left:

Without Dropout | With Dropout
------------ | -------------
![normal_mnist](https://i.imgur.com/2ayEHKT.png?1) | ![noisy_mnist](https://i.imgur.com/gnmrCLO.png)

<sub> taken w/o permission from: https://cs.stanford.edu/people/karpathy/convnetjs/mnist.png & https://csc.lsu.edu/~saikat/n-mnist/ </sub>

As you can see, the images are a lot noisier. The simplest benefit to Dropout is that it allows the model to become more tolerant of errors. Adding this artificial noise actually allows it to learn from "worse" but still representative data. For example, if you were training on cat images and some of your cat images are partially covered by another object, a dropout model would be more tolerant, and more accurate, then a model without dropout. This is one of the Dropout's biggest benefits against overfitting as well.

## Tutorial Compensation
### Convolutional Neural Networks
Instead of using Dense layers, such as ones that you found in the previous homeworks, instead, we're going to be using convolution layers. Convolutions are much better in attaining features on spatial based input such as images.

Deeper Dive:
http://cs231n.github.io/convolutional-networks/

Tutorials:
* Classification: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
* Autoencoder: https://blog.keras.io/building-autoencoders-in-keras.html

# Recurrent Neural Networks
The biggest difference between Recurrent NN compared to CNNs is that they have "memory" of the past. So in addition to the inputs that they're given, they're also able to assess how the previous inputs could potentially affect the outcome. For the most part, they're most effective on data that rely on sequences, the simplest examples being text/speech data.

Basics:
https://hackernoon.com/rnn-or-recurrent-neural-network-for-noobs-a9afbb00e860

Deeper Dive:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Tutorial:
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

# Generative Adversarial Networks 
You will be using a GAN implementation for this homework.

Basics:
https://skymind.ai/wiki/generative-adversarial-network-gan

Code example:
https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
