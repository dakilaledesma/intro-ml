# Convolutional Neural Networks
In this assignment, I would not need to go into such detail with building a neural network, as I've explained a lot of it in the previous homework.

I encourage you to read the last homework's reading about pre-processing. It explains some basics as to why you'd want to pre-process your data.

## Short Reading
### Dropout
In this homework, in addition to our convolution layers (which will be explained below), we're going to be using a Dropout layer in addition to those convolution layers. 

#### What is dropout? 
Simply put, Dropout is a way to reduce overfitting by "dropping out" a random set of neurons during training. This means that the connections to certain neurons are multiplied by 0, rending the connection useless.

#### Why does this work? 
If say, a neural network is stuck within a local minima, it allows the neural network to escape that local minima in hopes that it will find another minima closer to the global minima. In a sense, this makes the input "noisy," when a layer passes it's data onto the next with dropout, an input may look closer to the image on the right rather than the left:

![normal_mnist](https://i.imgur.com/2ayEHKT.png)
![noisy_mnist](https://i.imgur.com/gnmrCLO.png)

taken w/o permission from: https://cs.stanford.edu/people/karpathy/convnetjs/mnist.png & https://csc.lsu.edu/~saikat/n-mnist/

## Tutorial
### Pre-processing
This tutorial will be more about how to pre-process your work, instead of why you'd want to. In the previous homework, I had this example.

![thresholded strawberry leaf](https://i.imgur.com/rb9n4fM.png)

Now, the question is how can we do something like this?

Because we know that these images are basically turned into 3D arrays like in the last tutorial, where each MNIST digit image was just a 28 x 28 x 3 array per image, we can do this simply by setting certain pixels in the image to black, and retaining the green pixels that correspond to the strawberry leaf.

Think of it this way:


### Convolutional Neural Networks
Instead of using Dense layers, such as ones that you found in the previous homeworks, instead, we're going to be using convolution layers.
