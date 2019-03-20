# Convolutional Neural Networks
## Short Reading
### What is a convolutional neural network?
Generally, a supervised convolutional neural network comprises of one or more convolution layers added before a set of fully connected (Dense) layers for classification. Convolutions are in charge of taking features (feature extraction) and new representations of them in order to help the output. 

### Explanation
In case convolutional neural networks were not explained well enough in lecture, or if you would like to know more, here is an excellent resource:

https://brohrer.github.io/how_convolutional_neural_networks_work.html

The link above has videos and images to help aid the explanation of this.


### Dropout
In this homework, in addition to our convolution layers (which will be explained below), I would like to introduce a concept called Dropout.

#### What is dropout? 
Simply put, Dropout is a way to reduce overfitting by "dropping out" a random set of neurons during training. This means that the connections to certain neurons are multiplied by 0, rending the connection for that epoch "useless."

#### Why does this work? 
If say, a neural network is stuck within a local minima, it allows the neural network to escape that local minima in hopes that it will find another minima closer to the global minima. In a sense, this makes the input "noisy," when a layer passes it's data onto the next with dropout, an input may look closer to the image on the right rather than the left:

Without Dropout | With Dropout
------------ | -------------
![normal_mnist](https://i.imgur.com/2ayEHKT.png?1) | ![noisy_mnist](https://i.imgur.com/gnmrCLO.png)

<sub> taken w/o permission from: https://cs.stanford.edu/people/karpathy/convnetjs/mnist.png & https://csc.lsu.edu/~saikat/n-mnist/ </sub>

As you can see, the images are a lot noisier. The simplest benefit to Dropout is that it allows the model to become more tolerant of errors. Adding this artificial noise actually allows it to learn from "worse" but still representative data. For example, if you were training on cat images and some of your cat images are partially covered by another object, a dropout model would be more tolerant, and more accurate, then a model without dropout. This is one of the Dropout's biggest benefits against overfitting as well.

### Normalization
In addition to Dropout, various normalization techniques have become pertinent in the accuracies of state-of-the-art models.

#### Local-Response Normalization


#### Batch normalization
In addition to using dropout, many convolutional neural networks employ what's called batch normalization. In the case of the image below, you can see that the NN is training on whether the image is a cat or not. However, you may notice that the cats on the left are always black, whilst the cats change color. Yet, in both cases, the model should output y=1.

![batchnorm](https://i.imgur.com/5FAPSgF.png)
<sub> taken w/o permission from: https://www.youtube.com/watch?v=nUUqwaxLnWs </sub>

To put it very simply, this shift in color causes what is known as covariate shift.

If you'd like more information on this, here's a link to a video of Andrew Ng's. Some of you may be familiar with him, he is an associate professor at Stanford who teaches a majority of the machine learning classes there:

https://www.youtube.com/watch?v=nUUqwaxLnWs

### Deeper Dive
If you would like to know more about convolutional neural networks than what is currently explained, here is a Stanford resource regarding convolutional neural networks in the context of computer vision:

http://cs231n.github.io/convolutional-networks/

It's quite a long read, but <sub> it's written better than anything I'll be ever able to write (ngl) </sub>

## Tutorial
From what you've seen in lectures, much of convolutional neural networks are comprised of 
