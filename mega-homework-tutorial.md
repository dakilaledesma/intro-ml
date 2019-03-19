# Convolutional Neural Networks
In this assignment, I would not need to go into such detail with building a neural network, as I've explained a lot of it in the previous homework.

I encourage you to read the last homework's reading about pre-processing. It explains some basics as to why you'd want to pre-process your data.

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

## Tutorial
### Convolutional Neural Networks
Instead of using Dense layers, such as ones that you found in the previous homeworks, instead, we're going to be using convolution layers. Convolutions are much better in attaining features on spatial based input such as images.

In fact, you have seen convolutional neural networks before: HW2 used a Keras example convnet in order to train on MNIST images. We can look at their code and decipher what they're doing

```py

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

```

Thanks to the lecture work done by Dr. Liang, I would hopefully not have to explain a lot of what the layers do. However, there are some things that would be helpful for me to explain. Hopefully, the preceding code would look familiar to you in a sense. 

As you can see, it doesn't look much different from what we used to have in the last homework, just with a lot of new layers that are synonymous to ML.

We start with the usual:
Instantiate a serial model using:
```py
model = Sequential()
```

Now, instead of using Dense layers, we use 2D Convolution layers. 2D convolution layers actually take 3D arrays, much like the MNIST data found in the last homework (28 x 28 x 3).
```py
model.add(Conv2D(64, (3, 3), activation='relu'))
```

Much like what you have seen in class, a MaxPooling layer is also employed:
```py
model.add(MaxPooling2D())
```

# Recurrent Neural Networks
The biggest difference between Recurrent NN compared to CNNs is that they have "memory" of the past. So in addition to the inputs that they're given, they're also able to assess how the previous inputs could potentially affect the outcome. For the most part, they're most effective on data that rely on sequences, the simplest examples being text/speech data.

https://hackernoon.com/rnn-or-recurrent-neural-network-for-noobs-a9afbb00e860
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Generative Adversarial Networks 
You will be using a GAN implementation for this homework.
https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
