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

#### Local-Response Normalization (to-do)


#### Batch normalization
In addition to using dropout, many convolutional neural networks employ what's called batch normalization. In the case of the image below, you can see that the NN is training on whether the image is a cat or not. However, you may notice that the cats on the left are always black, whilst the cats change color. Yet, in both cases, the model should output y=1.

![batchnorm](https://i.imgur.com/5FAPSgF.png)
<sub> taken w/o permission from: https://www.youtube.com/watch?v=nUUqwaxLnWs </sub>

To put it simply, this shift in color/data values causes a shift in the input distribution. If you trained your model on black cats but your input distribution changes (such as multi-colored cats), you may have to retrain your model. 

Essentially, what batch normalization does is allows what data is propagating through the neural network to have the same mean and variance for each other, minimizing the range of shift that is found within the data.

If you'd like more information on this, here's a link to a video of Andrew Ng's. Some of you may be familiar with him, he is an associate professor at Stanford who teaches a majority of the machine learning classes there:

https://www.youtube.com/watch?v=nUUqwaxLnWs

### Deeper Dive
If you would like to know more about convolutional neural networks than what is currently explained, here is a Stanford resource regarding convolutional neural networks in the context of computer vision:

http://cs231n.github.io/convolutional-networks/

It's quite a long read, but <sub> it's written better than anything I'll be ever able to write (ngl) </sub>

## Tutorial
From what you've seen in lectures, much of convolutional neural networks are comprised of some defining features: convolutions, pooling, and a fully connected layer.

### Classification tutorial (needs explaining)
Currently 
```py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()
training_data = idg.flow_from_directory('dataset/train',
                                        target_size=(32, 32),
                                        batch_size=16,
                                        class_mode='binary')

testing_data = idg.flow_from_directory('dataset/test',
                                       target_size=(32, 32),
                                       batch_size=16,
                                       class_mode='binary')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(training_data, testing_data,
          batch_size=16,
          epochs=256)
```

### Conv Autoencoder tutorial (needs explaining)
An autoencoder can be split into three parts, the encoder and decoder. In an autoencoder, 'loss' is the computed reconstruction loss determined through the difference between your encoded representation (compressed) and your decoded representation (decompressed). 

Today two interesting practical applications of autoencoders are data denoising (which we feature later in this post), and dimensionality reduction for data visualization. With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.

One of the most obvious uses for autoencoders is the denoising of data. The noisy data is encoded into a compressed form, and from this compressed from it must reconstruct a non-noisy image. In addition, being an unsupervised network, more complicated autoencoders have their own successes in the generation of data (such as for words/NLP) despite not being a recurrent neural network.

We'll be using autoencoders for MNIST to do things like this:
![noisymnist](https://blog.keras.io/img/ae/denoised_digits.png)

<sub> Code source: https://blog.keras.io/building-autoencoders-in-keras.html </sub>
```py
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
```
