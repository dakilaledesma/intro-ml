# Compensation Resource
My deep apologies.

## Tutorial Compensation
### Convolutional Neural Networks
Instead of using Dense layers, such as ones that you found in the previous homeworks, instead, we're going to be using convolution layers. Convolutions are much better in attaining features on spatial based input such as images.
* Basics:
  * https://brohrer.github.io/how_convolutional_neural_networks_work.html
* Deeper Dive:
  * Stanford: http://cs231n.github.io/convolutional-networks/
* Tutorials:
  * Classification: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
  * Autoencoder: https://blog.keras.io/building-autoencoders-in-keras.html
  
#### What does data look like when it propagates through the neural network?
Probably the most helpful resource I can give you to visualize the effects of CNNs is just through this cool visualization:
https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
I very much encourage you to look at the demo above, scrolling down into "network visualization."

### Recurrent Neural Networks
The biggest difference between Recurrent NN compared to CNNs is that they have "memory" of the past. So in addition to the inputs that they're given, they're also able to assess how the previous inputs could potentially affect the outcome. For the most part, they're most effective on data that rely on sequences, the simplest examples being text/speech data.

* Basics:
  * https://hackernoon.com/rnn-or-recurrent-neural-network-for-noobs-a9afbb00e860
* Deeper Dive:
  * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* Tutorial:
  * https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

### Generative Adversarial Networks 
You will be using a GAN implementation for this homework.

* Basics:
  * https://skymind.ai/wiki/generative-adversarial-network-gan
* Code example:
  * https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
