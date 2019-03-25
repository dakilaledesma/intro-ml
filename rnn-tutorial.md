# Recurrent Neural Networks
## Short Reading
Recurrent Neural Networks (RNN) differ from CNNs by taking into consideration previous inputs for the current prediction. If in a CNN all previous input is discarded, in an RNN previous inputs are retained in some fashion to help what's being predicted. Because of this trait, they generally perform well on data with temporal features, and generally perform much worse than CNNs for data with only spatial features (e.g. image recognition).

To help tie this into real world applications, there are papers that use some CNN in order to learn features of the current frame of a video, and use an RNN in order to learn the features from one video frame to the other (essentially the correlations of the previous frames to the current frame).

Thus, when it comes to sequences, the first thing that people usually try to employ is a recurrent neural networks, such as an LSTM.
Remember the waveform I used in the CNN tutorial to make a point between 1D and 2D Convolutions?
![singlewaveform](https://i.imgur.com/m9mVQSs.png)

With this type of data, if you're trying to learn the actual waveform, it may be better to use an RNN. This is the case, for example, if you're trying to predict the next value within a time series, such as:
![tsprediction](https://i.imgur.com/1QTZnXV.png)

<sub> image taken w/o permission from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/</sub>

However, just because you have sequences does not mean you have to always use an RNN. You have to take into consideration how big of a time horizon you need to take into consideration. For example, if you're going to only predict over a very short time horizon, CNNs may still be used. If you're going to predict over a long time horizon, where memory is pretty important, then RNNs may probably be a better idea.

An example of this is text or word classification. Yes, a word is a sequence of letters, but just like how an image is a sequence of pixels, you are not trying to learn the temporal relations of each letter, but rather how the entire word looks ("at once"). Thus, there are a lot of models for word classification that are based on CNNs. RNNs, on the other hand, can be seen when whole sentences need to be generated, or translated. This is because each word's relation to each other has to be learnt.

Even then RNNs aren't very good with data that have long-term dependencies. In fact, this is the reason why LSTM was created.

LSTMs were created in order to fix this long-term dependency problem. They do this by allowing the modal neuron to have both a memory gate as well as a forget gate rather than a single layer. Because of their multi-layer repeating module, LSTMs are able to remember a lot more information than their simpler RNN counterparts (with a singular repeating module).

In addition, you may have heard of Gated Recurrent Units (GRUs) as well. GRUs are a variation on 

### LSTM Deeper Dive
If you'd like to know more about RNNs and specifically LSTMs, here's a good resource for how they work and some of the mathematics behind them:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Letter Sequence Tutorial
Today, we're going to be getting a recurrent neural network to learn the relationships between each letter.

## Tiny Language Learner Tutorial
