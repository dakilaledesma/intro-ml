# Recurrent Neural Networks
## Short Reading
Recurrent Neural Networks (RNN) differ from CNNs by taking into consideration previous inputs for the current prediction. Because of this trait, they generally perform well on data with temporal features, and generally perform much worse than CNNs for data with only spatial features (e.g. image recognition).

Thus, sequence to sequence or sequence generation are usually done using recurrent neural networks, such as LSTM.
Remember the waveform I used in the CNN tutorial to make a point between 1D and 2D Convolutions?
![singlewaveform](https://i.imgur.com/m9mVQSs.png)

This type of data, if you're trying to learn the actual waveform, may be better learnt through a RNN. This is the case, for example, if you're trying to predict the next value within a time series, such as:
![tsprediction](https://i.imgur.com/1QTZnXV.png)

<sub> image taken w/o permission from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/</sub>


Say for example you have a double waveform seen, you may want to consider a 2D convolution instead. You can visualize the below as a a list of list of scalar values, such as [[2, 5], [4, 8], ...]. Or, you can separate the two waveforms and train two separate 1D Convolution networks *if* the waveforms are highly divergent or have little correlation. You can tell when this happens if your neural network fails to optimize properly.
## Deeper Dive
If you'd like to know more about RNNs, here's a good resource for how they work:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
