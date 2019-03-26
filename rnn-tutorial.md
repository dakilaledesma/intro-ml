# Recurrent Neural Networks
## Short Reading
### Overview
Recurrent Neural Networks (RNN) differ from CNNs by taking into consideration previous inputs for the current prediction. If in a CNN all previous input is discarded, in an RNN previous inputs are retained in some fashion to help what's being predicted. Because of this trait, they generally perform well on data with temporal features, and generally perform much worse than CNNs for data with only spatial features (e.g. image recognition).

Thus, when it comes to sequences, the first thing that people usually try to employ is a recurrent neural networks, such as an LSTM.
Remember the waveform I used in the CNN tutorial to make a point between 1D and 2D Convolutions?
![singlewaveform](https://i.imgur.com/m9mVQSs.png)

With this type of data, if you're trying to learn the actual waveform, it may be better to use an RNN. This is the case, for example, if you're trying to predict the next value within a time series, such as:
![tsprediction](https://i.imgur.com/1QTZnXV.png)

<sub> image taken w/o permission from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/</sub>

### Applications of RNNs
To help tie this into real world applications, there are papers that use some CNN in order to learn features of the current frame of a video, and use an RNN in order to learn the features from one video frame to the other (essentially the correlations of the previous frames to the current frame).

There are many applications of RNNs today, many of them in natural language processing (NLP). You have LSTM Autoencoders that handle sequence to sequence, and are sometimes used for language translation (sentence to sentence), language models or generative models to generate new words given context, etc.

There is a pretty interesting generative model that is RNN based as well, from Google's DeepMind team: PixelRNN and PixelCNN.

![pixelrnncompletion](https://i.imgur.com/9DDBNVS.png)

The Pixel models work by generating values *pixel by pixel* rather than a computation all at once. Thus, for image completion for example, it is important for the model to know what the previously generated pixels were before computing the next.

However, just because you have sequences does not mean you have to always use an RNN. You have to take into consideration how big of a time horizon you need to take into consideration. For example, if you're going to only predict over a very short time horizon, CNNs may still be used. If you're going to predict over a long time horizon, where long-term dependency is pretty important, then RNNs are a better idea.

An example of this is text or word classification. Yes, a word is a sequence of letters, but just like how an image is a sequence of pixels, you are not trying to learn the temporal relations of each letter, but rather how the entire word looks ("at once"). Thus, there are a lot of models for word classification that are based on CNNs. RNNs, on the other hand, can be seen when whole sentences need to be generated, or translated. This is because each word's relation to each other has to be learnt. Thus, if your model needs to generate the next word in a sentence given the previous words, RNNs are the way to go.

### Backpropagation Through Time (BPTT)
As what I believe what covered in class, because RNNs don't only consider current outputs but also previous outputs, backpropagation is also a little different in RNNs. The main difference between backpropagation in feed-forward neural networks and RNNs is that at each time step, the gradient weight W are summed up.

### LSTMs and GRUs
Even then simple RNNs, such as:

![simplernnunit](https://i.imgur.com/AXlVa2q.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

aren't very good with data that have long-term dependencies, as it suffers from a vanishing or exploding gradient problem during BPTT. In fact, this is the reason why variations such as Long-Short Term Memory (LSTM) or Gated Recurrent Units (GRU) exist.

In contrast, this is what an LSTM unit looks like:

![lstmunit](https://i.imgur.com/XPHFHe1.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

And this is what a GRU unit (a variation on the LSTM unit) looks like

![gruunit](https://i.imgur.com/ySa2X9N.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

LSTMs were created in order to fix this long-term dependency problem, by fixing the vainishing/exploding gradient problem. They do this by allowing the modal neuron to have both a memory gate as well as a forget gate rather than a single layer. Because of their multi-layer repeating module, LSTMs are able to remember a lot more information than their simpler RNN counterparts (with a singular repeating module).

As seen from the above network, input going through an LSTM unit undergo a multistep process:
1. First, a forget gate, that determines whether or not it wants to remember or forget some input on a scale of 0 to 1.
2. Second, we decide which information to remember by the evaluation of candidate values
3. Third, we update the old cell state with the new one.
4. Lastly, we figure out what to output and output it.

This is a pretty heavy summarization of what goes on in an LSTM layer, a slightly more indepth explanation with equations and all can be found here (which is also my images sources): http://colah.github.io/posts/2015-08-Understanding-LSTMs/

<GRU Explanation here> 


### LSTM Deeper Dive
If you'd like to know more about RNNs and LSTM (specifically for MLP) this is a good, multipart resource into these models:

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/ (this is part 1 of the series)

## Letter Sequence Tutorial (undergoing code review)
Today, we're going to be getting a recurrent neural network to learn the relationships between each letter in the alphabet.

## Text Generator Tutorial (needs explanation probably)
This tutorial is handed off to another tutorial by Trung Tran. The objective is to generate text given a dataset, i.e.:
![resultoftg](https://i.imgur.com/n1UhVVX.png)

This was my result training on Alice in Wonderland after 13 Epochs. And as most of you may know, this *is* a sign of overfitting, as it is most likely memorizing words and regenerating them. However, through this tutorial I hope you can see that these types of things are possible.

Code source and intuitional explanations are found at this person's GitHub. It is pretty well explained:
https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

The code had some missing lines and errors, so here is the full working code:
```py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Activation, Dense

DATA_DIR = 'dataset/alice.txt'
SEQ_LENGTH = 100
HIDDEN_DIM = 700
LAYER_NUM = 3
BATCH_SIZE = 12

data = open(DATA_DIR, 'r').read()
chars = list(set(data))
VOCAB_SIZE = len(chars)

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

X = np.zeros((int(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((int(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, int(len(data) / SEQ_LENGTH)):
    X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")


def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


GENERATE_LENGTH = 20
nb_epoch = 0
while True:
    print('\n\n')
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
```
## Sources
1. https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
