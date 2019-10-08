from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from utils import add_noise
import os
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import random
import tensorflow.keras as keras
from sklearn.utils import shuffle

CHARS = 'aăâbcdefghiîjklmnopqrsștțuvwxyzAĂÂBCDEFGHIÎJKLMNOPQRSȘTȚUVWXYZ-.,!? '

MAX_SEQUENCE = 100
SOS = '<'
EOS = '>'
PAD = '='

char_index = {i+1: c for i, c in enumerate(CHARS)}
char_index[0] = PAD
char_index[len(CHARS) + 1] = SOS
char_index[len(CHARS) + 2] = EOS

index_char = {c: i+1 for i, c in enumerate(CHARS)}
index_char[PAD] = 0
index_char[SOS] = len(CHARS) + 1
index_char[EOS] = len(CHARS) + 2


class Generator(Sequence):
    def __init__(self, text, batch_size, prob=0.15):
        self.y = text
        self.batch_size = batch_size
        self.prob = prob

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def on_epoch_end(self):
        self.y = shuffle(self.y)

    def __getitem__(self, index):
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        x_sentences = []
        y_sentences = []
        for sentence in batch_y:
            x = []
            y = []
            noisy_sentence = add_noise(sentence, self.prob)
            for c in noisy_sentence[:-2]:
                try:
                    x.append(index_char[c])
                except Exception as e:
                    print(e, noisy_sentence)
                    exit()

            x_sentences.append([index_char[SOS]] + x + [index_char[EOS]])
            for c in sentence[:-2]:
                y.append(index_char[c])
            y_sentences.append([index_char[SOS]] + y + [index_char[EOS]])
        x_sentences = pad_sequences(x_sentences, maxlen = MAX_SEQUENCE, dtype = int, padding = 'post', truncating = 'post', value = index_char[PAD])
        y_sentences = pad_sequences(y_sentences, maxlen = MAX_SEQUENCE, dtype = int, padding = 'post', truncating = 'post', value = index_char[PAD])
        # return np.expand_dims(np.array(x_sentences), -1), np.expand_dims(np.array(y_sentences), -1)
        return np.array(x_sentences), np.array(y_sentences)

class DecodeCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, every):
        self.test_generator = test_generator
        self.iteration = 0
        self.every = every

    def on_batch_end(self, *args, **kwargs):
        self.iteration += 1

        if self.iteration % self.every == 0:
            x_sentences, y_sentences = self.test_generator[random.randint(0, len(self.test_generator) - 1)]

            predicted = self.model.predict(x_sentences)

            for clean, dirty, predicted in zip(y_sentences, x_sentences, predicted):
                print('#' * 25)
                print(''.join(char_index[i] for i in dirty if i != index_char[PAD]))
                print(''.join(char_index[i] for i in clean if i != index_char[PAD]))
                print(''.join(char_index[np.argmax(i)] for i in predicted))
                print('#' * 25)

            self.iteration = 0


def make_model(input_shape):
    input = keras.layers.Input(shape = input_shape)
    embedding = keras.layers.Embedding(len(char_index.keys()), 32)(input)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, recurrent_dropout=0.2, return_sequences = True))(embedding)
    attention = keras.layers.TimeDistributed(keras.layers.Dense(256, activation = 'softmax'))(embedding)
    x = keras.layers.Concatenate()([lstm, attention])

    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences = True, recurrent_dropout=0.2))(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(len(char_index.keys()), activation='softmax'))(lstm)
    model = keras.models.Model(input, out)
    return model


def attention_is_all_you_need(input_shape):
    input = keras.layers.Input(shape = input_shape)
    embedding = keras.layers.Embedding(len(char_index.keys()), 32)(input)

    def multi_head(_input, heads=5):
        layers = []
        d = input_shape[-1] // heads
        for i in range(heads):
            crop = keras.layers.Lambda(lambda x: x[:, i * d:(i + 1) * d, :], output_shape=[None, d, _input.shape[-1]])
            attention = keras.layers.TimeDistributed(keras.layers.Dense(_input.shape[-1], activation='softmax'))(crop(_input))

            layers.append(keras.layers.Multiply()([attention, crop(_input)]))
        concatenated = keras.layers.Concatenate(axis=1)(layers)
        return concatenated

    x = keras.layers.TimeDistributed(keras.layers.Dense(32, activation='relu'))(multi_head(embedding, heads=10))
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(multi_head(embedding, heads=20))
    x = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(multi_head(embedding, heads=25))
    x = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu'))(multi_head(embedding, heads=25))
    x = keras.layers.TimeDistributed(keras.layers.Dense(len(char_index.keys()), activation='softmax'))(multi_head(x, heads=25))

    model = keras.models.Model(input, x)
    return model

file_path = 'clean_text.txt'
with open(file_path, 'rt') as f:
    text = f.readlines()

train, test = train_test_split(text)

input_shape = (MAX_SEQUENCE, )
prob = 0.4
batch_size = 64

model = attention_is_all_you_need(input_shape)
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

train_generator = Generator(train, batch_size, 0.4)
test_generator = Generator(test, batch_size, 0.4)

checkpoint_path = './checkpoints/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
os.makedirs(checkpoint_path)

model.fit_generator(
    train_generator,
    validation_data = test_generator,
    epochs=100,
    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./logs/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")),
        keras.callbacks.ModelCheckpoint(checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min'),
        DecodeCallback(test_generator, 50)
    ],
)
