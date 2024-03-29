# CNN classifier for text

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Embedding
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder


class BaseClassifier(object):

    def __init__(self,
                 messages,
                 labels,
                 word_embedding,
                 max_sequence_length=1000,
                 max_num_words=20000,
                 embedding_dim=100):
        self.tokenizer = Tokenizer(num_words=max_num_words)
        self.tokenizer.fit_on_texts(messages)
        self.word_index = self.tokenizer.word_index
        self.le = LabelEncoder()
        self.le.fit(labels)

        # prepare embedding matrix
        self.max_sequence_length = max_sequence_length
        self.num_words = min(max_num_words, len(self.word_index) + 1)
        self.embedding_matrix = np.zeros((self.num_words, embedding_dim))
        for word, i in self.word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = word_embedding.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        # initialize model
        self.embedding_layer = Embedding(self.num_words,
                                         embedding_dim,
                                         embeddings_initializer=Constant(
                                             self.embedding_matrix),
                                         input_length=max_sequence_length,
                                         trainable=False)

    def data_preproccess(self, messages, labels):
        sequences = self.tokenizer.texts_to_sequences(messages)
        data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        labels = to_categorical(self.le.transform(labels))
        return data, labels

    def fit(self, X, y, epochs=10, validation_split=0.1):
        # split data into a training set and a validation set
        num_validation_samples = int(validation_split * X.shape[0])
        x_train = X[:-num_validation_samples]
        y_train = y[:-num_validation_samples]
        x_val = X[-num_validation_samples:]
        y_val = y[-num_validation_samples:]

        self.model.fit(x_train,
                       y_train,
                       batch_size=128,
                       epochs=epochs,
                       validation_data=(x_val, y_val))

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def score(self, X, y):
        scores = self.model.evaluate(X, y, batch_size=128, verbose=0)
        return scores

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)
