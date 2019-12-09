# CNN classifier for text

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


class CNNClassifier(object):
    def __init__(self,
                 messages,
                 word_embedding,
                 max_sequence_length=1000,
                 max_num_words=20000,
                 embedding_dim=100):
        self.tokenizer = Tokenizer(num_words=max_num_words)
        self.tokenizer.fit_on_texts(messages)
        self.word_index = self.tokenizer.word_index

        # prepare embedding matrix
        self.num_words = min(max_num_words, len(self.word_index) + 1)
        self.embedding_matrix = np.zeros((self.num_words, embedding_dim))
        for word, i in self.word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = word_embedding.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        # initialize model
        embedding_layer = Embedding(self.num_words,
                                    embedding_dim,
                                    embeddings_initializer=Constant(
                                        self.embedding_matrix),
                                    input_length=max_sequence_length,
                                    trainable=False)

        # 1D convnet withh global maxpooling
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedding_sequence = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedding_sequence)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])

    def data_preproccess(self, messages):
        pass

    def fit(self, X, y):
        pass

    def score(self, X, y):
        pass