# CNN-GRU classifier for text

from __future__ import print_function

from keras.layers import Dense, Input, Dropout
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

from BaseClassifier import BaseClassifier
from AttentionLayer import AttentionLayer


class GRUClassifier(BaseClassifier):
    def __init__(self,
                 messages,
                 labels,
                 word_embedding,
                 max_sequence_length=1000,
                 max_num_words=20000,
                 embedding_dim=100,
                 gru_output_size=70,
                 use_attention_layer=False):
        # initialize base class
        super(GRUClassifier,
              self).__init__(messages=messages,
                             labels=labels,
                             word_embedding=word_embedding,
                             max_sequence_length=max_sequence_length,
                             max_num_words=max_num_words,
                             embedding_dim=embedding_dim)

        # initialize model
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedding_sequence = self.embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedding_sequence)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Bidirectional(GRU(gru_output_size, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        if use_attention_layer:
            x = AttentionLayer()(x)

        preds = Dense(len(self.le.classes_), activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])
