# CNN classifier for text

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from keras import regularizers

from BaseClassifier import BaseClassifier


class CNNClassifier(BaseClassifier):

    def __init__(self,
                 messages,
                 labels,
                 word_embedding,
                 max_sequence_length=1000,
                 max_num_words=20000,
                 embedding_dim=100):

        # initialize base class
        super(CNNClassifier,
              self).__init__(messages=messages,
                             labels=labels,
                             word_embedding=word_embedding,
                             max_sequence_length=max_sequence_length,
                             max_num_words=max_num_words,
                             embedding_dim=embedding_dim)

        # 1D convnet withh global maxpooling
        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedding_sequence = self.embedding_layer(sequence_input)
        x = Conv1D(
            128,
            5,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001))(embedding_sequence)
        x = BatchNormalization()(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128,
                   5,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128,
                   5,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(x)
        preds = Dense(len(self.le.classes_), activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])
