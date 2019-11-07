#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module producing limitation neural network for random numbers choice."""
from random import random
from keras.layers import Input, Activation, Dense, Dropout, LSTM, merge
from keras.models import Model

import numpy as np

from models.model import CommonModel
from utils import generate_random_bit_array


DEFAULT_BITS_COUNT = 1024


class Limiter(CommonModel):
    """
    The neural network using to recommend a set of cryptographically
    random numbers by previous choices.
    """
    def __init__(self, bits_count=DEFAULT_BITS_COUNT):
        super().__init__()

        self._loss = 'mse'
        self._optimizer = 'rmsprop'

        self._epochs = 100
        self._validation_split = 0.05

        self._batch_size = 32
        self._verbose = 1

        self._bits_count = bits_count
        self._examples_count = 100
        self._example_size = 50

    def create(self):
        """
        Deep sequential neural network.
        Includes LSTM recurrent layer for previous user choices analyzing
        and an input for a number to predict probability of using it.
        """
        history_input = Input(shape=[self._example_size - 1, self._bits_count], name='HistoryInput')
        history_lstm = LSTM(self._bits_count,
                            return_sequences=False,
                            name='HistoryLSTM')(history_input)

        number_input = Input(shape=[self._bits_count], name='NumberInput')

        common_concat = merge.concatenate([history_lstm, number_input], name='CommonConcat')
        common_dropout = Dropout(0.2, name='CommonDropout')(common_concat)
        common_dense = Dense(1, name='CommonDense')(common_dropout)

        activation = Activation(activation='sigmoid', name='Activation')(common_dense)

        self._model = Model([history_input, number_input], activation)

    def get_dataset(self):
        """
        Preparing an array of examples.
        Each example is entertained by a set of random values.
        """
        result = []

        for i in range(self._examples_count):
            example = [
                generate_random_bit_array(self._bits_count) for _ in range(self._example_size)
            ]
            result.append(example)

        numbers_train_set = [
            generate_random_bit_array(self._bits_count) for _ in range(self._examples_count)
        ]
        probabilities_train_set = [random() for _ in range(self._examples_count)]

        result = np.array(result)
        numbers_train_set = np.array(numbers_train_set)
        probabilities_train_set = np.array(probabilities_train_set)

        data_sets_separate_index = int(round(0.9 * result.shape[0]))

        train_set = result[:data_sets_separate_index:]

        x_lstm_train = train_set[:, :-1]
        x_numbers_train = numbers_train_set[:data_sets_separate_index:]

        x_lstm_test = result[data_sets_separate_index:, :-1]
        x_numbers_test = numbers_train_set[data_sets_separate_index:]

        x_train = [x_lstm_train, x_numbers_train]
        y_train = probabilities_train_set[:data_sets_separate_index:]
        x_test = [x_lstm_test, x_numbers_test]

        return (x_train, y_train), x_test
