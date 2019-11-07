#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module producing recommendation neural network for random numbers generation."""
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.models import Sequential

import numpy as np

from models.model import CommonModel
from utils import generate_random_bit_array


DEFAULT_BITS_COUNT = 1024


class Recommender(CommonModel):
    """The neural network for recommendation of random number by previous user choices."""
    def __init__(self, bits_count=DEFAULT_BITS_COUNT):
        super().__init__()

        self.is_evaluable = True

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
        """Sequential model with recurrent layers."""
        self._model = Sequential()

        self._model.add(LSTM(self._bits_count,
                             input_shape=(None, self._bits_count),
                             return_sequences=True,
                             name='LSTM-1'))
        self._model.add(Dropout(0.2, name='Dropout-1'))

        self._model.add(LSTM(100, return_sequences=False, name='LSTM-2'))
        self._model.add(Dropout(0.2, name='Dropout-2'))

        self._model.add(Dense(self._bits_count, name='Dense'))
        self._model.add(Activation('hard_sigmoid', name='Activation'))

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

        result = np.array(result)

        data_sets_separate_index = int(round(0.9 * result.shape[0]))

        train_set = result[:data_sets_separate_index:]

        x_train = train_set[:, :-1]
        y_train = train_set[:, -1]
        x_test = result[data_sets_separate_index:, :-1]

        return (x_train, y_train), x_test
