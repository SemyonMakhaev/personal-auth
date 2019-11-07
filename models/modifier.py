#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module producing neural network for numbers modification."""
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Input

import numpy as np

from models.model import CommonModel
from utils import generate_random_bit_array


DEFAULT_NUMBER_BITS_COUNT = 1024
DEFAULT_PARAM_BITS_COUNT = 64


class Modifier(CommonModel):
    """
    The neural network using to modify a number with
    a parameter to generate random number.
    """
    def __init__(self,
                 number_bits_count=DEFAULT_NUMBER_BITS_COUNT,
                 param_bits_count=DEFAULT_PARAM_BITS_COUNT):
        super().__init__()

        self.is_evaluable = True

        self._loss = 'mse'
        self._optimizer = 'rmsprop'

        self._epochs = 100
        self._validation_split = 0.05

        self._batch_size = 32
        self._verbose = 1

        self._number_bits_count = number_bits_count
        self._param_bits_count = param_bits_count
        self._examples_count = 10000

    def create(self):
        """Sequential multi-layer perceptron."""
        input_units_count = self._number_bits_count + self._param_bits_count

        input_layer = Input(shape=[input_units_count-1], name='Input')

        dense_1 = Dense(input_units_count, activation='relu', name='Dense-1')(input_layer)
        dropout_1 = Dropout(0.2, name='Dropout-1')(dense_1)

        dense_2 = Dense(self._number_bits_count, activation='relu', name='Dense-2')(dropout_1)
        dropout_2 = Dropout(0.2, name='Dropout-2')(dense_2)

        activation = Activation('linear', name='Activation')(dropout_2)

        self._model = Model([input_layer], activation)

    def get_dataset(self):
        """
        Preparing an array of training examples.
        Each example is a compilation of a number and a parameter.
        """
        result = []

        for _ in range(self._examples_count):
            number = generate_random_bit_array(self._number_bits_count)
            param = generate_random_bit_array(self._param_bits_count)

            number_param_concat = np.concatenate([number, param])

            result.append(number_param_concat)

        result = np.array(result)

        data_sets_separate_index = int(round(0.9 * result.shape[0]))

        train_set = result[:data_sets_separate_index:]

        x_train = train_set[:, :-1]
        x_test = result[data_sets_separate_index:, :-1]

        y_train = np.array([
            generate_random_bit_array(self._number_bits_count) for _ in range(x_train.shape[0])
        ])

        return (x_train, y_train), x_test
