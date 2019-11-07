#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module producing common abstract neural network model."""
from abc import ABCMeta, abstractmethod
from keras.models import load_model
from keras.utils import plot_model


class CommonModel(metaclass=ABCMeta):
    """Abstract neural network model wrapper."""
    def __init__(self):
        self.is_evaluable = False

        self._bits_count = None
        self._batch_size = None
        self._epochs = None
        self._loss = None
        self._model = None
        self._optimizer = None
        self._validation_split = None
        self._verbose = None

    @abstractmethod
    def create(self):
        """Model layers description."""

    def compile(self):
        """Model compilation."""
        if not self._model:
            raise ReferenceError('Model has not been created')

        self._model.compile(loss=self._loss, optimizer=self._optimizer)

    def fit(self, dataset):
        """Train the model."""
        if not dataset:
            raise ReferenceError('Dataset is not provided')

        (x_train, y_train), x_test = dataset

        if not self._model:
            raise ReferenceError('Model has not been created')

        if not self._batch_size \
                or not self._epochs \
                or not self._validation_split \
                or not self._verbose:
            raise ReferenceError('Train parameters has not been specified')

        try:
            self._model.fit(x_train,
                            y_train,
                            batch_size=self._batch_size,
                            epochs=self._epochs,
                            validation_split=self._validation_split,
                            verbose=self._verbose)
        except KeyboardInterrupt:
            print('Training interrupted')

    def load(self, filename):
        """Load model from file."""
        if not filename:
            raise ValueError('Filename is not specified')

        self._model = load_model(filename)

    def plot(self, filename):
        """Plot a model schema and write it to the file."""
        if not self._model:
            raise ReferenceError('Model has not been created')

        if not filename:
            raise ValueError('Filename is not specified')

        plot_model(self._model, to_file=filename, show_shapes=True)

    def predict(self, x_test):
        """Predict by a x_test dataset."""
        if x_test is None:
            raise ReferenceError('Test dataset is not provided')

        if not self._model:
            raise ReferenceError('Model has not been created')

        return self._model.predict(x_test)

    def save(self, filename):
        """Save the model."""
        if not filename:
            raise ValueError('Filename is not specified')

        if not self._model:
            raise ReferenceError('Model has not been created')

        self._model.save(filename)

    def summary(self):
        """Print summary of the model."""
        if not self._model:
            raise ReferenceError('Model has not been created')

        self._model.summary()
