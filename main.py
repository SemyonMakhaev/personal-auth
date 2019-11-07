#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The program provides neural networks for recommendation
or modification of cryptographically random numbers.
"""
from argparse import ArgumentParser
from os import path

from models import Limiter, Modifier, Recommender
from sp800_22_tests import test_number


__version__ = '1.0'
__author__ = 'Semyon Makhaev'
__email__ = 'semenmakhaev@yandex.ru'


MODELS = {
    'limiter': Limiter,
    'modifier': Modifier,
    'recommender': Recommender
}


def parse_args():
    """Command-line arguments parsing."""
    parser = ArgumentParser(prog='main.py',
                            description='Program providing neural networks for recommendation \
                            or modification of cryptographically random numbers',
                            epilog='Semyon Makhaev, 2019.')
    parser.add_argument('model_type',
                        type=str,
                        choices=['limiter', 'modifier', 'recommender'],
                        help='Model filename')
    parser.add_argument('--model_filename', '-m', type=str, help='Model filename')
    parser.add_argument('--plot_filename', '-p', type=str, help='Plot filename')
    parser.add_argument('--summary', '-s', action='store_true', help='Print model summary')
    parser.add_argument('--fit', '-f', action='store_true', help='Fit model')

    return parser.parse_args()


def initialize_model(model_type, model_filename):
    """Model initialization."""
    if not model_type or model_type not in MODELS:
        raise NameError(f'Model {model_type} does not exist')

    model = MODELS[model_type]()

    if model_filename and path.exists(model_filename):
        model.load(model_filename)
    else:
        model.create()
        model.compile()

    return model


def postprocess_predictions(predictions):
    """Splits data to binary numbers."""
    result = []

    for prediction in predictions:
        bits = [0 if x < 0.5 else 1 for x in prediction]
        bits_str = ''.join([str(x) for x in bits])
        number = int(f'0b{bits_str}', 2)

        result.append(number)

    return result


def evaluate_prediction(numbers):
    """Run NIST tests to determine randomness of predicted numbers."""
    success_count = 0

    for number in numbers:
        if test_number(number):
            success_count += 1

    success_percentage = (success_count / len(numbers)) * 100

    print(f'Evaluation: {success_percentage}%')


def main():
    """Running tools for model creating, training and evaluating."""
    args = parse_args()

    model = initialize_model(args.model_type, args.model_filename)

    dataset = model.get_dataset()

    if args.plot_filename:
        model.plot(args.plot_filename)

    if args.fit:
        model.fit(dataset)
        model.save(args.model_filename)

    if args.summary:
        model.summary()

    _, x_test = dataset

    predictions = model.predict(x_test)

    if model.is_evaluable:
        numbers = postprocess_predictions(predictions)

        evaluate_prediction(numbers)
    else:
        print(f'Predictions: {predictions}')


if __name__ == '__main__':
    main()
