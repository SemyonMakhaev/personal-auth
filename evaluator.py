#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluating module for user data randomness estimation."""
from argparse import ArgumentParser
import sys
import numpy as np

from test_suite import test_bits


def parse_args():
    """Command-line arguments parsing."""
    parser = ArgumentParser(prog='evaluator.py',
                            description='Program for user data randomness evaluation',
                            epilog='Semyon Makhaev, 2019.')
    parser.add_argument('filename', type=str, help='User data filename')
    parser.add_argument('-t', '--tests_path', type=str, help='NIST statistical test suite path')

    return parser.parse_args()


def read_patterns(filename):
    """Read user data from file."""
    with open(filename, mode='r', encoding='utf-8') as file:
        file_data = file.read()

        return [pattern.encode('utf-8') for pattern in file_data.split('\n')]


def evaluate_patterns(patterns, test_suite_path):
    """Run NIST tests to determine randomness of patterns."""
    results = [test_bits(pattern, test_suite_path) if pattern else 0 for pattern in patterns]

    success_measure = np.mean(results)

    print(f'Evaluation: {success_measure * 100}%')


def main():
    """Start testing tools."""
    args = parse_args()

    if not args.filename:
        print('Filename has not been specified.')
        sys.exit(0)

    patterns = read_patterns(args.filename)

    if args.tests_path:
        evaluate_patterns(patterns, args.tests_path)


if __name__ == '__main__':
    main()
