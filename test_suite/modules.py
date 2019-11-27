#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NIST statistical test suite modules importing."""
from ctypes import CDLL
from os import path


# Test files and test functions matching
# NIST statistical test suite version: 2.1.2

TEST_MODULES = {
    'approximateEntropy': 'ApproximateEntropy',
    'blockFrequency': 'BlockFrequency',
    'cusum': 'CumulativeSums',
    'dfft': 'drfti1',
    'discreteFourierTransform': 'DiscreteFourierTransform',
    'frequency': 'Frequency',
    'linearComplexity': 'LinearComplexity',
    'longestRunOfOnes': 'LongestRunOfOnes',
    'nonOverlappingTemplateMatchings': 'NonOverlappingTemplateMatchings',
    'overlappingTemplateMatchings': 'OverlappingTemplateMatchings',
    'randomExcursions': 'RandomExcursions',
    'randomExcursionsVariant': 'RandomExcursionsVariant',
    'rank': 'Rank',
    'runs': 'Runs',
    'serial': 'Serial',
    'universal': 'Universal'
}


def import_test_modules(test_suite_path):
    """Import NIST test suite from compiled C libraries."""
    if not path.exists(test_suite_path):
        raise ValueError(f'Incorrect NIST test suite path: {test_suite_path}')

    tests = []

    for test_filename in TEST_MODULES:
        test_path = path.join(test_suite_path, f'{test_filename}.so')

        if not path.exists(test_path):
            print(f'Test file doesn\'t exist: {test_path}')

            continue

        test_module = CDLL(test_path)

        test = getattr(test_module, TEST_MODULES[test_filename])

        tests.append(test)

    return tests
