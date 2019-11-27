#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistical test suite tools."""
from test_suite.modules import import_test_modules


def test_number(number, test_suite_path):
    """Cast number to bytes and run tests."""
    number_bytes = number.to_bytes((number.bit_length() + 7) // 8, 'big')

    return test_bits(number_bytes, test_suite_path)


def test_bits(number_bytes, test_suite_path):
    """Run NIST tests for number bytes."""
    if not test_suite_path:
        print('Test suite has not been specified.')

        return None

    tests = import_test_modules(test_suite_path)

    success_count = 0

    for test in tests:
        success = False

        if not number_bytes:
            continue

        try:
            success = test(number_bytes)
        except ZeroDivisionError:
            print(f'Test is not started: {test.__name__}')

        if success:
            success_count += 1

    return success_count / len(tests)
