#!/usr/bin/env python3
from sp800_22_tests.sp800_22_approximate_entropy_test import approximate_entropy_test
from sp800_22_tests.sp800_22_cumulative_sums_test import cumulative_sums_test
from sp800_22_tests.sp800_22_dft_test import dft_test
from sp800_22_tests.sp800_22_frequency_within_block_test import frequency_within_block_test
from sp800_22_tests.sp800_22_longest_run_ones_in_a_block_test import longest_run_ones_in_a_block_test
from sp800_22_tests.sp800_22_monobit_test import monobit_test
from sp800_22_tests.sp800_22_non_overlapping_template_matching_test import non_overlapping_template_matching_test
from sp800_22_tests.sp800_22_random_excursion_test import random_excursion_test
from sp800_22_tests.sp800_22_random_excursion_variant_test import random_excursion_variant_test
from sp800_22_tests.sp800_22_runs_test import runs_test
from sp800_22_tests.sp800_22_serial_test import serial_test


TESTS = [
    monobit_test,
    frequency_within_block_test,
    runs_test,
    longest_run_ones_in_a_block_test,
    dft_test,
    non_overlapping_template_matching_test,
    serial_test,
    approximate_entropy_test,
    cumulative_sums_test,
    random_excursion_test,
    random_excursion_variant_test
]


def test_number(number):
    """Run NIST tests for number."""
    number_bytes = number.to_bytes((number.bit_length() + 7) // 8, 'big')

    for test in TESTS:
        success, p, plist = test(number_bytes)

        if not success:
            return False

    return True
