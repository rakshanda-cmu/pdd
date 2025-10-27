# test_calculator.py

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calculator import calculator
#from ut_calculator import calculator

class TestCalculator(unittest.TestCase):
    """
    Unit test suite for the calculate function in calculator.py.
    """

    # Scenario 1: Multiple Calculations (Looping)
    def test_multiple_calculations_in_sequence(self):
        """
        Tests a sequence of different calculations to simulate multiple uses.
        """
        print("\nRunning: test_multiple_calculations_in_sequence")
        calculations = [
            ('10', '5', '+', 15.0),
            ('20', '4', '*', 80.0),
            ('100', '10', '/', 10.0),
            ('5.5', '2.5', '-', 3.0)
        ]
        for num1, num2, op, expected in calculations:
            # The 'with self.subTest(...)' block allows the test to continue
            # even if one of the assertions fails, reporting all failures at the end.
            with self.subTest(f"Testing {num1} {op} {num2}"):
                self.assertEqual(calculator(num1, num2, op), expected)

    # Scenario 2: Floating-Point Precision
    def test_floating_point_precision(self):
        """
        Tests calculations that are known to cause floating-point precision issues.
        """
        print("Running: test_floating_point_precision")
        # 0.1 + 0.2 is notoriously imprecise in binary floating-point arithmetic
        self.assertAlmostEqual(calculator('0.1', '0.2', '+'), 0.3)
        self.assertAlmostEqual(calculator('1.123456', '2.987654', '+'), 4.11111)

    # Scenario 3: Empty Input Handling
    def test_empty_input_handling(self):
        """
        Tests that the function raises a ValueError for empty inputs.
        """
        print("Running: test_empty_input_handling")
        with self.assertRaisesRegex(ValueError, "Empty input is not allowed."):
            calculator('', '5', '+')
        with self.assertRaisesRegex(ValueError, "Empty input is not allowed."):
            calculator('10', '', '-')
        with self.assertRaisesRegex(ValueError, "Empty input is not allowed."):
            calculator('10', '5', '')

    # Scenario 4: Non-numeric Input (Special Characters)
    def test_non_numeric_input(self):
        """
        Tests that the function raises a ValueError for non-numeric inputs.
        """
        print("Running: test_non_numeric_input")
        with self.assertRaisesRegex(ValueError, "Invalid non-numeric input provided."):
            calculator('abc', '5', '+')
        with self.assertRaisesRegex(ValueError, "Invalid non-numeric input provided."):
            calculator('10', '$%^', '*')
        with self.assertRaisesRegex(ValueError, "Invalid operator: '@'"):
            calculator('10', '5', '@')

    # Scenario 5: Handling Edge Case for Negative Numbers
    def test_negative_number_operations(self):
        """
        Tests various calculations involving negative numbers.
        This ensures they are handled correctly mathematically.
        """
        print("Running: test_negative_number_operations")
        self.assertEqual(calculator('-10', '5', '+'), -5.0)
        self.assertEqual(calculator('-10', '-5', '+'), -15.0)
        self.assertEqual(calculator('10', '-5', '-'), 15.0) # 10 - (-5) = 15
        self.assertEqual(calculator('-10', '5', '*'), -50.0)
        self.assertEqual(calculator('-10', '-5', '*'), 50.0)
        self.assertEqual(calculator('-10', '-5', '/'), 2.0)

    # Scenario 6: Large or Small Numbers (Overflow/Underflow)
    def test_large_and_small_numbers(self):
        """
        Tests the calculator's behavior with very large and very small numbers
        to check for overflow and underflow issues.
        """
        print("Running: test_large_and_small_numbers")
        # Test with large numbers (potential for overflow)
        large_num = str(sys.float_info.max)
        self.assertEqual(calculator(large_num, '2', '*'), float('inf')) # Overflow

        # Test with small numbers (potential for underflow)
        small_num = str(sys.float_info.min)
        # Dividing the smallest positive float by 2 should result in underflow to 0.0
        self.assertEqual(calculator(small_num, '2', '/'), 0.0) # Underflow

        # Test with large numbers that don't overflow
        self.assertEqual(calculator('1e100', '1e100', '+'), 2e100)

    # Bonus: Test for division by zero
    def test_division_by_zero(self):
        """
        Ensures that dividing by zero raises the appropriate error.
        """
        print("Running: test_division_by_zero")
        with self.assertRaisesRegex(ValueError, "Cannot divide by zero."):
            calculator('10', '0', '/')

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main(verbosity=2)
