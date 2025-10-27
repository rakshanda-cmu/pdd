# main.py
from typing import Optional

def calculator(num1: Optional[float] = None, num2: Optional[float] = None, operator: Optional[str] = None) -> Optional[float]:
    """
    Performs basic arithmetic calculations (+, -, *, /).

    It can be called in two ways:
    1. With parameters: calculator(num1, num2, operator)
    2. Interactively: calculator() - prompts the user for inputs.

    Args:
        num1 (float, optional): The first number. Defaults to None.
        num2 (float, optional): The second number. Defaults to None.
        operator (str, optional): The operator ('+', '-', '*', '/'). Defaults to None.

    Returns:
        float or None: The result of the calculation, or None if an error occurs.
    """
    # --- Input Phase ---
    # If any parameter is not provided, prompt the user for it.
    if num1 is None:
        while True:
            try:
                num1 = float(input("Enter the first number: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled.")
                return None

    if operator is None:
        while True:
            operator = input("Enter an operator (+, -, *, /): ").strip()
            if operator in ['+', '-', '*', '/']:
                break
            else:
                print("Invalid operator. Please use one of +, -, *, /.")

    if num2 is None:
        while True:
            try:
                num2 = float(input("Enter the second number: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled.")
                return None

    # --- Calculation and Output Phase ---
    result = None
    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 == 0:
            print("Error: Division by zero is not allowed.")
            return None  # Return None on division by zero error
        else:
            result = num1 / num2
    else:
        # This case is primarily for when parameters are passed incorrectly,
        # as the interactive input loop already validates the operator.
        print(f"Error: Invalid operator '{operator}'.")
        return None

    # Display the operation and result
    print(f"Operation: {num1} {operator} {num2} = {result}")
    return result

# Example of how to run the function interactively
if __name__ == '__main__':
    print("Welcome to the simple calculator!")
    calculator()
