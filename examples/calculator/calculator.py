def calculator(num1=None, num2=None, operation=None):
    """
    A simple calculator function that performs basic arithmetic operations.
    It accepts two numbers and an operator (+, -, *, /) and returns the result.
    If no arguments are passed, it prompts the user for input.
    """
    if num1 is None or num2 is None or operation is None:
        print("--- Simple Python Calculator ---")

        # --- Get User Input and Handle Errors ---
        try:
            # Get the first number from the user
            num1 = float(input("Enter the first number: "))

            # Get the second number from the user
            num2 = float(input("Enter the second number: "))

        except ValueError:
            # Handle cases where the input is not a valid number
            print("Invalid input. Please enter numeric values only.")
            return None  # Exit the function if input is invalid

        # --- Get the Operation ---
        operation = input("Enter the operation to perform (+, -, *, /): ")

    # --- Perform Calculation and Display Result ---
    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        if num2 == 0:
            print("\nError! Division by zero is not possible.")
            return None
        result = num1 / num2
    else:
        print("\nInvalid operation. Please choose from +, -, *, /")
        return None

    print(f"\nOperation: {num1} {operation} {num2}")
    print(f"Result: {result}")
    return result


# --- Main part of the script ---
if __name__ == "__main__":
    # Call the function to run the calculator interactively
    calculator()
