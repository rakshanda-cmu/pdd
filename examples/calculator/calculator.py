def calculator():
    """
    A simple calculator function that performs basic arithmetic operations.
    It prompts the user for two numbers and an operator (+, -, *, /),
    then displays the operation and the result.
    """
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
        return # Exit the function if input is invalid

    # --- Get the Operation ---
    operation = input("Enter the operation to perform (+, -, *, /): ")

    # --- Perform Calculation and Display Result ---
    if operation == '+':
        result = num1 + num2
        print(f"\nOperation: {num1} + {num2}")
        print(f"Result: {result}")

    elif operation == '-':
        result = num1 - num2
        print(f"\nOperation: {num1} - {num2}")
        print(f"Result: {result}")

    elif operation == '*':
        result = num1 * num2
        print(f"\nOperation: {num1} * {num2}")
        print(f"Result: {result}")

    elif operation == '/':
        # Check for division by zero, which is not allowed
        if num2 == 0:
            print("\nError! Division by zero is not possible.")
        else:
            result = num1 / num2
            print(f"\nOperation: {num1} / {num2}")
            print(f"Result: {result}")

    else:
        # Handle cases where the operator is not one of the valid options
        print("\nInvalid operation. Please choose from +, -, *, /")


# --- Main part of the script ---
if __name__ == "__main__":
    # Call the function to run the calculator
    calculator()