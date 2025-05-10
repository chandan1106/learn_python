"""
Python Functions

This module covers functions in Python, including definition, parameters, return values,
and scope.
"""

# Function definition
def greet():
    """A simple function that prints a greeting."""
    print("Hello, World!")

# Function call
print("Basic function:")
greet()

# Function with parameters
def greet_person(name):
    """Greet a specific person."""
    print(f"Hello, {name}!")

print("\nFunction with parameters:")
greet_person("Alice")
greet_person("Bob")

# Function with default parameter
def greet_with_time(name, time_of_day="day"):
    """Greet a person with time of day."""
    print(f"Good {time_of_day}, {name}!")

print("\nFunction with default parameter:")
greet_with_time("Charlie")
greet_with_time("Diana", "evening")

# Function with multiple parameters
def describe_person(name, age, occupation):
    """Describe a person with multiple attributes."""
    print(f"{name} is {age} years old and works as a {occupation}.")

print("\nFunction with multiple parameters:")
describe_person("Eve", 28, "developer")

# Keyword arguments
print("\nKeyword arguments:")
describe_person(age=35, name="Frank", occupation="designer")

# Return values
def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

print("\nFunction with return value:")
result = add_numbers(5, 3)
print(f"5 + 3 = {result}")

# Multiple return values
def get_min_max(numbers):
    """Return both minimum and maximum values from a list."""
    return min(numbers), max(numbers)

print("\nFunction with multiple return values:")
numbers = [5, 3, 8, 1, 9, 2]
min_val, max_val = get_min_max(numbers)
print(f"Min: {min_val}, Max: {max_val}")

# Variable scope
x = 10  # Global variable

def scope_test():
    y = 5  # Local variable
    print(f"Inside function - x: {x}, y: {y}")

print("\nVariable scope:")
scope_test()
print(f"Outside function - x: {x}")
# print(f"Outside function - y: {y}")  # This would cause an error

# Modifying global variables
count = 0

def increment_count():
    global count  # Declare that we want to use the global variable
    count += 1
    print(f"Count inside function: {count}")

print("\nModifying global variables:")
print(f"Count before: {count}")
increment_count()
print(f"Count after: {count}")

# Nested functions
def outer_function(x):
    """Demonstrate a nested function."""
    def inner_function(y):
        return x + y
    return inner_function

print("\nNested functions:")
add_five = outer_function(5)
print(f"add_five(3) = {add_five(3)}")
print(f"add_five(7) = {add_five(7)}")

# Lambda functions (anonymous functions)
square = lambda x: x ** 2

print("\nLambda functions:")
print(f"square(4) = {square(4)}")

# Lambda with multiple parameters
multiply = lambda x, y: x * y
print(f"multiply(3, 4) = {multiply(3, 4)}")

# Lambda with filter
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_numbers}")

# Lambda with map
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(f"Squared numbers: {squared_numbers}")

# Docstrings
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle.
        width (float): The width of the rectangle.
        
    Returns:
        float: The area of the rectangle.
    """
    return length * width

print("\nFunction with docstring:")
print(calculate_area.__doc__)
print(f"Area of rectangle: {calculate_area(5, 3)}")

# EXERCISE 1: Create a function that checks if a number is prime
print("\nEXERCISE 1: Prime Number Checker")
# Your code here:
# Example solution:
def is_prime(number):
    """
    Check if a number is prime.
    
    Args:
        number (int): The number to check.
        
    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if number <= 1:
        return False
    if number <= 3:
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    
    return True

# Test the function
for num in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    print(f"{num} is prime: {is_prime(num)}")

# EXERCISE 2: Create a function that calculates the factorial of a number
print("\nEXERCISE 2: Factorial Calculator")
# Your code here:
# Example solution:
def factorial(n):
    """
    Calculate the factorial of a number.
    
    Args:
        n (int): The number to calculate factorial for.
        
    Returns:
        int: The factorial of n.
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Test the function
for num in range(6):
    print(f"{num}! = {factorial(num)}")