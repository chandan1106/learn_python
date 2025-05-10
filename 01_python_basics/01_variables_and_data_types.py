"""
Python Variables and Data Types

This module covers the fundamental concepts of variables and data types in Python.
"""

# Variables in Python
# Python is dynamically typed - you don't need to declare variable types

# Variable assignment
name = "John"  # String
age = 30       # Integer
height = 5.9   # Float
is_student = True  # Boolean

# Printing variables and their types
print("Variable Examples:")
print(f"Name: {name} (Type: {type(name)})")
print(f"Age: {age} (Type: {type(age)})")
print(f"Height: {height} (Type: {type(height)})")
print(f"Is Student: {is_student} (Type: {type(is_student)})")

# Multiple assignment
x, y, z = 1, 2, 3
print(f"\nMultiple assignment: x = {x}, y = {y}, z = {z}")

# Constants (by convention, use uppercase)
PI = 3.14159
GRAVITY = 9.8
print(f"\nConstants: PI = {PI}, GRAVITY = {GRAVITY}")

# Data Type Conversion
# Converting between types
age_str = str(age)
height_int = int(height)
num_str = "100"
num_int = int(num_str)

print("\nType Conversion:")
print(f"Age as string: {age_str} (Type: {type(age_str)})")
print(f"Height as integer: {height_int} (Type: {type(height_int)})")
print(f"String '100' to integer: {num_int} (Type: {type(num_int)})")

# Complex data types
# List (ordered, mutable collection)
fruits = ["apple", "banana", "cherry"]

# Tuple (ordered, immutable collection)
coordinates = (10.0, 20.0)

# Dictionary (key-value pairs)
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Set (unordered collection of unique items)
unique_numbers = {1, 2, 3, 4, 5}

print("\nComplex Data Types:")
print(f"List: {fruits} (Type: {type(fruits)})")
print(f"Tuple: {coordinates} (Type: {type(coordinates)})")
print(f"Dictionary: {person} (Type: {type(person)})")
print(f"Set: {unique_numbers} (Type: {type(unique_numbers)})")

# None type (represents absence of value)
empty_variable = None
print(f"\nNone type: {empty_variable} (Type: {type(empty_variable)})")

# EXERCISE 1: Create variables of different types and print their values and types
# Your code here:
print("\nEXERCISE 1: Create your own variables")
# Example solution:
name = "Chandan"
my_age = 23
my_favorite_numbers = [6, 11, 2001]
print(f"My name: {my_name} (Type: {type(my_name)})")
print(f"My age: {my_age} (Type: {type(my_age)})")
print(f"My favorite numbers: {my_favorite_numbers} (Type: {type(my_favorite_numbers)})")

# EXERCISE 2: Convert the following:
# - Convert float 9.99 to an integer
# - Convert integer 42 to a string
# - Convert the string "3.14159" to a float
print("\nEXERCISE 2: Type conversions")
# Your code here:
# Example solution:
price = 10.99
price_int = int(price)
answer = 42
answer_str = str(answer)
pi_str = "3.14159"
pi_float = float(pi_str)
print(f"9.99 as integer: {price_int}")
print(f"42 as string: {answer_str}")
print(f"'3.14159' as float: {pi_float}")