"""
Advanced Python Concepts: Decorators and Generators
"""

# ===== DECORATORS =====
print("===== DECORATORS =====")

def simple_decorator(func):
    """A simple decorator that prints messages before and after function execution"""
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

# Using the decorator with the @ syntax
@simple_decorator
def greet(name):
    """A function that greets someone"""
    print(f"Hello, {name}!")
    return f"Greeted {name}"

# Using the decorated function
print("Calling the decorated function:")
result = greet("Alice")
print(f"Result: {result}")
print()

# Decorator with arguments
def repeat(n=1):
    """A decorator that repeats the function execution n times"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(n=3)
def say_hi(name):
    """Say hi to someone"""
    return f"Hi, {name}!"

print("Repeating a function with a decorator:")
print(say_hi("Bob"))
print()

# Class as a decorator
class CountCalls:
    """A decorator class that counts the number of times a function is called"""
    
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    """Say hello"""
    return "Hello!"

print("Using a class as a decorator:")
print(say_hello())
print(say_hello())
print(say_hello())
print()

# ===== GENERATORS =====
print("===== GENERATORS =====")

# Simple generator function
def count_up_to(max):
    """A generator that counts up to a maximum value"""
    count = 1
    while count <= max:
        yield count
        count += 1

print("Simple generator:")
for number in count_up_to(5):
    print(number, end=" ")
print("\n")

# Generator expression (similar to list comprehension)
print("Generator expression:")
squares = (x**2 for x in range(1, 6))
for square in squares:
    print(square, end=" ")
print("\n")

# Infinite generator with takewhile
from itertools import takewhile

def infinite_fibonacci():
    """An infinite generator for Fibonacci numbers"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

print("First 10 Fibonacci numbers:")
for i, fib in enumerate(takewhile(lambda x: i < 10, infinite_fibonacci())):
    print(fib, end=" ")
print("\n")

# Generator for processing large files efficiently
def read_large_file(file_path, chunk_size=1024):
    """A generator that reads a large file in chunks"""
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Example usage (commented out as we don't have a file to read)
# for chunk in read_large_file('large_file.txt'):
#     process_data(chunk)

# Chaining generators
def integers():
    """Generate integers 0, 1, 2, ..."""
    i = 0
    while True:
        yield i
        i += 1

def squares():
    """Generate squares of integers"""
    for i in integers():
        yield i * i

def take(n, seq):
    """Take first n elements from sequence"""
    for i, val in enumerate(seq):
        if i >= n:
            break
        yield val

print("First 5 square numbers using chained generators:")
for square in take(5, squares()):
    print(square, end=" ")
print("\n")

# Practical example: Pipeline with generators
def pipeline_example():
    """Example of a data processing pipeline using generators"""
    
    # Generate some data
    def generate_data():
        for i in range(1, 11):
            yield i
    
    # Filter even numbers
    def filter_even(numbers):
        for number in numbers:
            if number % 2 == 0:
                yield number
    
    # Multiply by 10
    def multiply_by_10(numbers):
        for number in numbers:
            yield number * 10
    
    # Build the pipeline
    data = generate_data()
    filtered_data = filter_even(data)
    result = multiply_by_10(filtered_data)
    
    return list(result)

print("Generator pipeline result:", pipeline_example())