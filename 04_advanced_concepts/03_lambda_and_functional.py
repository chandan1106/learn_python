"""
Advanced Python Concepts: Lambda Functions and Functional Programming
"""

# ===== LAMBDA FUNCTIONS =====
print("===== LAMBDA FUNCTIONS =====")

# Basic lambda function
add = lambda x, y: x + y
print("Lambda function to add two numbers:")
print(f"add(5, 3) = {add(5, 3)}")
print()

# Lambda with built-in functions
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Using lambda with filter
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print("Filtering even numbers with lambda:")
print(f"even_numbers = {even_numbers}")

# Using lambda with map
squared = list(map(lambda x: x**2, numbers))
print("Mapping numbers to their squares with lambda:")
print(f"squared = {squared}")

# Using lambda with sorted
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
sorted_by_second = sorted(pairs, key=lambda pair: pair[1])
print("Sorting pairs by second element with lambda:")
print(f"sorted_by_second = {sorted_by_second}")
print()

# Lambda in list comprehensions
print("Lambda in list comprehensions:")
operations = [lambda x: x+1, lambda x: x*2, lambda x: x**2]
for f in operations:
    print(f"f(10) = {f(10)}")
print()

# ===== FUNCTIONAL PROGRAMMING =====
print("===== FUNCTIONAL PROGRAMMING =====")

# Immutability
print("Immutability example:")
immutable_list = (1, 2, 3, 4, 5)  # Using a tuple for immutability
print(f"Original tuple: {immutable_list}")
# Creating a new tuple instead of modifying the original
new_list = immutable_list + (6,)
print(f"New tuple: {new_list}")
print()

# Pure functions
print("Pure functions:")

def pure_add(a, b):
    """A pure function that adds two numbers"""
    return a + b

print(f"pure_add(3, 4) = {pure_add(3, 4)}")

# Example of an impure function for comparison
count = 0
def impure_add(a, b):
    """An impure function that adds two numbers but also modifies global state"""
    global count
    count += 1
    return a + b

print(f"impure_add(3, 4) = {impure_add(3, 4)}")
print(f"impure_add(3, 4) = {impure_add(3, 4)}")
print(f"Side effect - count: {count}")
print()

# Higher-order functions
print("Higher-order functions:")

def apply_operation(func, a, b):
    """A higher-order function that applies a function to two arguments"""
    return func(a, b)

print(f"apply_operation(lambda x, y: x + y, 5, 3) = {apply_operation(lambda x, y: x + y, 5, 3)}")
print(f"apply_operation(lambda x, y: x * y, 5, 3) = {apply_operation(lambda x, y: x * y, 5, 3)}")
print()

# Function composition
print("Function composition:")

def compose(f, g):
    """Compose two functions: f(g(x))"""
    return lambda x: f(g(x))

# Example functions
def double(x):
    return x * 2

def increment(x):
    return x + 1

# Compose them
double_then_increment = compose(increment, double)
increment_then_double = compose(double, increment)

print(f"double_then_increment(5) = {double_then_increment(5)}")  # (5*2)+1 = 11
print(f"increment_then_double(5) = {increment_then_double(5)}")  # (5+1)*2 = 12
print()

# Recursion instead of loops
print("Recursion example:")

def factorial(n):
    """Calculate factorial using recursion"""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

print(f"factorial(5) = {factorial(5)}")
print()

# Using functools module
print("Using functools module:")
import functools

# Reduce function
product = functools.reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])
print(f"Product of [1, 2, 3, 4, 5] using reduce: {product}")

# Partial function
base_two_log = functools.partial(pow, 2)
print(f"2^10 using partial function: {base_two_log(10)}")

# Cache function results
@functools.lru_cache(maxsize=None)
def fibonacci(n):
    """Calculate Fibonacci number with caching for efficiency"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("Fibonacci sequence with memoization:")
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")
print()

# Currying (transforming a function with multiple arguments into a sequence of functions)
print("Currying example:")

def curry_add(x):
    """Curried version of addition"""
    def add_y(y):
        return x + y
    return add_y

add_five = curry_add(5)
print(f"add_five(3) = {add_five(3)}")
print(f"curry_add(2)(3) = {curry_add(2)(3)}")
print()

# List comprehensions as functional programming
print("List comprehensions:")
squares_comp = [x**2 for x in range(1, 6)]
print(f"Squares using list comprehension: {squares_comp}")

# Dictionary comprehensions
dict_comp = {x: x**2 for x in range(1, 6)}
print(f"Squares dictionary using comprehension: {dict_comp}")

# Set comprehensions
set_comp = {x % 3 for x in range(10)}
print(f"Remainders when divided by 3 using set comprehension: {set_comp}")
print()

# Combining functional techniques
print("Combining functional techniques:")

# Example: Calculate the sum of squares of even numbers from 1 to 10
result = sum(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(1, 11))))
print(f"Sum of squares of even numbers from 1 to 10: {result}")

# Same with list comprehension
result_comp = sum([x**2 for x in range(1, 11) if x % 2 == 0])
print(f"Same result using list comprehension: {result_comp}")
print()

# Using itertools for functional operations
print("Using itertools module:")
import itertools

# Infinite counting
print("First 5 numbers from an infinite count:")
for i, num in enumerate(itertools.count(10)):
    if i >= 5:
        break
    print(num, end=" ")
print()

# Cycle through elements
print("Cycling through elements:")
cycle_count = 0
for item in itertools.cycle(['A', 'B', 'C']):
    if cycle_count >= 6:
        break
    print(item, end=" ")
    cycle_count += 1
print()

# Combinations and permutations
print("Combinations of [1, 2, 3] taken 2 at a time:")
for combo in itertools.combinations([1, 2, 3], 2):
    print(combo, end=" ")
print()

print("Permutations of [1, 2, 3] taken 2 at a time:")
for perm in itertools.permutations([1, 2, 3], 2):
    print(perm, end=" ")
print()