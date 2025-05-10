"""
Python Lists and Tuples

This module covers lists and tuples in Python, including creation, access, modification,
and common operations.
"""

# LISTS
print("LISTS")
print("-" * 50)

# Creating lists
fruits = ["apple", "banana", "cherry", "orange", "kiwi"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, [1, 2, 3]]

print("List examples:")
print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed: {mixed}")

# Accessing list elements
print("\nAccessing list elements:")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")
print(f"First three fruits: {fruits[0:3]}")  # Slicing
print(f"Every other fruit: {fruits[::2]}")   # Step slicing

# Modifying lists
print("\nModifying lists:")
fruits[1] = "blueberry"  # Change an element
print(f"After changing banana to blueberry: {fruits}")

fruits.append("mango")   # Add to the end
print(f"After appending mango: {fruits}")

fruits.insert(2, "grape")  # Insert at specific position
print(f"After inserting grape at position 2: {fruits}")

fruits.remove("cherry")  # Remove by value
print(f"After removing cherry: {fruits}")

popped_fruit = fruits.pop()  # Remove and return the last item
print(f"Popped fruit: {popped_fruit}")
print(f"After popping: {fruits}")

popped_index = fruits.pop(1)  # Remove and return item at index
print(f"Popped fruit at index 1: {popped_index}")
print(f"After popping index 1: {fruits}")

# List operations
print("\nList operations:")
vegetables = ["carrot", "broccoli", "spinach"]
food = fruits + vegetables  # Concatenation
print(f"Fruits + Vegetables: {food}")

numbers = [1, 2, 3]
doubled = numbers * 2  # Repetition
print(f"Numbers doubled: {doubled}")

# List methods
print("\nList methods:")
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]
print(f"Original numbers: {numbers}")

numbers.sort()  # Sort in place
print(f"After sorting: {numbers}")

numbers.reverse()  # Reverse in place
print(f"After reversing: {numbers}")

print(f"Count of 5: {numbers.count(5)}")
print(f"Index of first 5: {numbers.index(5)}")

# List comprehensions
print("\nList comprehensions:")
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

even_numbers = [x for x in range(1, 11) if x % 2 == 0]
print(f"Even numbers: {even_numbers}")

# Nested lists (2D lists)
print("\nNested lists:")
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(f"Matrix: {matrix}")
print(f"Element at row 1, col 2: {matrix[1][2]}")  # Access 6

# TUPLES
print("\nTUPLES")
print("-" * 50)

# Creating tuples
coordinates = (10.0, 20.0)
person = ("John", 30, "New York")
singleton = (42,)  # Note the comma for single-element tuples
empty_tuple = ()

print("Tuple examples:")
print(f"Coordinates: {coordinates}")
print(f"Person: {person}")
print(f"Singleton: {singleton}")
print(f"Empty tuple: {empty_tuple}")

# Accessing tuple elements (similar to lists)
print("\nAccessing tuple elements:")
print(f"First coordinate: {coordinates[0]}")
print(f"Person's name: {person[0]}")
print(f"Person's age: {person[1]}")

# Tuple unpacking
print("\nTuple unpacking:")
x, y = coordinates
name, age, city = person
print(f"Unpacked coordinates: x={x}, y={y}")
print(f"Unpacked person: name={name}, age={age}, city={city}")

# Tuples are immutable
print("\nTuples are immutable:")
try:
    coordinates[0] = 15.0  # This will cause an error
except TypeError as e:
    print(f"Error: {e}")

# Tuple methods
print("\nTuple methods:")
numbers = (1, 2, 3, 2, 4, 2)
print(f"Count of 2: {numbers.count(2)}")
print(f"Index of first 3: {numbers.index(3)}")

# Converting between lists and tuples
print("\nConverting between lists and tuples:")
fruits_list = ["apple", "banana", "cherry"]
fruits_tuple = tuple(fruits_list)
print(f"List to tuple: {fruits_tuple}")

numbers_tuple = (1, 2, 3, 4, 5)
numbers_list = list(numbers_tuple)
print(f"Tuple to list: {numbers_list}")

# When to use tuples vs lists
print("\nWhen to use tuples vs lists:")
print("- Use tuples for fixed collections of items (like coordinates)")
print("- Use tuples for heterogeneous data (like database records)")
print("- Use tuples as dictionary keys (lists can't be used as keys)")
print("- Use lists when you need a mutable sequence")

# EXERCISE 1: List Operations
print("\nEXERCISE 1: List Operations")
# Create a list of your favorite movies
# Add a new movie to the list
# Remove one movie from the list
# Sort the list alphabetically
# Print the first and last movies in the sorted list

# Your code here:
# Example solution:
favorite_movies = ["The Matrix", "Inception", "Interstellar", "The Dark Knight"]
print(f"Original movie list: {favorite_movies}")

favorite_movies.append("Pulp Fiction")
print(f"After adding a movie: {favorite_movies}")

favorite_movies.remove("The Matrix")
print(f"After removing a movie: {favorite_movies}")

favorite_movies.sort()
print(f"Sorted movie list: {favorite_movies}")

print(f"First movie: {favorite_movies[0]}")
print(f"Last movie: {favorite_movies[-1]}")

# EXERCISE 2: Tuple Usage
print("\nEXERCISE 2: Tuple Usage")
# Create a tuple representing a point in 3D space (x, y, z)
# Create a list of such points representing a path
# Write a function that calculates the distance between two 3D points
# Calculate and print the total path length

# Your code here:
# Example solution:
import math

def distance_3d(point1, point2):
    """Calculate the Euclidean distance between two 3D points."""
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Create points and path
point1 = (0, 0, 0)
point2 = (1, 1, 1)
point3 = (2, 3, 1)
point4 = (4, 2, 2)

path = [point1, point2, point3, point4]
print(f"3D path: {path}")

# Calculate total path length
total_distance = 0
for i in range(len(path) - 1):
    segment_distance = distance_3d(path[i], path[i + 1])
    total_distance += segment_distance
    print(f"Distance from {path[i]} to {path[i + 1]}: {segment_distance:.2f}")

print(f"Total path length: {total_distance:.2f}")