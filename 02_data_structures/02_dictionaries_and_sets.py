"""
Python Dictionaries and Sets

This module covers dictionaries and sets in Python, including creation, access, modification,
and common operations.
"""

# DICTIONARIES
print("DICTIONARIES")
print("-" * 50)

# Creating dictionaries
person = {
    "name": "John",
    "age": 30,
    "city": "New York",
    "is_student": False
}

# Empty dictionary
empty_dict = {}

# Dictionary with mixed key types (not recommended but possible)
mixed_dict = {
    "name": "Alice",
    42: "The answer",
    (1, 2): "Tuple key"
}

print("Dictionary examples:")
print(f"Person: {person}")
print(f"Empty dictionary: {empty_dict}")
print(f"Mixed dictionary: {mixed_dict}")

# Accessing dictionary values
print("\nAccessing dictionary values:")
print(f"Name: {person['name']}")
print(f"Age: {person['age']}")

# Using get() method (safer access)
print(f"City (using get): {person.get('city')}")
print(f"Phone (using get): {person.get('phone')}")  # Returns None
print(f"Phone (with default): {person.get('phone', 'Not available')}")  # Returns default

# Modifying dictionaries
print("\nModifying dictionaries:")
person["age"] = 31  # Update existing key
print(f"After updating age: {person}")

person["phone"] = "555-1234"  # Add new key-value pair
print(f"After adding phone: {person}")

# Dictionary methods
print("\nDictionary methods:")

# keys(), values(), items()
print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Items: {list(person.items())}")

# pop() - remove and return a value
phone = person.pop("phone")
print(f"Popped phone: {phone}")
print(f"After popping phone: {person}")

# popitem() - remove and return the last inserted key-value pair
last_item = person.popitem()
print(f"Popped last item: {last_item}")
print(f"After popping last item: {person}")

# update() - update dictionary with another dictionary
person.update({"city": "Boston", "occupation": "Developer"})
print(f"After update: {person}")

# clear() - remove all items
empty_dict.clear()
print(f"After clear: {empty_dict}")

# Dictionary comprehensions
print("\nDictionary comprehensions:")
squares = {x: x**2 for x in range(1, 6)}
print(f"Squares: {squares}")

# Nested dictionaries
print("\nNested dictionaries:")
contacts = {
    "John": {
        "phone": "555-1234",
        "email": "john@example.com"
    },
    "Mary": {
        "phone": "555-5678",
        "email": "mary@example.com"
    }
}
print(f"Contacts: {contacts}")
print(f"John's email: {contacts['John']['email']}")

# SETS
print("\nSETS")
print("-" * 50)

# Creating sets
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 4, 5}
mixed_set = {1, "hello", 3.14, (1, 2)}  # Note: only immutable elements allowed

# Empty set (can't use {} as that creates an empty dictionary)
empty_set = set()

print("Set examples:")
print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed set: {mixed_set}")
print(f"Empty set: {empty_set}")

# Sets automatically remove duplicates
print("\nSets remove duplicates:")
numbers_with_duplicates = {1, 2, 2, 3, 4, 4, 5}
print(f"Set with duplicates: {numbers_with_duplicates}")

# Set operations
print("\nSet operations:")

# add() and remove()
fruits.add("orange")
print(f"After adding orange: {fruits}")

fruits.remove("banana")  # Raises error if not found
print(f"After removing banana: {fruits}")

fruits.discard("kiwi")  # No error if not found
print(f"After discarding kiwi (not in set): {fruits}")

popped = fruits.pop()  # Remove and return an arbitrary element
print(f"Popped element: {popped}")
print(f"After popping: {fruits}")

# Mathematical set operations
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print("\nMathematical set operations:")
print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Union
union_set = set1.union(set2)  # or set1 | set2
print(f"Union: {union_set}")

# Intersection
intersection_set = set1.intersection(set2)  # or set1 & set2
print(f"Intersection: {intersection_set}")

# Difference
difference_set = set1.difference(set2)  # or set1 - set2
print(f"Difference (set1 - set2): {difference_set}")

# Symmetric difference
symmetric_difference = set1.symmetric_difference(set2)  # or set1 ^ set2
print(f"Symmetric difference: {symmetric_difference}")

# Set comprehensions
print("\nSet comprehensions:")
even_squares = {x**2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Set membership testing (very fast)
print("\nSet membership testing:")
print(f"Is 3 in set1? {'3' in set1}")
print(f"Is 10 in set1? {10 in set1}")

# Frozen sets (immutable sets)
print("\nFrozen sets:")
frozen = frozenset([1, 2, 3, 4])
print(f"Frozen set: {frozen}")

try:
    frozen.add(5)  # This will cause an error
except AttributeError as e:
    print(f"Error: {e}")

# EXERCISE 1: Dictionary Operations
print("\nEXERCISE 1: Dictionary Operations")
# Create a dictionary representing a book with title, author, year, and genre
# Add a new key for "pages"
# Update the year
# Print all the book information in a formatted way
# Check if "publisher" exists in the dictionary, if not add it

# Your code here:
# Example solution:
book = {
    "title": "The Hobbit",
    "author": "J.R.R. Tolkien",
    "year": 1937,
    "genre": "Fantasy"
}
print(f"Original book: {book}")

book["pages"] = 310
print(f"After adding pages: {book}")

book["year"] = 1938  # Updating the year
print(f"After updating year: {book}")

print("\nBook Information:")
for key, value in book.items():
    print(f"{key.capitalize()}: {value}")

if "publisher" not in book:
    book["publisher"] = "Allen & Unwin"
print(f"After checking for publisher: {book}")

# EXERCISE 2: Set Operations
print("\nEXERCISE 2: Set Operations")
# Create two sets: one with programming languages you know, one with languages you want to learn
# Find languages that are in both sets
# Find languages that you want to learn but don't know yet
# Add a new language to the "want to learn" set
# Print the total number of unique languages across both sets

# Your code here:
# Example solution:
known_languages = {"Python", "JavaScript", "HTML", "CSS"}
want_to_learn = {"Python", "Rust", "Go", "Swift", "Kotlin"}

print(f"Languages I know: {known_languages}")
print(f"Languages I want to learn: {want_to_learn}")

both_sets = known_languages.intersection(want_to_learn)
print(f"Languages in both sets: {both_sets}")

to_learn = want_to_learn.difference(known_languages)
print(f"Languages I want to learn but don't know yet: {to_learn}")

want_to_learn.add("TypeScript")
print(f"After adding TypeScript to want_to_learn: {want_to_learn}")

all_languages = known_languages.union(want_to_learn)
print(f"Total unique languages: {len(all_languages)}")
print(f"All languages: {all_languages}")