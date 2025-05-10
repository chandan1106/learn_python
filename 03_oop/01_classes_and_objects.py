"""
Python Classes and Objects

This module covers the basics of object-oriented programming in Python,
including class definition, object creation, attributes, and methods.
"""

# Basic class definition
class Person:
    """A simple class representing a person."""
    
    # Class attribute (shared by all instances)
    species = "Homo sapiens"
    
    # Constructor method
    def __init__(self, name, age):
        """Initialize a new Person instance."""
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def introduce(self):
        """Return a string introduction of this person."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    # Instance method with parameters
    def celebrate_birthday(self):
        """Increment age by 1 and return a birthday message."""
        self.age += 1
        return f"Happy {self.age}th birthday, {self.name}!"
    
    # Class method (operates on the class, not instances)
    @classmethod
    def get_species(cls):
        """Return the species of all Person instances."""
        return cls.species
    
    # Static method (doesn't operate on the class or instances)
    @staticmethod
    def is_adult(age):
        """Return True if the age is 18 or older."""
        return age >= 18


# Creating objects (instances of the class)
print("Creating Person objects:")
person1 = Person("Alice", 25)
person2 = Person("Bob", 17)

# Accessing attributes
print("\nAccessing attributes:")
print(f"{person1.name} is {person1.age} years old.")
print(f"{person2.name} is {person2.age} years old.")
print(f"Both are {Person.species}.")

# Calling methods
print("\nCalling methods:")
print(person1.introduce())
print(person2.introduce())

# Modifying attributes
print("\nModifying attributes:")
person1.age = 26
print(f"{person1.name} is now {person1.age} years old.")

# Calling methods that modify attributes
print("\nCalling methods that modify attributes:")
print(person2.celebrate_birthday())
print(f"{person2.name} is now {person2.age} years old.")

# Calling class and static methods
print("\nCalling class and static methods:")
print(f"Species: {Person.get_species()}")
print(f"Is Alice an adult? {Person.is_adult(person1.age)}")
print(f"Is Bob an adult? {Person.is_adult(person2.age)}")

# Adding attributes dynamically
print("\nAdding attributes dynamically:")
person1.email = "alice@example.com"
print(f"{person1.name}'s email is {person1.email}")

# Class with property decorators
print("\nClass with property decorators:")

class Circle:
    """A class representing a circle."""
    
    def __init__(self, radius):
        """Initialize with radius."""
        self._radius = radius  # Protected attribute (by convention)
    
    @property
    def radius(self):
        """Getter for radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter for radius with validation."""
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def diameter(self):
        """Calculate diameter from radius."""
        return self._radius * 2
    
    @property
    def area(self):
        """Calculate area from radius."""
        import math
        return math.pi * self._radius ** 2


# Using properties
circle = Circle(5)
print(f"Circle radius: {circle.radius}")
print(f"Circle diameter: {circle.diameter}")
print(f"Circle area: {circle.area:.2f}")

# Setting properties
print("\nSetting properties:")
circle.radius = 7
print(f"New circle radius: {circle.radius}")
print(f"New circle diameter: {circle.diameter}")
print(f"New circle area: {circle.area:.2f}")

# Validation with properties
print("\nValidation with properties:")
try:
    circle.radius = -2
except ValueError as e:
    print(f"Error: {e}")

# Private attributes and name mangling
print("\nPrivate attributes and name mangling:")

class BankAccount:
    """A class representing a bank account."""
    
    def __init__(self, owner, balance=0):
        """Initialize with owner and optional balance."""
        self.owner = owner
        self.__balance = balance  # Private attribute
    
    def deposit(self, amount):
        """Add money to the account."""
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Amount must be positive"
    
    def withdraw(self, amount):
        """Remove money from the account if sufficient funds."""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        """Return the current balance."""
        return self.__balance


# Using a class with private attributes
account = BankAccount("Charlie", 1000)
print(f"Account owner: {account.owner}")
print(f"Initial balance: ${account.get_balance()}")
print(account.deposit(500))
print(account.withdraw(200))
print(account.withdraw(2000))  # Should fail

# Trying to access private attribute directly
try:
    print(account.__balance)  # This will raise an AttributeError
except AttributeError as e:
    print(f"Error: {e}")

# Name mangling (how Python implements private attributes)
print(f"Accessing private attribute with name mangling: ${account._BankAccount__balance}")

# EXERCISE 1: Create a Rectangle class
print("\nEXERCISE 1: Rectangle Class")
# Create a Rectangle class with width and height attributes
# Add methods to calculate area and perimeter
# Add a method to check if the rectangle is a square
# Create some rectangle objects and test your methods

# Your code here:
# Example solution:
class Rectangle:
    """A class representing a rectangle."""
    
    def __init__(self, width, height):
        """Initialize with width and height."""
        self.width = width
        self.height = height
    
    def area(self):
        """Calculate the area of the rectangle."""
        return self.width * self.height
    
    def perimeter(self):
        """Calculate the perimeter of the rectangle."""
        return 2 * (self.width + self.height)
    
    def is_square(self):
        """Check if the rectangle is a square."""
        return self.width == self.height
    
    def __str__(self):
        """Return a string representation of the rectangle."""
        return f"Rectangle(width={self.width}, height={self.height})"


# Test the Rectangle class
rect1 = Rectangle(5, 10)
rect2 = Rectangle(7, 7)

print(f"Rectangle 1: {rect1}")
print(f"Area: {rect1.area()}")
print(f"Perimeter: {rect1.perimeter()}")
print(f"Is square? {rect1.is_square()}")

print(f"\nRectangle 2: {rect2}")
print(f"Area: {rect2.area()}")
print(f"Perimeter: {rect2.perimeter()}")
print(f"Is square? {rect2.is_square()}")

# EXERCISE 2: Create a Student class
print("\nEXERCISE 2: Student Class")
# Create a Student class with name, age, and grades (list) attributes
# Add methods to add a grade, calculate average grade, and get highest/lowest grade
# Create some student objects and test your methods

# Your code here:
# Example solution:
class Student:
    """A class representing a student."""
    
    def __init__(self, name, age):
        """Initialize with name, age, and empty grades list."""
        self.name = name
        self.age = age
        self.grades = []
    
    def add_grade(self, grade):
        """Add a grade to the student's record."""
        if 0 <= grade <= 100:
            self.grades.append(grade)
            return True
        return False
    
    def average_grade(self):
        """Calculate the average grade."""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def highest_grade(self):
        """Get the highest grade."""
        if not self.grades:
            return None
        return max(self.grades)
    
    def lowest_grade(self):
        """Get the lowest grade."""
        if not self.grades:
            return None
        return min(self.grades)
    
    def __str__(self):
        """Return a string representation of the student."""
        return f"Student(name='{self.name}', age={self.age}, grades={self.grades})"


# Test the Student class
student1 = Student("David", 20)
student1.add_grade(85)
student1.add_grade(92)
student1.add_grade(78)
student1.add_grade(90)

print(f"Student: {student1}")
print(f"Average grade: {student1.average_grade():.2f}")
print(f"Highest grade: {student1.highest_grade()}")
print(f"Lowest grade: {student1.lowest_grade()}")