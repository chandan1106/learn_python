"""
Python Inheritance and Polymorphism

This module covers inheritance and polymorphism in Python's object-oriented programming,
including class hierarchies, method overriding, and abstract classes.
"""

# Base class (parent class)
class Animal:
    """A base class representing an animal."""
    
    def __init__(self, name, species):
        """Initialize with name and species."""
        self.name = name
        self.species = species
    
    def make_sound(self):
        """The sound the animal makes (to be overridden by subclasses)."""
        return "Some generic animal sound"
    
    def __str__(self):
        """Return a string representation of the animal."""
        return f"{self.name} the {self.species}"


# Derived class (child class)
class Dog(Animal):
    """A class representing a dog, derived from Animal."""
    
    def __init__(self, name, breed):
        """Initialize a dog with name and breed."""
        # Call the parent class's __init__ method
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self):
        """Override the make_sound method."""
        return "Woof!"
    
    def fetch(self, item):
        """Dogs can fetch items."""
        return f"{self.name} is fetching the {item}."


# Another derived class
class Cat(Animal):
    """A class representing a cat, derived from Animal."""
    
    def __init__(self, name, color):
        """Initialize a cat with name and color."""
        super().__init__(name, species="Cat")
        self.color = color
    
    def make_sound(self):
        """Override the make_sound method."""
        return "Meow!"
    
    def scratch(self):
        """Cats can scratch."""
        return f"{self.name} is scratching."


# Creating and using objects
print("Creating Animal objects:")
generic_animal = Animal("Generic", "Unknown")
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Tabby")

print(f"\nString representations:")
print(generic_animal)
print(dog)
print(cat)

print(f"\nMaking sounds:")
print(f"{generic_animal}: {generic_animal.make_sound()}")
print(f"{dog}: {dog.make_sound()}")
print(f"{cat}: {cat.make_sound()}")

print(f"\nSpecific behaviors:")
print(dog.fetch("ball"))
print(cat.scratch())

# Checking inheritance relationships
print("\nChecking inheritance:")
print(f"Is dog an instance of Dog? {isinstance(dog, Dog)}")
print(f"Is dog an instance of Animal? {isinstance(dog, Animal)}")
print(f"Is dog an instance of Cat? {isinstance(dog, Cat)}")
print(f"Is Dog a subclass of Animal? {issubclass(Dog, Animal)}")

# Multiple inheritance
print("\nMultiple inheritance:")

class Flyable:
    """A mixin class for things that can fly."""
    
    def fly(self):
        """Fly in the air."""
        return f"{self.__class__.__name__} is flying!"


class Swimmable:
    """A mixin class for things that can swim."""
    
    def swim(self):
        """Swim in water."""
        return f"{self.__class__.__name__} is swimming!"


class Duck(Animal, Flyable, Swimmable):
    """A duck can both fly and swim."""
    
    def __init__(self, name):
        """Initialize a duck with a name."""
        super().__init__(name, species="Duck")
    
    def make_sound(self):
        """Override the make_sound method."""
        return "Quack!"


# Using multiple inheritance
duck = Duck("Donald")
print(duck)
print(f"Sound: {duck.make_sound()}")
print(duck.fly())
print(duck.swim())

# Method Resolution Order (MRO)
print("\nMethod Resolution Order (MRO):")
print(f"Duck MRO: {Duck.__mro__}")

# Polymorphism through a common interface
print("\nPolymorphism through a common interface:")

def animal_sound(animal):
    """Function that works with any animal object."""
    return animal.make_sound()

animals = [generic_animal, dog, cat, duck]
for animal in animals:
    print(f"{animal}: {animal_sound(animal)}")

# Abstract Base Classes
print("\nAbstract Base Classes:")
from abc import ABC, abstractmethod

class Shape(ABC):
    """An abstract base class for shapes."""
    
    @abstractmethod
    def area(self):
        """Calculate the area of the shape."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape."""
        pass
    
    def describe(self):
        """Describe the shape (non-abstract method)."""
        return f"This is a {self.__class__.__name__} with area {self.area()} and perimeter {self.perimeter()}."


class Circle(Shape):
    """A circle is a shape with a radius."""
    
    def __init__(self, radius):
        """Initialize with radius."""
        self.radius = radius
    
    def area(self):
        """Calculate the area of the circle."""
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        """Calculate the perimeter (circumference) of the circle."""
        import math
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    """A rectangle is a shape with width and height."""
    
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


# Using abstract base classes
# shape = Shape()  # This would raise TypeError: Can't instantiate abstract class

circle = Circle(5)
rectangle = Rectangle(4, 6)

print(circle.describe())
print(rectangle.describe())

# Duck typing (polymorphism without inheritance)
print("\nDuck typing:")

class Airplane:
    """An airplane can fly but doesn't inherit from Flyable."""
    
    def fly(self):
        """Airplanes can fly too."""
        return "The airplane is flying!"


def make_it_fly(entity):
    """This function works with any object that has a fly method."""
    return entity.fly()

flying_objects = [duck, Airplane()]
for obj in flying_objects:
    print(make_it_fly(obj))

# EXERCISE 1: Vehicle Hierarchy
print("\nEXERCISE 1: Vehicle Hierarchy")
# Create a base Vehicle class with attributes like make, model, year
# Create subclasses Car and Motorcycle that inherit from Vehicle
# Add specific attributes and methods to each subclass
# Create instances of each and demonstrate polymorphism

# Your code here:
# Example solution:
class Vehicle:
    """Base class for all vehicles."""
    
    def __init__(self, make, model, year):
        """Initialize with make, model, and year."""
        self.make = make
        self.model = model
        self.year = year
    
    def start(self):
        """Start the vehicle."""
        return f"The {self.year} {self.make} {self.model} is starting."
    
    def stop(self):
        """Stop the vehicle."""
        return f"The {self.year} {self.make} {self.model} is stopping."
    
    def __str__(self):
        """Return a string representation of the vehicle."""
        return f"{self.year} {self.make} {self.model}"


class Car(Vehicle):
    """A car is a type of vehicle with doors."""
    
    def __init__(self, make, model, year, num_doors):
        """Initialize with make, model, year, and number of doors."""
        super().__init__(make, model, year)
        self.num_doors = num_doors
    
    def honk(self):
        """Cars can honk."""
        return "Beep beep!"
    
    def __str__(self):
        """Return a string representation of the car."""
        return f"{super().__str__()} with {self.num_doors} doors"


class Motorcycle(Vehicle):
    """A motorcycle is a type of vehicle with an engine size."""
    
    def __init__(self, make, model, year, engine_size):
        """Initialize with make, model, year, and engine size."""
        super().__init__(make, model, year)
        self.engine_size = engine_size
    
    def wheelie(self):
        """Motorcycles can do wheelies."""
        return f"The {self.make} {self.model} pops a wheelie!"
    
    def __str__(self):
        """Return a string representation of the motorcycle."""
        return f"{super().__str__()} with {self.engine_size}cc engine"


# Test the vehicle classes
car = Car("Toyota", "Camry", 2020, 4)
motorcycle = Motorcycle("Harley-Davidson", "Sportster", 2019, 883)

print(car)
print(motorcycle)
print(car.start())
print(motorcycle.start())
print(car.honk())
print(motorcycle.wheelie())

# Demonstrate polymorphism
vehicles = [car, motorcycle]
for vehicle in vehicles:
    print(vehicle.start())
    print(vehicle.stop())

# EXERCISE 2: Shape Hierarchy with Abstract Methods
print("\nEXERCISE 2: Shape Hierarchy with Abstract Methods")
# Extend the Shape hierarchy by adding a Triangle class
# Implement the required abstract methods
# Create a function that calculates the total area of a list of shapes

# Your code here:
# Example solution:
class Triangle(Shape):
    """A triangle is a shape with three sides."""
    
    def __init__(self, a, b, c):
        """Initialize with the lengths of three sides."""
        self.a = a
        self.b = b
        self.c = c
    
    def area(self):
        """Calculate the area using Heron's formula."""
        # Semi-perimeter
        s = (self.a + self.b + self.c) / 2
        # Heron's formula
        import math
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def perimeter(self):
        """Calculate the perimeter of the triangle."""
        return self.a + self.b + self.c


def total_area(shapes):
    """Calculate the total area of a list of shapes."""
    return sum(shape.area() for shape in shapes)


# Test the Triangle class and total_area function
triangle = Triangle(3, 4, 5)
print(triangle.describe())

shapes = [circle, rectangle, triangle]
print(f"Total area of all shapes: {total_area(shapes):.2f}")