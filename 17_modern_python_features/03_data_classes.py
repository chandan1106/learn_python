"""
Modern Python Features: Data Classes
"""

# ===== INTRODUCTION TO DATA CLASSES =====
print("\n===== INTRODUCTION TO DATA CLASSES =====")
"""
Data classes (introduced in Python 3.7 via PEP 557) provide a way to create classes
that are primarily used to store data. They automatically generate special methods
like __init__, __repr__, and __eq__ based on the class attributes.

Benefits:
1. Reduced boilerplate code
2. Automatic generation of special methods
3. Built-in comparison capabilities
4. Easy customization
5. Integration with type hints
"""

from dataclasses import dataclass, field, asdict, astuple, replace
from typing import List, Optional, ClassVar, Dict, Any

# ===== BASIC DATA CLASS =====
print("\n===== BASIC DATA CLASS =====")

# Simple data class
@dataclass
class Point:
    x: int
    y: int

# Create an instance
p = Point(10, 20)
print(f"Point: {p}")  # __repr__ is automatically generated
print(f"Point coordinates: x={p.x}, y={p.y}")

# Compare instances
p1 = Point(10, 20)
p2 = Point(10, 20)
p3 = Point(30, 40)

print(f"p1 == p2: {p1 == p2}")  # __eq__ is automatically generated
print(f"p1 == p3: {p1 == p3}")

# ===== DEFAULT VALUES =====
print("\n===== DEFAULT VALUES =====")

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Default value
    in_stock: bool = True  # Default value
    tags: List[str] = field(default_factory=list)  # Mutable default

# Create instances
product1 = Product("Laptop", 999.99)
product2 = Product("Phone", 499.99, 10, False, ["electronics", "mobile"])

print(f"Product 1: {product1}")
print(f"Product 2: {product2}")

# Add tags to product1
product1.tags.append("electronics")
product1.tags.append("computers")
print(f"Product 1 with tags: {product1}")

# ===== FIELD OPTIONS =====
print("\n===== FIELD OPTIONS =====")

@dataclass
class User:
    username: str
    email: str
    password: str = field(repr=False)  # Don't include in repr
    active: bool = True
    login_attempts: int = field(default=0, compare=False)  # Don't include in comparison
    last_login: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    _id: int = field(init=False)  # Not included in __init__ parameters
    
    def __post_init__(self):
        # Called after __init__
        self._id = hash(self.username)

# Create users
user1 = User("alice", "alice@example.com", "secret123")
user2 = User("bob", "bob@example.com", "password456", roles=["admin"])

print(f"User 1: {user1}")  # password is not shown in repr
print(f"User 2: {user2}")
print(f"User 1 ID: {user1._id}")

# Increment login attempts
user1.login_attempts += 1
print(f"User 1 login attempts: {user1.login_attempts}")

# Compare users (login_attempts is not considered)
user1_copy = User("alice", "alice@example.com", "secret123")
user1_copy.login_attempts = 5
print(f"user1 == user1_copy: {user1 == user1_copy}")

# ===== IMMUTABLE DATA CLASSES =====
print("\n===== IMMUTABLE DATA CLASSES =====")

@dataclass(frozen=True)
class ImmutablePoint:
    x: int
    y: int

# Create an immutable point
immutable_p = ImmutablePoint(5, 10)
print(f"Immutable point: {immutable_p}")

# Try to modify (will raise an error)
try:
    immutable_p.x = 20
except Exception as e:
    print(f"Error when trying to modify: {type(e).__name__}: {e}")

# Create a new instance with replace
new_p = replace(immutable_p, x=20)
print(f"New point after replace: {new_p}")
print(f"Original point (unchanged): {immutable_p}")

# ===== INHERITANCE WITH DATA CLASSES =====
print("\n===== INHERITANCE WITH DATA CLASSES =====")

@dataclass
class Person:
    name: str
    age: int
    address: str = "Unknown"

@dataclass
class Employee(Person):
    employee_id: str
    department: str
    salary: float = 0.0

# Create an employee
employee = Employee("John Doe", 30, "123 Main St", "E12345", "Engineering", 75000.0)
print(f"Employee: {employee}")

# ===== CLASS VARIABLES =====
print("\n===== CLASS VARIABLES =====")

@dataclass
class Configuration:
    name: str
    value: Any
    description: str = ""
    
    # Class variables
    VERSION: ClassVar[str] = "1.0.0"
    VALID_TYPES: ClassVar[List[type]] = [str, int, float, bool]
    
    def is_valid_type(self) -> bool:
        return type(self.value) in self.VALID_TYPES

# Create configurations
config1 = Configuration("timeout", 30, "Connection timeout in seconds")
config2 = Configuration("debug", True, "Enable debug mode")
config3 = Configuration("complex", complex(1, 2), "Complex number")

print(f"Config 1: {config1}, valid type: {config1.is_valid_type()}")
print(f"Config 2: {config2}, valid type: {config2.is_valid_type()}")
print(f"Config 3: {config3}, valid type: {config3.is_valid_type()}")
print(f"Configuration version: {Configuration.VERSION}")

# ===== POST-INITIALIZATION PROCESSING =====
print("\n===== POST-INITIALIZATION PROCESSING =====")

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Calculated field
    perimeter: float = field(init=False)  # Calculated field
    
    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)

# Create a rectangle
rect = Rectangle(10.0, 5.0)
print(f"Rectangle: {rect}")
print(f"Area: {rect.area}, Perimeter: {rect.perimeter}")

# ===== CONVERTING TO DICT OR TUPLE =====
print("\n===== CONVERTING TO DICT OR TUPLE =====")

@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

@dataclass
class Customer:
    id: int
    name: str
    email: str
    address: Address
    active: bool = True

# Create a customer with address
address = Address("123 Main St", "Anytown", "CA", "12345")
customer = Customer(1, "Alice Smith", "alice@example.com", address)

# Convert to dictionary
customer_dict = asdict(customer)
print(f"Customer as dict: {customer_dict}")

# Convert to tuple
customer_tuple = astuple(customer)
print(f"Customer as tuple: {customer_tuple}")

# ===== ADVANCED FIELD CUSTOMIZATION =====
print("\n===== ADVANCED FIELD CUSTOMIZATION =====")

def validate_positive(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be positive")

@dataclass
class Product:
    name: str
    price: float = field(metadata={"validate": validate_positive})
    quantity: int = field(default=0, metadata={"validate": validate_positive})
    
    def __post_init__(self):
        for field_name, field_value in self.__dataclass_fields__.items():
            if "validate" in field_value.metadata:
                validator = field_value.metadata["validate"]
                validator(self, field_value, getattr(self, field_name))

# Create valid product
try:
    valid_product = Product("Keyboard", 49.99, 10)
    print(f"Valid product: {valid_product}")
except ValueError as e:
    print(f"Error: {e}")

# Try to create invalid product
try:
    invalid_product = Product("Mouse", -10.0)
    print(f"Invalid product: {invalid_product}")
except ValueError as e:
    print(f"Error: {e}")

# ===== SLOTS WITH DATA CLASSES =====
print("\n===== SLOTS WITH DATA CLASSES =====")
"""
Using __slots__ with dataclasses can reduce memory usage and improve attribute access speed.
"""

@dataclass
class PointWithSlots:
    x: int
    y: int
    
    __slots__ = ('x', 'y')  # Define slots

# Create a point with slots
point_with_slots = PointWithSlots(10, 20)
print(f"Point with slots: {point_with_slots}")

# Try to add a new attribute (will raise an error)
try:
    point_with_slots.z = 30
except Exception as e:
    print(f"Error when adding new attribute: {type(e).__name__}: {e}")

# ===== DATACLASSES VS NAMEDTUPLE =====
print("\n===== DATACLASSES VS NAMEDTUPLE =====")
"""
Comparison between dataclasses and namedtuple:

Dataclasses:
- Mutable by default (can be made immutable with frozen=True)
- More flexible and customizable
- Support for default values and post-initialization
- Better integration with type hints
- Can use inheritance

NamedTuple:
- Always immutable
- More lightweight
- Compatible with regular tuples
- Less customizable
"""

from collections import namedtuple

# Using namedtuple
PointTuple = namedtuple('PointTuple', ['x', 'y'])
pt = PointTuple(10, 20)
print(f"Named tuple point: {pt}")
print(f"x={pt.x}, y={pt.y}")

# Using dataclass
@dataclass(frozen=True)
class PointClass:
    x: int
    y: int

pc = PointClass(10, 20)
print(f"Data class point: {pc}")
print(f"x={pc.x}, y={pc.y}")

# ===== PRACTICAL EXAMPLE: CONFIGURATION SYSTEM =====
print("\n===== PRACTICAL EXAMPLE: CONFIGURATION SYSTEM =====")

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str = field(repr=False)
    database: str
    pool_size: int = 5
    timeout: int = 30
    ssl_enabled: bool = False
    
    def get_connection_string(self) -> str:
        """Generate a connection string."""
        protocol = "postgresql"
        if self.ssl_enabled:
            protocol += "+ssl"
        return f"{protocol}://{self.username}:***@{self.host}:{self.port}/{self.database}"

@dataclass
class AppConfig:
    app_name: str
    version: str
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    database: DatabaseConfig = None
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug

# Create configuration
db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    username="admin",
    password="secret",
    database="myapp",
    ssl_enabled=True
)

app_config = AppConfig(
    app_name="MyAwesomeApp",
    version="1.0.0",
    debug=True,
    database=db_config
)

print(f"App configuration: {app_config}")
print(f"Database connection string: {app_config.database.get_connection_string()}")
print(f"Running in development mode: {app_config.is_development()}")

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Data classes provide a clean, concise way to create classes that primarily store data.
They reduce boilerplate code while maintaining flexibility and customization options.

Key benefits:
1. Automatic generation of special methods (__init__, __repr__, __eq__, etc.)
2. Support for default values and field customization
3. Post-initialization processing
4. Easy conversion to dictionaries and tuples
5. Integration with type hints for better IDE support

When to use data classes:
- When you need a simple container for data
- When you want to reduce boilerplate code
- When you need automatic generation of special methods
- When you want to maintain type hints for better IDE support

Data classes are a powerful feature in modern Python that can make your code more
concise, readable, and maintainable.
""")

print("\n===== END OF DATA CLASSES =====")