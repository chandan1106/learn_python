"""
Modern Python Features: Type Hints and Static Type Checking
"""

# ===== INTRODUCTION TO TYPE HINTS =====
print("\n===== INTRODUCTION TO TYPE HINTS =====")
"""
Type hints (introduced in PEP 484) allow you to annotate variables, function parameters,
and return values with their expected types. This helps with:

1. Code readability - Makes code intentions clearer
2. IDE support - Better autocompletion and error detection
3. Static type checking - Catch type errors before runtime
4. Documentation - Self-documenting code

Type hints are optional and don't affect runtime behavior - they're just hints.
"""

# ===== BASIC TYPE ANNOTATIONS =====
print("\n===== BASIC TYPE ANNOTATIONS =====")

# Variable annotations
age: int = 25
name: str = "Alice"
is_active: bool = True
height: float = 5.9

# Function annotations
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

# Using the functions
print(greet("World"))
print(add(5, 3))

# ===== COMPLEX TYPE ANNOTATIONS =====
print("\n===== COMPLEX TYPE ANNOTATIONS =====")

from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable

# List type
def process_names(names: List[str]) -> List[str]:
    return [name.upper() for name in names]

# Dictionary type
def count_words(text: str) -> Dict[str, int]:
    words = text.lower().split()
    return {word: words.count(word) for word in set(words)}

# Tuple type
def get_coordinates() -> Tuple[float, float]:
    return (10.5, 20.3)

# Set type
def unique_characters(text: str) -> Set[str]:
    return set(text)

# Optional type (something or None)
def find_user(user_id: int) -> Optional[Dict[str, Any]]:
    users = {1: {"name": "Alice", "age": 30}}
    return users.get(user_id)

# Union type (multiple possible types)
def process_id(user_id: Union[int, str]) -> str:
    return f"Processing ID: {user_id}"

# Callable type (functions)
def execute_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Examples
names = ["alice", "bob", "charlie"]
print(f"Processed names: {process_names(names)}")

text = "the quick brown fox jumps over the lazy dog"
print(f"Word count: {count_words(text)}")

coords = get_coordinates()
print(f"Coordinates: {coords}")

print(f"Unique characters in 'hello': {unique_characters('hello')}")

user = find_user(1)
print(f"Found user: {user}")

print(f"Process ID result: {process_id(123)}")
print(f"Process ID result: {process_id('ABC123')}")

print(f"Execute function result: {execute_function(lambda x, y: x * y, 5, 3)}")

# ===== TYPE ALIASES =====
print("\n===== TYPE ALIASES =====")

from typing import NewType, TypeVar, Generic

# Type aliases for complex types
UserId = int
UserDict = Dict[UserId, Dict[str, Any]]

def get_user(users: UserDict, user_id: UserId) -> Optional[Dict[str, Any]]:
    return users.get(user_id)

# NewType creates a distinct type
AdminId = NewType('AdminId', int)

def get_admin(admin_id: AdminId) -> Dict[str, Any]:
    return {"id": admin_id, "role": "admin"}

# Using the types
users_db: UserDict = {
    1: {"name": "Alice", "email": "alice@example.com"},
    2: {"name": "Bob", "email": "bob@example.com"}
}

print(f"User 1: {get_user(users_db, 1)}")

# Create an AdminId (note the constructor call)
admin_id = AdminId(5)
print(f"Admin: {get_admin(admin_id)}")

# ===== GENERICS =====
print("\n===== GENERICS =====")

T = TypeVar('T')  # Define a type variable

class Box(Generic[T]):
    def __init__(self, content: T):
        self.content = content
        
    def get_content(self) -> T:
        return self.content

# Using the generic class
int_box = Box[int](123)
str_box = Box[str]("Hello")

print(f"Int box content: {int_box.get_content()}")
print(f"String box content: {str_box.get_content()}")

# Generic function
def first_element(items: List[T]) -> Optional[T]:
    return items[0] if items else None

print(f"First element of [1, 2, 3]: {first_element([1, 2, 3])}")
print(f"First element of ['a', 'b', 'c']: {first_element(['a', 'b', 'c'])}")

# ===== PROTOCOLS AND STRUCTURAL SUBTYPING =====
print("\n===== PROTOCOLS AND STRUCTURAL SUBTYPING =====")

from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:
    def draw(self) -> None:
        print("Drawing a circle")

class Square:
    def draw(self) -> None:
        print("Drawing a square")

class Triangle:
    def calculate_area(self) -> float:
        return 0.0  # Doesn't implement draw()

def render(drawable: Drawable) -> None:
    drawable.draw()

# Using the protocol
circle = Circle()
square = Square()
triangle = Triangle()

render(circle)  # Works
render(square)  # Works
# render(triangle)  # Would fail type checking and runtime check

print(f"Is circle Drawable? {isinstance(circle, Drawable)}")
print(f"Is triangle Drawable? {isinstance(triangle, Drawable)}")

# ===== TYPE CHECKING =====
print("\n===== TYPE CHECKING =====")
"""
To perform static type checking, you can use tools like mypy, pyright, or pyre.

Example command:
```
mypy your_file.py
```

Type checking can catch errors like:
- Passing a string where an int is expected
- Calling a method that doesn't exist on a type
- Returning the wrong type from a function
- Missing required arguments

Runtime type checking is also possible with libraries like 'typeguard'.
"""

# Conditional imports for type checking
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imports only used during type checking, not at runtime
    from some_module import SomeClass

# Type checking guards
def process_data(data: Any) -> None:
    if isinstance(data, str):
        # Type checker knows data is str in this block
        print(f"String length: {len(data)}")
    elif isinstance(data, list):
        # Type checker knows data is list in this block
        print(f"List length: {len(data)}")

# ===== TYPING BEST PRACTICES =====
print("\n===== TYPING BEST PRACTICES =====")
"""
1. Start with critical code paths
   - You don't need to add types to everything at once
   - Focus on public APIs and complex functions

2. Use Optional for values that might be None
   - Use Optional[str] instead of Union[str, None]

3. Be specific with container types
   - Use List[int] instead of just List
   - Use Dict[str, int] instead of just Dict

4. Use Any sparingly
   - Any opts out of type checking
   - Use it only when necessary

5. Consider using TypedDict for dictionaries with specific structures
   - Helps document and enforce dictionary structure

6. Use Literal for functions that accept specific string/numeric values
   - Helps catch invalid arguments

7. Add type stubs (.pyi files) for untyped third-party libraries
   - Allows type checking code that uses these libraries

8. Run a type checker regularly
   - Integrate it into your CI/CD pipeline
"""

from typing import TypedDict, Literal

# TypedDict example
class UserProfile(TypedDict):
    name: str
    age: int
    email: str
    is_active: bool

def create_user(profile: UserProfile) -> None:
    print(f"Created user: {profile['name']}")

# Literal example
def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    print(f"Setting log level to {level}")

# Using the typed functions
user_profile: UserProfile = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com",
    "is_active": True
}

create_user(user_profile)
set_log_level("INFO")
# set_log_level("INVALID")  # Would fail type checking

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Type hints are a powerful feature in modern Python that improve code quality by:

1. Making code more readable and self-documenting
2. Enabling better IDE support with autocompletion and error detection
3. Allowing static type checking to catch errors before runtime
4. Facilitating better refactoring tools

While Python remains a dynamically typed language, type hints provide many of the benefits
of static typing without sacrificing Python's flexibility and ease of use.

To get the most out of type hints:
- Use a static type checker like mypy
- Configure your IDE to use type information
- Add types gradually to your codebase, focusing on critical areas first
- Consider type hints as living documentation for your code
""")

print("\n===== END OF TYPE HINTS =====")