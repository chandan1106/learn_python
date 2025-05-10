"""
Advanced Python Concepts: Context Managers
"""

# ===== CONTEXT MANAGERS =====
print("===== CONTEXT MANAGERS =====")

# Using built-in context managers
print("Using built-in context manager (file handling):")
try:
    with open("example.txt", "w") as file:
        file.write("Hello, context managers!")
    print("File written successfully")
    
    with open("example.txt", "r") as file:
        content = file.read()
        print(f"File content: {content}")
except FileNotFoundError:
    print("File operation simulation (no actual file created in this example)")

# Creating context managers using classes
print("\nCreating a context manager using a class:")

class DatabaseConnection:
    """A simple database connection context manager"""
    
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.connection = None
    
    def __enter__(self):
        """Method called when entering the context"""
        print(f"Connecting to database at {self.host}...")
        # In a real scenario, this would establish a database connection
        self.connection = {"status": "connected", "host": self.host}
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Method called when exiting the context"""
        print(f"Closing database connection to {self.host}...")
        # In a real scenario, this would close the database connection
        self.connection = None
        
        # Return True to suppress exceptions, False to propagate them
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            return False  # Propagate the exception
        return True

# Using our custom context manager
with DatabaseConnection("localhost", "user", "password") as conn:
    print(f"Connection established: {conn}")
    # Perform database operations here
print()

# Creating context managers using contextlib
print("Creating a context manager using contextlib:")
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    """A simple file manager using contextlib"""
    try:
        print(f"Opening file {filename} in {mode} mode")
        # In a real scenario, this would open an actual file
        file = {"name": filename, "mode": mode, "content": ""}
        yield file
        print(f"File operations completed on {filename}")
    finally:
        print(f"Closing file {filename}")
        file = None

# Using our contextlib-based context manager
with file_manager("data.txt", "w") as f:
    print(f"Working with file: {f['name']}")
    f["content"] = "Some data"
    print(f"Wrote content: {f['content']}")
print()

# Nested context managers
print("Nested context managers:")
try:
    with open("outer.txt", "w") as outer_file:
        with open("inner.txt", "w") as inner_file:
            outer_file.write("Outer content")
            inner_file.write("Inner content")
    print("Nested file operations completed")
except FileNotFoundError:
    print("Nested file operation simulation (no actual files created)")
print()

# Context manager for resource cleanup
print("Context manager for resource cleanup:")

class Resource:
    """A resource that needs proper cleanup"""
    
    def __init__(self, name):
        self.name = name
        print(f"Resource {name} initialized")
    
    def cleanup(self):
        print(f"Resource {self.name} cleaned up")

@contextmanager
def resource_manager(name):
    """Ensure resource cleanup happens"""
    resource = Resource(name)
    try:
        yield resource
    finally:
        resource.cleanup()

# Using the resource manager
with resource_manager("database") as db:
    print(f"Working with {db.name}")
    # Simulate an exception
    if True:  # Change to True to see exception handling
        print("Operations completed successfully")
    else:
        raise Exception("Something went wrong!")
print()

# Context manager for timing code execution
print("Context manager for timing code execution:")
import time

@contextmanager
def timer(name):
    """Time the execution of a code block"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{name} took {end_time - start_time:.6f} seconds to execute")

# Using the timer context manager
with timer("Sleep operation"):
    print("Starting a time-consuming operation...")
    time.sleep(0.5)  # Simulate a time-consuming operation
    print("Operation completed")
print()

# Context manager for changing directory
print("Context manager for changing directory:")
import os

@contextmanager
def change_directory(new_dir):
    """Temporarily change the working directory"""
    old_dir = os.getcwd()
    try:
        print(f"Changing directory from {old_dir} to {new_dir}")
        # In a real scenario: os.chdir(new_dir)
        yield
    finally:
        print(f"Changing back to {old_dir}")
        # In a real scenario: os.chdir(old_dir)

# Using the directory change context manager
with change_directory("/tmp"):
    print("Working in temporary directory")
    # Do operations in the temporary directory
print("Back to original directory")