"""
Modern Python Features: Asynchronous Programming with asyncio
"""

# ===== INTRODUCTION TO ASYNC PROGRAMMING =====
print("\n===== INTRODUCTION TO ASYNC PROGRAMMING =====")
"""
Asynchronous programming allows you to write concurrent code that can perform multiple
operations without blocking the execution flow. Python's asyncio module (introduced in Python 3.4)
provides tools for writing single-threaded concurrent code using coroutines.

Key concepts:
1. Coroutines - Functions that can pause execution and yield control
2. Event Loop - Manages and distributes the execution of different coroutines
3. Awaitables - Objects that can be used with the await expression
4. Tasks - Wrappers around coroutines to track their execution

Benefits:
1. Improved performance for I/O-bound operations
2. Better resource utilization
3. Simplified concurrent code compared to threads
4. Reduced need for callbacks and complex synchronization
"""

# ===== COROUTINES AND ASYNC/AWAIT SYNTAX =====
print("\n===== COROUTINES AND ASYNC/AWAIT SYNTAX =====")
"""
The async/await syntax (introduced in Python 3.5) provides a clean way to define and use coroutines.

- async def: Defines a coroutine function
- await: Pauses the coroutine until the awaitable completes
"""

import asyncio

# Define a simple coroutine
async def hello_world():
    print("Hello")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("World")

# Define another coroutine that calls the first one
async def main():
    await hello_world()  # Wait for hello_world to complete
    
    # Run multiple coroutines concurrently
    await asyncio.gather(
        hello_world(),
        hello_world()
    )

# Run the event loop
print("Running the event loop:")
asyncio.run(main())  # Python 3.7+ simplified API

# ===== WORKING WITH TASKS =====
print("\n===== WORKING WITH TASKS =====")
"""
Tasks are used to schedule coroutines concurrently. A Task is a subclass of Future that
wraps a coroutine and allows you to track its execution.
"""

async def count_down(name, start):
    print(f"{name} starting countdown from {start}")
    for i in range(start, 0, -1):
        print(f"{name}: {i}")
        await asyncio.sleep(0.2)
    print(f"{name}: Done!")
    return name

async def task_demo():
    # Create tasks
    task1 = asyncio.create_task(count_down("Task A", 3))
    task2 = asyncio.create_task(count_down("Task B", 5))
    
    # Wait for both tasks to complete
    print("Waiting for tasks to complete...")
    results = await asyncio.gather(task1, task2)
    print(f"Tasks completed with results: {results}")
    
    # Create a task and cancel it
    task3 = asyncio.create_task(count_down("Task C", 10))
    await asyncio.sleep(0.5)  # Let it run a bit
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        print("Task C was cancelled")

print("Running task demo:")
asyncio.run(task_demo())

# ===== HANDLING CONCURRENT TASKS =====
print("\n===== HANDLING CONCURRENT TASKS =====")
"""
asyncio provides several ways to run coroutines concurrently:

1. gather() - Run awaitables concurrently and wait for all of them
2. wait() - Wait for awaitables with timeout options
3. as_completed() - Iterate over awaitables as they complete
"""

async def fetch_data(name, delay):
    print(f"Fetching data from {name}...")
    await asyncio.sleep(delay)  # Simulate network delay
    print(f"Data from {name} fetched!")
    return f"Data from {name}"

async def concurrent_demo():
    # Using gather
    print("\nUsing asyncio.gather():")
    results = await asyncio.gather(
        fetch_data("API 1", 1),
        fetch_data("API 2", 2),
        fetch_data("API 3", 1.5)
    )
    print(f"All results: {results}")
    
    # Using wait with timeout
    print("\nUsing asyncio.wait() with timeout:")
    tasks = [
        asyncio.create_task(fetch_data("Service 1", 1)),
        asyncio.create_task(fetch_data("Service 2", 3)),
        asyncio.create_task(fetch_data("Service 3", 2))
    ]
    
    done, pending = await asyncio.wait(tasks, timeout=2.5)
    print(f"Completed tasks: {len(done)}")
    print(f"Pending tasks: {len(pending)}")
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    
    # Using as_completed
    print("\nUsing asyncio.as_completed():")
    tasks = [
        fetch_data("Source 1", 2),
        fetch_data("Source 2", 1),
        fetch_data("Source 3", 3)
    ]
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Result received: {result}")

print("Running concurrent demo:")
asyncio.run(concurrent_demo())

# ===== ERROR HANDLING =====
print("\n===== ERROR HANDLING =====")
"""
Error handling in async code is similar to synchronous code, but there are some
special considerations for tasks and coroutines.
"""

async def might_fail(name, should_fail):
    print(f"Starting {name}")
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError(f"Error in {name}")
    return f"Success from {name}"

async def error_handling_demo():
    # Try/except with a single coroutine
    try:
        result = await might_fail("Coroutine A", True)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Caught exception: {e}")
    
    # Error handling with gather
    print("\nError handling with gather:")
    results = await asyncio.gather(
        might_fail("Coroutine B", False),
        might_fail("Coroutine C", True),
        might_fail("Coroutine D", False),
        return_exceptions=True  # Important: prevents exceptions from propagating
    )
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed with: {result}")
        else:
            print(f"Task {i} succeeded with: {result}")
    
    # Without return_exceptions, the first exception would propagate
    print("\nWithout return_exceptions:")
    try:
        await asyncio.gather(
            might_fail("Coroutine E", False),
            might_fail("Coroutine F", True),
            might_fail("Coroutine G", False)
        )
    except ValueError as e:
        print(f"Gather failed with: {e}")

print("Running error handling demo:")
asyncio.run(error_handling_demo())

# ===== ASYNC CONTEXT MANAGERS AND ITERATORS =====
print("\n===== ASYNC CONTEXT MANAGERS AND ITERATORS =====")
"""
Python supports async versions of context managers and iterators:

- async with: For asynchronous context managers
- async for: For asynchronous iterators
"""

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource asynchronously...")
        await asyncio.sleep(1)
        print("Resource acquired")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource asynchronously...")
        await asyncio.sleep(0.5)
        print("Resource released")
        return False  # Don't suppress exceptions
    
    async def process(self):
        print("Processing with resource")
        await asyncio.sleep(0.5)

class AsyncCounter:
    def __init__(self, limit):
        self.limit = limit
        self.counter = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.counter >= self.limit:
            raise StopAsyncIteration
        self.counter += 1
        await asyncio.sleep(0.2)
        return self.counter

async def context_iterator_demo():
    # Async context manager
    print("\nAsync context manager:")
    async with AsyncResource() as resource:
        await resource.process()
    
    # Async iterator
    print("\nAsync iterator:")
    async for i in AsyncCounter(5):
        print(f"Count: {i}")

print("Running async context manager and iterator demo:")
asyncio.run(context_iterator_demo())

# ===== PRACTICAL EXAMPLE: WEB SCRAPING =====
print("\n===== PRACTICAL EXAMPLE: WEB SCRAPING =====")
"""
A practical example of using asyncio for concurrent web requests.
Note: This example requires the aiohttp library (pip install aiohttp).
"""

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not installed. Skipping web scraping example.")

if AIOHTTP_AVAILABLE:
    async def fetch_url(session, url):
        print(f"Fetching {url}")
        async with session.get(url) as response:
            return await response.text()

    async def web_scraping_demo():
        urls = [
            "https://example.com",
            "https://python.org",
            "https://github.com"
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    print(f"Error fetching {url}: {result}")
                else:
                    print(f"Successfully fetched {url}, content length: {len(result)} bytes")
    
    print("Running web scraping demo:")
    # Uncomment to run the actual web requests:
    # asyncio.run(web_scraping_demo())
    print("(Web scraping demo commented out to avoid making actual web requests)")

# ===== ASYNCIO BEST PRACTICES =====
print("\n===== ASYNCIO BEST PRACTICES =====")
"""
Best practices for working with asyncio:

1. Don't block the event loop
   - Avoid CPU-intensive operations in coroutines
   - Use run_in_executor for CPU-bound tasks or blocking I/O

2. Handle exceptions properly
   - Use try/except in coroutines
   - Use return_exceptions=True with gather() when appropriate

3. Use asyncio.create_task() for fire-and-forget operations
   - Keep references to tasks if you need to cancel them later

4. Set timeouts for network operations
   - Use asyncio.wait_for() to prevent hanging on slow operations

5. Use asyncio.shield() to protect critical tasks from cancellation

6. Consider using higher-level abstractions
   - aiohttp for HTTP requests
   - asyncpg for PostgreSQL
   - motor for MongoDB

7. Be careful with synchronization
   - Use asyncio.Lock(), asyncio.Semaphore(), etc. for coordination
   - Avoid mixing threads and asyncio without proper synchronization
"""

# Example of running a blocking function in a thread pool
import time

def blocking_function(name, seconds):
    print(f"{name} started (blocking)")
    time.sleep(seconds)  # This would block the event loop if called directly
    print(f"{name} finished (blocking)")
    return f"Result from {name}"

async def non_blocking_demo():
    print("\nRunning blocking functions in thread pool:")
    loop = asyncio.get_running_loop()
    
    # Run blocking function in the default thread pool
    result1 = await loop.run_in_executor(
        None,  # Use default executor
        blocking_function,
        "Task 1",
        2
    )
    
    # Run multiple blocking functions concurrently
    results = await asyncio.gather(
        loop.run_in_executor(None, blocking_function, "Task 2", 1),
        loop.run_in_executor(None, blocking_function, "Task 3", 3)
    )
    
    print(f"Results: {[result1] + results}")

print("Running non-blocking demo:")
asyncio.run(non_blocking_demo())

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Asynchronous programming with asyncio is a powerful paradigm for writing concurrent code in Python.
It's particularly well-suited for I/O-bound applications like web servers, API clients, and database access.

Key takeaways:
1. Use async/await for clean, readable asynchronous code
2. Leverage asyncio.gather() for concurrent execution
3. Handle exceptions properly in asynchronous code
4. Use appropriate timeouts to prevent hanging
5. Run CPU-bound tasks in separate processes or thread pools

While asyncio has a learning curve, it provides significant benefits for applications
that need to handle many concurrent operations efficiently.
""")

print("\n===== END OF ASYNC PROGRAMMING =====")