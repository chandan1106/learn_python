"""
Python Searching Algorithms

This module covers common searching algorithms in Python, including linear search,
binary search, and their time complexity analysis.
"""

import time
import random
import matplotlib.pyplot as plt

# Linear Search
def linear_search(arr, target):
    """
    Search for target in arr using linear search.
    
    Args:
        arr: List of elements to search through
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(n) - where n is the length of the array
    Space Complexity: O(1) - constant space
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Binary Search (Iterative)
def binary_search_iterative(arr, target):
    """
    Search for target in a sorted arr using binary search (iterative).
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n) - where n is the length of the array
    Space Complexity: O(1) - constant space
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if target is at the middle
        if arr[mid] == target:
            return mid
        
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
        
        # If target is smaller, ignore right half
        else:
            right = mid - 1
    
    # Target not found
    return -1

# Binary Search (Recursive)
def binary_search_recursive(arr, target, left=None, right=None):
    """
    Search for target in a sorted arr using binary search (recursive).
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        left: Left boundary index (default: 0)
        right: Right boundary index (default: len(arr)-1)
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n) - where n is the length of the array
    Space Complexity: O(log n) - due to recursive call stack
    """
    # Initialize left and right for first call
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # Base case: element not found
    if left > right:
        return -1
    
    # Find middle element
    mid = (left + right) // 2
    
    # If element is at mid, return mid
    if arr[mid] == target:
        return mid
    
    # If element is smaller than mid, search in left subarray
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    
    # If element is larger than mid, search in right subarray
    else:
        return binary_search_recursive(arr, target, mid + 1, right)

# Jump Search
def jump_search(arr, target):
    """
    Search for target in a sorted arr using jump search.
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(√n) - where n is the length of the array
    Space Complexity: O(1) - constant space
    """
    import math
    
    n = len(arr)
    # Finding block size to be jumped
    step = int(math.sqrt(n))
    
    # Finding the block where element is present (if it is present)
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Doing a linear search for target in block beginning with prev
    while arr[prev] < target:
        prev += 1
        
        # If we reached next block or end of array, element is not present
        if prev == min(step, n):
            return -1
    
    # If element is found
    if arr[prev] == target:
        return prev
    
    return -1

# Interpolation Search
def interpolation_search(arr, target):
    """
    Search for target in a sorted arr using interpolation search.
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: 
        - Average case: O(log log n) if elements are uniformly distributed
        - Worst case: O(n) when elements are not uniformly distributed
    Space Complexity: O(1) - constant space
    """
    low = 0
    high = len(arr) - 1
    
    while low <= high and target >= arr[low] and target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        
        # Probing the position with uniform distribution
        pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        
        # Target found
        if arr[pos] == target:
            return pos
        
        # If target is larger, target is in right subarray
        if arr[pos] < target:
            low = pos + 1
        
        # If target is smaller, target is in left subarray
        else:
            high = pos - 1
    
    return -1

# Exponential Search
def exponential_search(arr, target):
    """
    Search for target in a sorted arr using exponential search.
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n) - where n is the length of the array
    Space Complexity: O(1) - constant space (when using iterative binary search)
    """
    n = len(arr)
    
    # If target is at the first position
    if arr[0] == target:
        return 0
    
    # Find range for binary search by repeated doubling
    i = 1
    while i < n and arr[i] <= target:
        i = i * 2
    
    # Call binary search for the found range
    return binary_search_iterative(arr, target, i // 2, min(i, n - 1))

# Helper function for exponential search
def binary_search_iterative(arr, target, left, right):
    """Helper binary search function with specified boundaries."""
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        
        elif arr[mid] < target:
            left = mid + 1
        
        else:
            right = mid - 1
    
    return -1

# Performance comparison
def compare_search_algorithms(sizes=[100, 1000, 10000, 100000]):
    """Compare the performance of different search algorithms."""
    results = {
        "Linear Search": [],
        "Binary Search (Iterative)": [],
        "Binary Search (Recursive)": [],
        "Jump Search": [],
        "Interpolation Search": [],
        "Exponential Search": []
    }
    
    for size in sizes:
        # Create a sorted array of the specified size
        arr = sorted(random.sample(range(size * 10), size))
        
        # Target to search for (existing element)
        target = arr[random.randint(0, size - 1)]
        
        # Linear Search
        start = time.time()
        linear_search(arr, target)
        results["Linear Search"].append(time.time() - start)
        
        # Binary Search (Iterative)
        start = time.time()
        binary_search_iterative(arr, target)
        results["Binary Search (Iterative)"].append(time.time() - start)
        
        # Binary Search (Recursive)
        start = time.time()
        binary_search_recursive(arr, target)
        results["Binary Search (Recursive)"].append(time.time() - start)
        
        # Jump Search
        start = time.time()
        jump_search(arr, target)
        results["Jump Search"].append(time.time() - start)
        
        # Interpolation Search
        start = time.time()
        interpolation_search(arr, target)
        results["Interpolation Search"].append(time.time() - start)
        
        # Exponential Search
        start = time.time()
        exponential_search(arr, target)
        results["Exponential Search"].append(time.time() - start)
    
    return results, sizes

# Visualize the results
def plot_results(results, sizes):
    """Plot the performance results of different search algorithms."""
    plt.figure(figsize=(12, 8))
    
    for algorithm, times in results.items():
        plt.plot(sizes, times, marker='o', label=algorithm)
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('Search Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('search_algorithm_comparison.png')
    plt.close()

# Main demonstration
def main():
    print("Searching Algorithms Demonstration")
    print("-" * 40)
    
    # Create a sorted array for testing
    arr = sorted(random.sample(range(1000), 100))
    print(f"Array (first 10 elements): {arr[:10]}...")
    
    # Choose a target that exists in the array
    target = arr[random.randint(0, 99)]
    print(f"Target: {target}")
    
    # Linear Search
    index = linear_search(arr, target)
    print(f"Linear Search: Found at index {index}")
    
    # Binary Search (Iterative)
    index = binary_search_iterative(arr, target)
    print(f"Binary Search (Iterative): Found at index {index}")
    
    # Binary Search (Recursive)
    index = binary_search_recursive(arr, target)
    print(f"Binary Search (Recursive): Found at index {index}")
    
    # Jump Search
    index = jump_search(arr, target)
    print(f"Jump Search: Found at index {index}")
    
    # Interpolation Search
    index = interpolation_search(arr, target)
    print(f"Interpolation Search: Found at index {index}")
    
    # Exponential Search
    index = exponential_search(arr, target)
    print(f"Exponential Search: Found at index {index}")
    
    # Compare performance
    print("\nComparing performance...")
    results, sizes = compare_search_algorithms()
    
    # Print results
    print("\nExecution times (seconds):")
    print("-" * 40)
    print(f"{'Array Size':<15}", end="")
    for size in sizes:
        print(f"{size:<15}", end="")
    print()
    
    for algorithm, times in results.items():
        print(f"{algorithm:<15}", end="")
        for time_val in times:
            print(f"{time_val:.8f}    ", end="")
        print()
    
    # Uncomment to generate plot (requires matplotlib)
    # plot_results(results, sizes)
    # print("\nPlot saved as 'search_algorithm_comparison.png'")

# EXERCISE 1: Implement a function to find the first occurrence of a target in a sorted array with duplicates
print("\nEXERCISE 1: Find First Occurrence")
# Your code here:
# Example solution:
def find_first_occurrence(arr, target):
    """
    Find the index of the first occurrence of target in a sorted array.
    
    Args:
        arr: Sorted list of elements to search through
        target: Element to search for
        
    Returns:
        Index of first occurrence of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        # If target is found, look in the left half for earlier occurrences
        if arr[mid] == target:
            result = mid
            right = mid - 1
        
        # If target is greater, look in the right half
        elif arr[mid] < target:
            left = mid + 1
        
        # If target is smaller, look in the left half
        else:
            right = mid - 1
    
    return result

# Test the function
test_arr = [1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6]
target = 5
print(f"Array: {test_arr}")
print(f"First occurrence of {target} is at index: {find_first_occurrence(test_arr, target)}")

# EXERCISE 2: Implement a search algorithm for a rotated sorted array
print("\nEXERCISE 2: Search in Rotated Sorted Array")
# Your code here:
# Example solution:
def search_rotated_array(arr, target):
    """
    Search for target in a rotated sorted array.
    
    Args:
        arr: Rotated sorted array
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # If target is found at mid
        if arr[mid] == target:
            return mid
        
        # If left half is sorted
        if arr[left] <= arr[mid]:
            # If target is in the left half
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        
        # If right half is sorted
        else:
            # If target is in the right half
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Test the function
rotated_arr = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(f"Rotated array: {rotated_arr}")
print(f"Index of {target} in rotated array: {search_rotated_array(rotated_arr, target)}")

if __name__ == "__main__":
    main()