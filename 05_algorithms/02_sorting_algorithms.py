"""
Python Sorting Algorithms

This module covers common sorting algorithms in Python, including bubble sort,
insertion sort, selection sort, merge sort, quick sort, and their time complexity analysis.
"""

import time
import random
import matplotlib.pyplot as plt

# Bubble Sort
def bubble_sort(arr):
    """
    Sort an array using bubble sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n) when array is already sorted
        - Average Case: O(n²)
        - Worst Case: O(n²)
    Space Complexity: O(1) - in-place sorting
    """
    n = len(arr)
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    # Traverse through all array elements
    for i in range(n):
        # Flag to optimize if no swapping occurs
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
    
    return arr

# Selection Sort
def selection_sort(arr):
    """
    Sort an array using selection sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n²)
        - Average Case: O(n²)
        - Worst Case: O(n²)
    Space Complexity: O(1) - in-place sorting
    """
    n = len(arr)
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    # Traverse through all array elements
    for i in range(n):
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# Insertion Sort
def insertion_sort(arr):
    """
    Sort an array using insertion sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n) when array is already sorted
        - Average Case: O(n²)
        - Worst Case: O(n²)
    Space Complexity: O(1) - in-place sorting
    """
    n = len(arr)
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    # Traverse through 1 to len(arr)
    for i in range(1, n):
        key = arr[i]
        
        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    return arr

# Merge Sort
def merge_sort(arr):
    """
    Sort an array using merge sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n log n)
    Space Complexity: O(n) - requires additional space
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    if len(arr) > 1:
        # Finding the middle of the array
        mid = len(arr) // 2
        
        # Dividing the array elements into 2 halves
        L = arr[:mid]
        R = arr[mid:]
        
        # Sorting the first half
        L = merge_sort(L)
        
        # Sorting the second half
        R = merge_sort(R)
        
        # Merge the sorted halves
        i = j = k = 0
        
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        
        # Check if any elements were left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    
    return arr

# Quick Sort
def quick_sort(arr):
    """
    Sort an array using quick sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n²) when array is already sorted
    Space Complexity: O(log n) - due to recursion stack
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    def _quick_sort(arr, low, high):
        if low < high:
            # Partition the array and get the pivot index
            pivot_index = partition(arr, low, high)
            
            # Sort elements before and after partition
            _quick_sort(arr, low, pivot_index - 1)
            _quick_sort(arr, pivot_index + 1, high)
    
    def partition(arr, low, high):
        # Choose the rightmost element as pivot
        pivot = arr[high]
        
        # Index of smaller element
        i = low - 1
        
        for j in range(low, high):
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                # Increment index of smaller element
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in its correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    _quick_sort(arr, 0, len(arr) - 1)
    return arr

# Heap Sort
def heap_sort(arr):
    """
    Sort an array using heap sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n log n)
    Space Complexity: O(1) - in-place sorting
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    n = len(arr)
    
    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Swap root (maximum element) with the last element
        arr[i], arr[0] = arr[0], arr[i]
        
        # Heapify the reduced heap
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    """
    Heapify subtree rooted at index i.
    
    Args:
        arr: Array representation of heap
        n: Size of the heap
        i: Index of the root of the subtree to heapify
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2
    
    # See if left child of root exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # See if right child of root exists and is greater than root
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        
        # Heapify the affected sub-tree
        heapify(arr, n, largest)

# Counting Sort (for integers with a limited range)
def counting_sort(arr, max_val=None):
    """
    Sort an array using counting sort algorithm.
    
    Args:
        arr: List of non-negative integers to sort
        max_val: Maximum value in the array (optional)
        
    Returns:
        Sorted list
        
    Time Complexity: O(n + k) where k is the range of input
    Space Complexity: O(n + k)
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # Find the maximum value if not provided
    if max_val is None:
        max_val = max(arr)
    
    # Create a count array to store count of each element
    count = [0] * (max_val + 1)
    
    # Store count of each element
    for num in arr:
        count[num] += 1
    
    # Change count[i] so that it contains the position of this element in output array
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build the output array
    output = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
    
    return output

# Radix Sort (for integers)
def radix_sort(arr):
    """
    Sort an array using radix sort algorithm.
    
    Args:
        arr: List of non-negative integers to sort
        
    Returns:
        Sorted list
        
    Time Complexity: O(d * (n + k)) where d is the number of digits and k is the range of each digit
    Space Complexity: O(n + k)
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # Find the maximum number to know number of digits
    max_val = max(arr)
    
    # Do counting sort for every digit
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    """
    Counting sort by digit.
    
    Args:
        arr: Array to sort
        exp: Current digit place value (1, 10, 100, etc.)
    """
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # Store count of occurrences in count[]
    for i in range(n):
        index = arr[i] // exp % 10
        count[index] += 1
    
    # Change count[i] so that count[i] contains the position of this digit in output[]
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build the output array
    for i in range(n - 1, -1, -1):
        index = arr[i] // exp % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    
    # Copy the output array to arr[]
    for i in range(n):
        arr[i] = output[i]

# Bucket Sort
def bucket_sort(arr, num_buckets=10):
    """
    Sort an array using bucket sort algorithm.
    
    Args:
        arr: List of floating point numbers in range [0, 1)
        num_buckets: Number of buckets to use
        
    Returns:
        Sorted list
        
    Time Complexity: 
        - Average Case: O(n + k) where k is the number of buckets
        - Worst Case: O(n²) when all elements are placed in a single bucket
    Space Complexity: O(n + k)
    """
    # Create a copy to avoid modifying the original array
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Put array elements in different buckets
    for num in arr:
        # Ensure num is in [0, 1)
        if not (0 <= num < 1):
            raise ValueError("Bucket sort expects values in range [0, 1)")
        
        bucket_idx = int(num * num_buckets)
        buckets[bucket_idx].append(num)
    
    # Sort individual buckets
    for i in range(num_buckets):
        buckets[i].sort()
    
    # Concatenate all buckets into arr
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result

# Performance comparison
def compare_sorting_algorithms(sizes=[100, 1000, 5000, 10000]):
    """Compare the performance of different sorting algorithms."""
    results = {
        "Bubble Sort": [],
        "Selection Sort": [],
        "Insertion Sort": [],
        "Merge Sort": [],
        "Quick Sort": [],
        "Heap Sort": [],
        "Python's Built-in Sort": []
    }
    
    for size in sizes:
        # Create a random array of the specified size
        arr = random.sample(range(size * 10), size)
        
        # Bubble Sort
        start = time.time()
        bubble_sort(arr)
        results["Bubble Sort"].append(time.time() - start)
        
        # Selection Sort
        start = time.time()
        selection_sort(arr)
        results["Selection Sort"].append(time.time() - start)
        
        # Insertion Sort
        start = time.time()
        insertion_sort(arr)
        results["Insertion Sort"].append(time.time() - start)
        
        # Merge Sort
        start = time.time()
        merge_sort(arr)
        results["Merge Sort"].append(time.time() - start)
        
        # Quick Sort
        start = time.time()
        quick_sort(arr)
        results["Quick Sort"].append(time.time() - start)
        
        # Heap Sort
        start = time.time()
        heap_sort(arr)
        results["Heap Sort"].append(time.time() - start)
        
        # Python's Built-in Sort
        start = time.time()
        sorted(arr)
        results["Python's Built-in Sort"].append(time.time() - start)
    
    return results, sizes

# Visualize the results
def plot_results(results, sizes):
    """Plot the performance results of different sorting algorithms."""
    plt.figure(figsize=(12, 8))
    
    for algorithm, times in results.items():
        plt.plot(sizes, times, marker='o', label=algorithm)
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('Sorting Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('sorting_algorithm_comparison.png')
    plt.close()

# Main demonstration
def main():
    print("Sorting Algorithms Demonstration")
    print("-" * 40)
    
    # Create a random array for testing
    arr = random.sample(range(100), 10)
    print(f"Original array: {arr}")
    
    # Bubble Sort
    sorted_arr = bubble_sort(arr)
    print(f"Bubble Sort: {sorted_arr}")
    
    # Selection Sort
    sorted_arr = selection_sort(arr)
    print(f"Selection Sort: {sorted_arr}")
    
    # Insertion Sort
    sorted_arr = insertion_sort(arr)
    print(f"Insertion Sort: {sorted_arr}")
    
    # Merge Sort
    sorted_arr = merge_sort(arr)
    print(f"Merge Sort: {sorted_arr}")
    
    # Quick Sort
    sorted_arr = quick_sort(arr)
    print(f"Quick Sort: {sorted_arr}")
    
    # Heap Sort
    sorted_arr = heap_sort(arr)
    print(f"Heap Sort: {sorted_arr}")
    
    # Python's Built-in Sort
    sorted_arr = sorted(arr)
    print(f"Python's Built-in Sort: {sorted_arr}")
    
    # Compare performance
    print("\nComparing performance...")
    results, sizes = compare_sorting_algorithms()
    
    # Print results
    print("\nExecution times (seconds):")
    print("-" * 40)
    print(f"{'Array Size':<20}", end="")
    for size in sizes:
        print(f"{size:<15}", end="")
    print()
    
    for algorithm, times in results.items():
        print(f"{algorithm:<20}", end="")
        for time_val in times:
            print(f"{time_val:.8f}    ", end="")
        print()
    
    # Uncomment to generate plot (requires matplotlib)
    # plot_results(results, sizes)
    # print("\nPlot saved as 'sorting_algorithm_comparison.png'")

# EXERCISE 1: Implement a function to sort an array of strings by their length
print("\nEXERCISE 1: Sort Strings by Length")
# Your code here:
# Example solution:
def sort_strings_by_length(strings):
    """
    Sort an array of strings by their length.
    
    Args:
        strings: List of strings to sort
        
    Returns:
        List of strings sorted by length (shortest to longest)
    """
    return sorted(strings, key=len)

# Test the function
words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
print(f"Original strings: {words}")
print(f"Sorted by length: {sort_strings_by_length(words)}")

# EXERCISE 2: Implement a custom sorting function to sort a list of dictionaries by a specific key
print("\nEXERCISE 2: Sort Dictionaries by Key")
# Your code here:
# Example solution:
def sort_dicts_by_key(dict_list, key):
    """
    Sort a list of dictionaries by a specific key.
    
    Args:
        dict_list: List of dictionaries to sort
        key: The key to sort by
        
    Returns:
        List of dictionaries sorted by the specified key
    """
    return sorted(dict_list, key=lambda x: x.get(key, 0))

# Test the function
people = [
    {"name": "Alice", "age": 25, "height": 165},
    {"name": "Bob", "age": 30, "height": 180},
    {"name": "Charlie", "age": 22, "height": 175},
    {"name": "Diana", "age": 28, "height": 170}
]
print(f"Original dictionaries: {people}")
print(f"Sorted by age: {sort_dicts_by_key(people, 'age')}")
print(f"Sorted by height: {sort_dicts_by_key(people, 'height')}")
print(f"Sorted by name: {sort_dicts_by_key(people, 'name')}")

if __name__ == "__main__":
    main()