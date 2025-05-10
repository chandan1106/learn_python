"""
Interview Preparation: Common Patterns and Techniques
"""

# ===== ALGORITHMIC PATTERNS =====
print("\n===== ALGORITHMIC PATTERNS =====")
"""
Recognizing common patterns is key to solving interview problems efficiently.
These patterns appear frequently across different problem types.
"""

# ===== TWO POINTERS PATTERN =====
print("\n===== TWO POINTERS PATTERN =====")
"""
The Two Pointers pattern uses two pointers to iterate through a data structure.
This pattern is especially useful for sorted arrays or linked lists.

Common use cases:
1. Finding a pair with a target sum
2. Removing duplicates
3. Squaring a sorted array
4. Finding triplets with a target sum
5. Dutch national flag problem (sorting 0s, 1s, and 2s)

Example: Check if an array has a pair that sums to a target
"""

def pair_with_target_sum(arr, target_sum):
    """
    Find if there's a pair in sorted array that sums to target_sum.
    Returns the indices of the pair if found, otherwise [-1, -1].
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target_sum:
            return [left, right]  # Found the pair
        
        if current_sum < target_sum:
            left += 1  # Need a larger sum, move left pointer
        else:
            right -= 1  # Need a smaller sum, move right pointer
            
    return [-1, -1]  # No pair found

# Test the two pointers approach
sorted_array = [1, 2, 3, 4, 6, 8, 9]
target = 10
print(f"Array: {sorted_array}, Target: {target}")
print(f"Pair indices: {pair_with_target_sum(sorted_array, target)}")

"""
Example: Remove duplicates from a sorted array in-place
"""

def remove_duplicates(arr):
    """
    Remove duplicates from a sorted array in-place.
    Returns the length of the array after removing duplicates.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not arr:
        return 0
        
    # 'next_non_duplicate' is the position where we'll place the next non-duplicate
    next_non_duplicate = 1
    
    for i in range(1, len(arr)):
        if arr[next_non_duplicate - 1] != arr[i]:
            arr[next_non_duplicate] = arr[i]
            next_non_duplicate += 1
            
    return next_non_duplicate  # Length of array with unique elements

# Test removing duplicates
duplicate_array = [2, 3, 3, 3, 6, 9, 9]
length = remove_duplicates(duplicate_array)
print(f"\nAfter removing duplicates: {duplicate_array[:length]}")
print(f"New length: {length}")

# ===== SLIDING WINDOW PATTERN =====
print("\n===== SLIDING WINDOW PATTERN =====")
"""
The Sliding Window pattern is used to perform operations on a specific window size of an array or string.
This pattern is useful for problems involving subarrays or substrings.

Common use cases:
1. Finding the longest/shortest subarray with a given condition
2. Finding the maximum/minimum sum of a subarray of size K
3. Finding the longest substring with K distinct characters
4. Finding the longest substring without repeating characters

Example: Maximum sum subarray of size K
"""

def max_subarray_sum_of_size_k(arr, k):
    """
    Find the maximum sum of any contiguous subarray of size K.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_sum, window_sum = 0, 0
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]  # Add the next element
        
        # Slide the window once we hit size k
        if window_end >= k - 1:
            max_sum = max(max_sum, window_sum)
            window_sum -= arr[window_start]  # Remove the element going out
            window_start += 1  # Slide the window
            
    return max_sum

# Test the sliding window approach
array = [2, 1, 5, 1, 3, 2]
k = 3
print(f"Array: {array}, k: {k}")
print(f"Maximum sum of subarray of size {k}: {max_subarray_sum_of_size_k(array, k)}")

"""
Example: Longest substring with no more than K distinct characters
"""

def longest_substring_with_k_distinct(str1, k):
    """
    Find the length of the longest substring with no more than K distinct characters.
    
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    window_start = 0
    max_length = 0
    char_frequency = {}
    
    for window_end in range(len(str1)):
        right_char = str1[window_end]
        
        # Add the current character to our frequency map
        if right_char not in char_frequency:
            char_frequency[right_char] = 0
        char_frequency[right_char] += 1
        
        # Shrink the window if we have more than k distinct characters
        while len(char_frequency) > k:
            left_char = str1[window_start]
            char_frequency[left_char] -= 1
            if char_frequency[left_char] == 0:
                del char_frequency[left_char]
            window_start += 1
        
        # Update the maximum length
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length

# Test the sliding window with distinct characters
string = "araaci"
k = 2
print(f"\nString: {string}, k: {k}")
print(f"Length of longest substring with {k} distinct characters: {longest_substring_with_k_distinct(string, k)}")

# ===== FAST AND SLOW POINTERS PATTERN =====
print("\n===== FAST AND SLOW POINTERS PATTERN =====")
"""
The Fast and Slow Pointers pattern (also called Floyd's Tortoise and Hare algorithm)
uses two pointers moving at different speeds to solve problems.

Common use cases:
1. Detecting cycles in a linked list
2. Finding the middle of a linked list
3. Finding if a linked list has a cycle
4. Finding the start of a cycle in a linked list

Example: Detecting a cycle in a linked list
"""

class ListNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

def has_cycle(head):
    """
    Determine if a linked list has a cycle.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next  # Move one step
        fast = fast.next.next  # Move two steps
        
        if slow == fast:  # Found a cycle
            return True
    
    return False  # No cycle

# Create a linked list with a cycle
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)
head.next.next.next.next.next = ListNode(6)

# Create a cycle (6 -> 3)
head.next.next.next.next.next.next = head.next.next

print(f"Linked list has cycle: {has_cycle(head)}")

"""
Example: Finding the middle of a linked list
"""

def find_middle_of_linked_list(head):
    """
    Find the middle node of a linked list.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head:
        return None
    
    slow = head
    fast = head
    
    # When fast reaches the end, slow will be at the middle
    while fast and fast.next:
        slow = slow.next  # Move one step
        fast = fast.next.next  # Move two steps
    
    return slow  # Middle node

# Create a linked list without a cycle
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

middle = find_middle_of_linked_list(head)
print(f"\nMiddle of linked list: {middle.value}")

# ===== MERGE INTERVALS PATTERN =====
print("\n===== MERGE INTERVALS PATTERN =====")
"""
The Merge Intervals pattern deals with overlapping intervals.

Common use cases:
1. Merging overlapping intervals
2. Finding conflicting appointments
3. Finding the minimum number of meeting rooms required
4. Finding the maximum CPU load

Example: Merge overlapping intervals
"""

def merge_intervals(intervals):
    """
    Merge all overlapping intervals.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for the output array
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for i in range(1, len(intervals)):
        current_interval = intervals[i]
        last_merged = merged[-1]
        
        # If current interval overlaps with the last merged interval
        if current_interval[0] <= last_merged[1]:
            # Update the end of the last merged interval if needed
            last_merged[1] = max(last_merged[1], current_interval[1])
        else:
            # Add the current interval to the merged list
            merged.append(current_interval)
    
    return merged

# Test merging intervals
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(f"Intervals: {intervals}")
print(f"Merged intervals: {merge_intervals(intervals)}")

"""
Example: Insert interval into a list of non-overlapping intervals
"""

def insert_interval(intervals, new_interval):
    """
    Insert a new interval into a list of non-overlapping intervals.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for the output array
    """
    merged = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that come before the new interval
    while i < n and intervals[i][1] < new_interval[0]:
        merged.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    # Add the merged interval
    merged.append(new_interval)
    
    # Add all intervals that come after the new interval
    while i < n:
        merged.append(intervals[i])
        i += 1
    
    return merged

# Test inserting an interval
intervals = [[1, 3], [6, 9]]
new_interval = [2, 5]
print(f"\nIntervals: {intervals}")
print(f"New interval: {new_interval}")
print(f"After insertion: {insert_interval(intervals, new_interval)}")

# ===== CYCLIC SORT PATTERN =====
print("\n===== CYCLIC SORT PATTERN =====")
"""
The Cyclic Sort pattern is useful for problems involving arrays containing numbers in a given range.

Common use cases:
1. Finding the missing number in an array
2. Finding the duplicate number in an array
3. Finding all missing numbers in an array
4. Finding the first K missing positive numbers

Example: Find the missing number in an array containing 0 to n
"""

def find_missing_number(nums):
    """
    Find the missing number in an array containing 0 to n with one number missing.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    i = 0
    n = len(nums)
    
    # Cyclic sort
    while i < n:
        j = nums[i]  # Correct position for nums[i]
        
        # If the number is within range and not at its correct position
        if j < n and nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]  # Swap
        else:
            i += 1
    
    # Find the missing number
    for i in range(n):
        if nums[i] != i:
            return i
    
    # If no number is missing in the range 0 to n-1, then n is missing
    return n

# Test finding the missing number
nums = [3, 0, 1]
print(f"Array: {nums}")
print(f"Missing number: {find_missing_number(nums)}")

"""
Example: Find all duplicate numbers in an array
"""

def find_all_duplicates(nums):
    """
    Find all duplicate numbers in an array where each number appears once or twice.
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding the output array
    """
    duplicates = []
    i = 0
    
    while i < len(nums):
        # Correct position for current number (0-indexed)
        correct_idx = nums[i] - 1
        
        if nums[i] != nums[correct_idx]:
            # Swap to put the number at its correct position
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1
    
    # Find numbers that are not at their correct positions
    for i in range(len(nums)):
        if nums[i] != i + 1:
            duplicates.append(nums[i])
    
    return duplicates

# Test finding duplicates
nums = [4, 3, 2, 7, 8, 2, 3, 1]
print(f"\nArray: {nums}")
print(f"Duplicate numbers: {find_all_duplicates(nums)}")

# ===== IN-PLACE REVERSAL OF LINKED LIST PATTERN =====
print("\n===== IN-PLACE REVERSAL OF LINKED LIST PATTERN =====")
"""
The In-place Reversal of Linked List pattern is used to reverse a linked list without using extra space.

Common use cases:
1. Reversing a linked list
2. Reversing a sub-list
3. Reversing every K-element sub-list
4. Rotating a linked list

Example: Reverse a linked list
"""

def reverse_linked_list(head):
    """
    Reverse a linked list in-place.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    current = head
    previous = None
    
    while current:
        # Store the next node
        next_node = current.next
        
        # Reverse the link
        current.next = previous
        
        # Move to the next nodes
        previous = current
        current = next_node
    
    # The new head is the previous last node
    return previous

# Create a linked list: 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

# Print the original linked list
def print_linked_list(head):
    values = []
    current = head
    while current:
        values.append(str(current.value))
        current = current.next
    return " -> ".join(values)

print(f"\nOriginal linked list: {print_linked_list(head)}")

# Reverse the linked list
reversed_head = reverse_linked_list(head)
print(f"Reversed linked list: {print_linked_list(reversed_head)}")

# ===== TREE BREADTH-FIRST SEARCH (BFS) PATTERN =====
print("\n===== TREE BREADTH-FIRST SEARCH (BFS) PATTERN =====")
"""
The Tree BFS pattern uses a queue to traverse a tree level by level.

Common use cases:
1. Level order traversal
2. Finding the minimum depth of a binary tree
3. Connect level order siblings
4. Right view of a binary tree

Example: Level order traversal of a binary tree
"""

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def level_order_traversal(root):
    """
    Perform level order traversal of a binary tree.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            current_node = queue.pop(0)
            current_level.append(current_node.value)
            
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        
        result.append(current_level)
    
    return result

# Create a binary tree
#       1
#      / \
#     2   3
#    / \   \
#   4   5   6
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.right = TreeNode(6)

print(f"\nLevel order traversal: {level_order_traversal(root)}")

# ===== TREE DEPTH-FIRST SEARCH (DFS) PATTERN =====
print("\n===== TREE DEPTH-FIRST SEARCH (DFS) PATTERN =====")
"""
The Tree DFS pattern uses recursion or a stack to traverse a tree depth-first.

Common use cases:
1. Binary tree path sum
2. All paths for a sum
3. Path with maximum sum
4. Diameter of a binary tree

Example: Binary tree path sum
"""

def has_path_sum(root, target_sum):
    """
    Check if the tree has a root-to-leaf path with the given sum.
    
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    if not root:
        return False
    
    # If it's a leaf node, check if the value equals the remaining sum
    if not root.left and not root.right:
        return root.value == target_sum
    
    # Recursively check left and right subtrees with reduced sum
    return (has_path_sum(root.left, target_sum - root.value) or
            has_path_sum(root.right, target_sum - root.value))

# Test path sum
target_sum = 8  # Path 1 -> 2 -> 5 = 8
print(f"Has path with sum {target_sum}: {has_path_sum(root, target_sum)}")

"""
Example: All paths with a given sum
"""

def find_paths(root, target_sum):
    """
    Find all root-to-leaf paths with the given sum.
    
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    all_paths = []
    find_paths_recursive(root, target_sum, [], all_paths)
    return all_paths

def find_paths_recursive(current_node, target_sum, current_path, all_paths):
    if not current_node:
        return
    
    # Add the current node to the path
    current_path.append(current_node.value)
    
    # If it's a leaf node and the sum matches
    if not current_node.left and not current_node.right and current_node.value == target_sum:
        all_paths.append(list(current_path))
    else:
        # Recursively search left and right subtrees
        find_paths_recursive(current_node.left, target_sum - current_node.value, current_path, all_paths)
        find_paths_recursive(current_node.right, target_sum - current_node.value, current_path, all_paths)
    
    # Backtrack
    current_path.pop()

# Create a simple binary tree for path sum
#       5
#      / \
#     4   8
#    /   / \
#   11  13  4
#  / \     / \
# 7   2   5   1
root = TreeNode(5)
root.left = TreeNode(4)
root.right = TreeNode(8)
root.left.left = TreeNode(11)
root.left.left.left = TreeNode(7)
root.left.left.right = TreeNode(2)
root.right.left = TreeNode(13)
root.right.right = TreeNode(4)
root.right.right.left = TreeNode(5)
root.right.right.right = TreeNode(1)

target_sum = 22  # Paths: 5 -> 4 -> 11 -> 2 and 5 -> 8 -> 4 -> 5
print(f"\nPaths with sum {target_sum}: {find_paths(root, target_sum)}")

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Understanding these common patterns will help you solve a wide range of interview problems.
When approaching a new problem:

1. Try to identify if it fits one of these patterns
2. Apply the pattern-specific techniques
3. Adapt the solution to the specific requirements of the problem

Remember that many interview problems are variations or combinations of these patterns.
With practice, you'll become more efficient at recognizing and applying them.
""")

print("\n===== END OF COMMON PATTERNS AND TECHNIQUES =====")