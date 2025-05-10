"""
Interview Preparation: Systematic Problem Solving Approach
"""

# ===== PROBLEM SOLVING METHODOLOGY =====
print("\n===== PROBLEM SOLVING METHODOLOGY =====")
"""
A systematic approach to solving coding problems is essential for interviews.
This methodology works for basic, intermediate, and advanced problems.

1. UNDERSTAND THE PROBLEM
   - Read the problem statement carefully, multiple times if needed
   - Identify the inputs and expected outputs
   - Clarify constraints and edge cases
   - Ask questions if anything is unclear
   - Restate the problem in your own words to confirm understanding

2. EXPLORE EXAMPLES
   - Work through the provided examples step by step
   - Create additional examples, including edge cases:
     * Empty inputs
     * Minimal cases (single element)
     * Typical cases
     * Complex cases
     * Invalid inputs
   - Trace through examples manually to understand the pattern

3. BREAK DOWN THE PROBLEM
   - Divide the problem into smaller sub-problems
   - Identify patterns or similarities to known problems
   - Consider different approaches:
     * Brute force solution
     * Optimized algorithms
     * Data structures that might help

4. SOLVE OR SIMPLIFY
   - Start with a simple, working solution (even if inefficient)
   - If stuck, solve a simpler version of the problem first
   - Build up from the simplified version to the full solution
   - Focus on correctness before optimization

5. IMPLEMENT THE SOLUTION
   - Write clean, readable code
   - Use meaningful variable names
   - Add comments for complex logic
   - Break down complex operations into helper functions

6. TEST YOUR SOLUTION
   - Test with the examples provided
   - Test with your additional examples and edge cases
   - Trace through the code execution manually
   - Debug any issues methodically

7. ANALYZE & OPTIMIZE
   - Analyze time and space complexity
   - Look for redundant operations or calculations
   - Consider more efficient data structures
   - Apply optimization techniques without sacrificing readability

8. REFLECT
   - Explain your approach and reasoning
   - Discuss alternative approaches
   - Highlight trade-offs between different solutions
"""

# ===== EXAMPLE: APPLYING THE METHODOLOGY =====
print("\n===== EXAMPLE: APPLYING THE METHODOLOGY =====")
"""
Problem: Find the two numbers in an array that add up to a target sum.

1. UNDERSTAND THE PROBLEM
   - Input: An array of integers and a target sum
   - Output: Indices of the two numbers that add up to the target
   - Constraints: Exactly one valid solution exists
   - Example: [2, 7, 11, 15], target = 9 → Output: [0, 1] (2 + 7 = 9)

2. EXPLORE EXAMPLES
   - Example 1: [2, 7, 11, 15], target = 9 → [0, 1]
   - Example 2: [3, 2, 4], target = 6 → [1, 2]
   - Example 3: [3, 3], target = 6 → [0, 1]
   - Edge case: What if array is empty? (Clarify with interviewer)

3. BREAK DOWN THE PROBLEM
   - For each number, we need to find if (target - number) exists in the array
   - Approaches:
     * Brute force: Check all pairs (O(n²))
     * Hash map: Store numbers we've seen (O(n))

4. SOLVE OR SIMPLIFY
   - Start with brute force solution to ensure correctness
   - Then optimize using a hash map approach

5. IMPLEMENT THE SOLUTION
"""

def two_sum_brute_force(nums, target):
    """
    Brute force approach to find two numbers that add up to target.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None  # No solution found

def two_sum_optimized(nums, target):
    """
    Optimized approach using a hash map.
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Map of value -> index
    num_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # If the complement exists in our map, we found the solution
        if complement in num_map:
            return [num_map[complement], i]
        
        # Store the current number and its index
        num_map[num] = i
    
    return None  # No solution found

"""
6. TEST THE SOLUTION
   - Test with example 1: [2, 7, 11, 15], target = 9
     * Brute force: Check (2,7), found solution [0,1]
     * Optimized: Map={}, i=0, num=2, complement=7, Map={2:0}
                  i=1, num=7, complement=2, found in map, return [0,1]
   
   - Test with example 2: [3, 2, 4], target = 6
   - Test with example 3: [3, 3], target = 6

7. ANALYZE & OPTIMIZE
   - Brute force: O(n²) time, O(1) space
   - Optimized: O(n) time, O(n) space
   - The hash map approach is optimal for this problem

8. REFLECT
   - The hash map approach trades space for time
   - We could sort the array first and use two pointers, but that would be O(n log n)
   - The current solution handles all the requirements efficiently
"""

# Test the implementations
test_cases = [
    ([2, 7, 11, 15], 9),
    ([3, 2, 4], 6),
    ([3, 3], 6)
]

print("Testing brute force solution:")
for nums, target in test_cases:
    print(f"Input: nums = {nums}, target = {target}")
    print(f"Output: {two_sum_brute_force(nums, target)}")

print("\nTesting optimized solution:")
for nums, target in test_cases:
    print(f"Input: nums = {nums}, target = {target}")
    print(f"Output: {two_sum_optimized(nums, target)}")

# ===== BREAKING DOWN COMPLEX PROBLEMS =====
print("\n===== BREAKING DOWN COMPLEX PROBLEMS =====")
"""
Complex problems can be intimidating. Here's how to break them down:

1. IDENTIFY THE CORE PROBLEM
   - What is the fundamental operation being asked?
   - Strip away details to see the underlying pattern

2. RELATE TO KNOWN PATTERNS
   - Is it a variation of a classic problem?
   - Can you apply a well-known algorithm or data structure?
   - Common patterns:
     * Two pointers
     * Sliding window
     * Binary search
     * BFS/DFS
     * Dynamic programming
     * Greedy algorithms
     * Divide and conquer

3. DIVIDE INTO SUB-PROBLEMS
   - Break the problem into smaller, manageable pieces
   - Solve each sub-problem independently
   - Combine the solutions

4. INCREMENTAL APPROACH
   - Start with a simplified version
   - Add complexity step by step
   - Validate each step before moving on

5. VISUALIZE THE PROBLEM
   - Draw diagrams or tables
   - Trace through examples visually
   - Use abstraction to simplify complex relationships

6. USE HELPER FUNCTIONS
   - Encapsulate complex logic in separate functions
   - Focus on one aspect of the problem at a time
   - Improve readability and maintainability
"""

# ===== MANAGING FRUSTRATION =====
print("\n===== MANAGING FRUSTRATION =====")
"""
Frustration is normal when solving difficult problems. Here's how to manage it:

1. RECOGNIZE THE SIGNS
   - Feeling stuck or going in circles
   - Overthinking simple aspects
   - Rushing to implement without a clear plan
   - Emotional responses (anxiety, irritation)

2. TAKE A STEP BACK
   - Pause and breathe
   - Restate the problem from scratch
   - Review what you know and what you're trying to find

3. CHANGE YOUR APPROACH
   - If one method isn't working, try another
   - Switch between top-down and bottom-up thinking
   - Consider completely different algorithms or data structures

4. START FRESH
   - Temporarily set aside your current approach
   - Begin with a new perspective
   - Sometimes the simplest solution is overlooked

5. USE SYSTEMATIC DEBUGGING
   - Add print statements to trace execution
   - Test with minimal examples
   - Isolate the part that's not working

6. TIME MANAGEMENT
   - Set a time limit for each approach
   - If stuck for too long, move on to a different strategy
   - In interviews, communicate your thought process even when stuck

7. PRACTICE REGULARLY
   - Consistent practice reduces frustration over time
   - Expose yourself to various problem types
   - Learn to recognize patterns across problems
"""

# ===== INTERVIEW-SPECIFIC STRATEGIES =====
print("\n===== INTERVIEW-SPECIFIC STRATEGIES =====")
"""
Interviews have unique challenges. Here are strategies specifically for interviews:

1. THINK ALOUD
   - Verbalize your thought process
   - Explain your reasoning for each approach
   - Show how you evaluate trade-offs

2. CLARIFY REQUIREMENTS
   - Ask questions about constraints, edge cases, and expected behavior
   - Confirm your understanding before diving into code
   - Discuss assumptions you're making

3. START WITH A PLAN
   - Outline your approach before coding
   - Discuss high-level strategy with the interviewer
   - Get feedback on your approach before implementation

4. MANAGE TIME WISELY
   - Spend adequate time understanding the problem
   - Don't rush to code without a plan
   - If stuck, communicate and ask for hints

5. HANDLE HINTS GRACEFULLY
   - Listen carefully to interviewer hints
   - Incorporate suggestions into your approach
   - Show that you can adapt and learn

6. TEST YOUR CODE
   - Walk through your solution with examples
   - Identify and fix bugs proactively
   - Discuss potential optimizations

7. ANALYZE COMPLEXITY
   - Explain the time and space complexity
   - Discuss how your solution scales
   - Mention potential optimizations
"""

# ===== PROGRESSION PATH =====
print("\n===== PROGRESSION PATH =====")
"""
How to progress from basic to advanced problems:

1. BASIC PROBLEMS (Start here)
   - Focus on fundamental data structures: arrays, strings, linked lists
   - Master basic algorithms: sorting, searching, traversal
   - Build confidence with simple problems
   - Example topics:
     * Array manipulation
     * String operations
     * Basic math problems
     * Simple recursion

2. INTERMEDIATE PROBLEMS
   - Combine multiple data structures
   - Apply standard algorithms to complex scenarios
   - Optimize brute force solutions
   - Example topics:
     * Binary trees and BSTs
     * Graphs and graph traversal
     * Backtracking
     * Two pointers and sliding window
     * Hash tables for optimization

3. ADVANCED PROBLEMS
   - Implement complex algorithms from scratch
   - Solve problems requiring multiple algorithmic techniques
   - Optimize for extreme constraints
   - Example topics:
     * Dynamic programming
     * Advanced graph algorithms
     * Complex recursion and memoization
     * System design considerations
     * Specialized data structures (segment trees, Fenwick trees)

4. PRACTICE STRATEGY
   - Start with 20-30 basic problems
   - Move to 30-40 intermediate problems
   - Tackle 20-30 advanced problems
   - Review and revisit problems you struggled with
   - Focus on understanding patterns, not memorizing solutions
"""

# ===== RECOMMENDED RESOURCES =====
print("\n===== RECOMMENDED RESOURCES =====")
"""
Resources to improve your problem-solving skills:

1. BOOKS
   - "Cracking the Coding Interview" by Gayle Laakmann McDowell
   - "Elements of Programming Interviews" by Adnan Aziz, Tsung-Hsien Lee, and Amit Prakash
   - "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein

2. ONLINE PLATFORMS
   - LeetCode: Comprehensive problem set with difficulty levels
   - HackerRank: Good for beginners with structured tracks
   - CodeSignal: Interactive practice with real interview questions
   - AlgoExpert: Curated list of problems with video explanations

3. COURSES
   - Princeton's Algorithms course on Coursera
   - MIT's Introduction to Algorithms on OCW
   - Stanford's Algorithm Specialization on Coursera

4. YOUTUBE CHANNELS
   - Back To Back SWE
   - Tech Dose
   - Tushar Roy - Coding Made Simple
   - Kevin Naughton Jr.
   - Nick White

5. PRACTICE GROUPS
   - Join or form a study group
   - Participate in mock interviews
   - Engage in coding competitions
"""

print("\n===== END OF PROBLEM SOLVING APPROACH =====")