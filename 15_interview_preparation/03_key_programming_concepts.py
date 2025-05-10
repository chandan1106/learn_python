"""
Interview Preparation: Key Programming Concepts and Logic Building
"""

# ===== UNDERSTANDING KEY PROGRAMMING CONCEPTS =====
print("\n===== UNDERSTANDING KEY PROGRAMMING CONCEPTS =====")
"""
Mastering key programming concepts is essential for solving interview problems effectively.
These concepts form the foundation of algorithmic thinking and problem-solving.
"""

# ===== TIME AND SPACE COMPLEXITY =====
print("\n===== TIME AND SPACE COMPLEXITY =====")
"""
Understanding computational complexity helps you evaluate and optimize algorithms.

Time Complexity:
- Measures how runtime grows as input size increases
- Usually expressed in Big O notation
- Common complexities (from fastest to slowest):
  * O(1): Constant time - operations don't depend on input size
  * O(log n): Logarithmic time - divide and conquer algorithms
  * O(n): Linear time - processing each element once
  * O(n log n): Linearithmic time - efficient sorting algorithms
  * O(n²): Quadratic time - nested iterations over data
  * O(2^n): Exponential time - recursive algorithms without memoization
  * O(n!): Factorial time - generating all permutations

Space Complexity:
- Measures additional memory used as input size increases
- Includes auxiliary space (temporary space used by algorithm)
- Does not include space used by inputs
- Also expressed in Big O notation
"""

# Example: Different time complexities
def constant_time(arr):
    """O(1) time complexity - constant time"""
    return arr[0] if arr else None

def linear_time(arr):
    """O(n) time complexity - linear time"""
    total = 0
    for num in arr:  # One pass through array
        total += num
    return total

def quadratic_time(arr):
    """O(n²) time complexity - quadratic time"""
    result = []
    for i in arr:  # Nested loops -> n²
        for j in arr:
            result.append(i * j)
    return result

def logarithmic_time(arr, target):
    """O(log n) time complexity - logarithmic time"""
    # Binary search (requires sorted array)
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# ===== RECURSION =====
print("\n===== RECURSION =====")
"""
Recursion is when a function calls itself to solve smaller instances of the same problem.

Key components of recursion:
1. Base case(s): Condition(s) that stop the recursion
2. Recursive case: Where the function calls itself
3. Progress toward base case: Each recursive call should move closer to the base case

Common recursive patterns:
1. Linear recursion: One recursive call per function call
2. Binary recursion: Two recursive calls per function call
3. Multiple recursion: Multiple recursive calls per function call
4. Mutual recursion: Multiple functions calling each other recursively
"""

# Example: Factorial using recursion
def factorial(n):
    """Calculate factorial using recursion"""
    # Base case
    if n == 0 or n == 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)

# Example: Fibonacci using recursion
def fibonacci_recursive(n):
    """Calculate Fibonacci number using recursion"""
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Recursive case
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Example: Optimized Fibonacci using memoization
def fibonacci_memoized(n, memo={}):
    """Calculate Fibonacci number using recursion with memoization"""
    # Check if we've already calculated this value
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Recursive case with memoization
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

# ===== ITERATION VS RECURSION =====
print("\n===== ITERATION VS RECURSION =====")
"""
Many problems can be solved using either iteration or recursion.

Iteration:
- Uses loops (for, while)
- Generally more efficient in terms of memory
- Often easier to understand for simple problems
- No risk of stack overflow

Recursion:
- Function calls itself
- Can be more elegant for certain problems
- Natural fit for tree/graph traversal, divide and conquer
- Risk of stack overflow for deep recursion
- May require optimization (tail recursion, memoization)
"""

# Example: Factorial with iteration vs recursion
def factorial_iterative(n):
    """Calculate factorial using iteration"""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(f"Factorial of 5 (recursive): {factorial(5)}")
print(f"Factorial of 5 (iterative): {factorial_iterative(5)}")

# Example: Fibonacci with iteration vs recursion
def fibonacci_iterative(n):
    """Calculate Fibonacci number using iteration"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print(f"Fibonacci of 10 (recursive): {fibonacci_recursive(10)}")
print(f"Fibonacci of 10 (memoized): {fibonacci_memoized(10)}")
print(f"Fibonacci of 10 (iterative): {fibonacci_iterative(10)}")

# ===== DIVIDE AND CONQUER =====
print("\n===== DIVIDE AND CONQUER =====")
"""
Divide and Conquer is a problem-solving paradigm that breaks a problem into smaller subproblems,
solves them independently, and then combines their solutions.

Steps:
1. Divide: Break the problem into smaller subproblems
2. Conquer: Solve the subproblems recursively
3. Combine: Merge the solutions to subproblems into a solution for the original problem

Common applications:
- Merge Sort
- Quick Sort
- Binary Search
- Strassen's Matrix Multiplication
- Closest Pair of Points
"""

# Example: Merge Sort using divide and conquer
def merge_sort(arr):
    """Sort an array using the merge sort algorithm"""
    # Base case: array of length 0 or 1 is already sorted
    if len(arr) <= 1:
        return arr
    
    # Divide: Split the array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Combine: Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays into a single sorted array"""
    result = []
    i = j = 0
    
    # Compare elements from both arrays and take the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Test merge sort
arr = [38, 27, 43, 3, 9, 82, 10]
print(f"Original array: {arr}")
print(f"Sorted array: {merge_sort(arr)}")

# ===== DYNAMIC PROGRAMMING =====
print("\n===== DYNAMIC PROGRAMMING =====")
"""
Dynamic Programming (DP) is a technique for solving complex problems by breaking them down
into simpler subproblems and storing the results to avoid redundant calculations.

Key characteristics:
1. Overlapping subproblems: Same subproblems are solved multiple times
2. Optimal substructure: Optimal solution can be constructed from optimal solutions of subproblems

Approaches:
1. Top-down (Memoization): Recursive approach with caching
2. Bottom-up (Tabulation): Iterative approach building from smallest subproblems

Common applications:
- Fibonacci sequence
- Longest Common Subsequence
- Knapsack problem
- Shortest path algorithms
- Matrix chain multiplication
"""

# Example: Fibonacci with dynamic programming (top-down)
def fibonacci_dp_top_down(n, memo={}):
    """Calculate Fibonacci number using dynamic programming (top-down)"""
    if n in memo:
        return memo[n]
    
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    memo[n] = fibonacci_dp_top_down(n - 1, memo) + fibonacci_dp_top_down(n - 2, memo)
    return memo[n]

# Example: Fibonacci with dynamic programming (bottom-up)
def fibonacci_dp_bottom_up(n):
    """Calculate Fibonacci number using dynamic programming (bottom-up)"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Create table to store results
    dp = [0] * (n + 1)
    dp[1] = 1
    
    # Fill the table bottom-up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

print(f"Fibonacci of 10 (DP top-down): {fibonacci_dp_top_down(10)}")
print(f"Fibonacci of 10 (DP bottom-up): {fibonacci_dp_bottom_up(10)}")

# Example: Longest Common Subsequence using DP
def longest_common_subsequence(text1, text2):
    """Find the length of the longest common subsequence between two strings"""
    m, n = len(text1), len(text2)
    
    # Create a table to store results of subproblems
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill the table in bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Test longest common subsequence
text1 = "abcde"
text2 = "ace"
print(f"Longest common subsequence of '{text1}' and '{text2}': {longest_common_subsequence(text1, text2)}")

# ===== GREEDY ALGORITHMS =====
print("\n===== GREEDY ALGORITHMS =====")
"""
Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum.

Key characteristics:
1. Greedy choice property: A global optimum can be reached by making locally optimal choices
2. Optimal substructure: An optimal solution contains optimal solutions to subproblems

Advantages:
- Simple and intuitive
- Often efficient

Limitations:
- May not always find the global optimum
- Requires proof of correctness

Common applications:
- Activity selection
- Huffman coding
- Minimum spanning trees (Prim's, Kruskal's)
- Dijkstra's shortest path algorithm
- Coin change (with certain coin systems)
"""

# Example: Activity selection problem
def activity_selection(start, finish):
    """
    Select maximum number of activities that don't overlap
    
    Args:
        start: List of start times
        finish: List of finish times
    
    Returns:
        List of selected activity indices
    """
    n = len(start)
    
    # Sort activities by finish time
    activities = sorted(zip(range(n), start, finish), key=lambda x: x[2])
    
    # Select the first activity
    selected = [activities[0][0]]
    last_finish = activities[0][2]
    
    # Consider the rest of the activities
    for i in range(1, n):
        # If this activity starts after the last selected activity finishes
        if activities[i][1] >= last_finish:
            selected.append(activities[i][0])
            last_finish = activities[i][2]
    
    return selected

# Test activity selection
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
print(f"Selected activities: {activity_selection(start, finish)}")

# ===== BACKTRACKING =====
print("\n===== BACKTRACKING =====")
"""
Backtracking is an algorithmic technique that builds candidates for solutions incrementally
and abandons a candidate ("backtracks") as soon as it determines the candidate cannot lead to a valid solution.

Key characteristics:
1. Incremental construction of candidates
2. Feasibility function to test if a candidate can lead to a solution
3. Recursive exploration of search space
4. Pruning of search space when candidates are determined to be invalid

Common applications:
- N-Queens problem
- Sudoku solver
- Hamiltonian path
- Subset sum
- Graph coloring
- Permutations and combinations
"""

# Example: N-Queens problem
def solve_n_queens(n):
    """
    Solve the N-Queens problem: place N queens on an NxN chessboard
    so that no two queens attack each other.
    
    Returns a list of solutions, where each solution is a list of column
    positions for queens in each row.
    """
    solutions = []
    
    def backtrack(row, current_solution):
        if row == n:
            # Found a valid solution
            solutions.append(current_solution[:])
            return
        
        for col in range(n):
            # Check if placing a queen at (row, col) is valid
            if is_valid(current_solution, row, col):
                current_solution.append(col)
                backtrack(row + 1, current_solution)
                current_solution.pop()  # Backtrack
    
    def is_valid(solution, row, col):
        # Check if a queen can be placed at (row, col)
        for prev_row, prev_col in enumerate(solution):
            # Check if queens attack each other
            if prev_col == col:  # Same column
                return False
            if prev_row + prev_col == row + col:  # Same diagonal (/)
                return False
            if prev_row - prev_col == row - col:  # Same diagonal (\)
                return False
        return True
    
    backtrack(0, [])
    return solutions

# Test N-Queens
n = 4
solutions = solve_n_queens(n)
print(f"Number of solutions for {n}-Queens: {len(solutions)}")
print(f"First solution: {solutions[0]}")

# ===== GRAPH ALGORITHMS =====
print("\n===== GRAPH ALGORITHMS =====")
"""
Graph algorithms operate on graph data structures consisting of vertices and edges.

Common graph representations:
1. Adjacency Matrix: 2D array where matrix[i][j] indicates an edge from i to j
2. Adjacency List: Array of lists where list[i] contains vertices adjacent to i

Basic graph algorithms:
1. Breadth-First Search (BFS): Explores all neighbors before moving to next level
2. Depth-First Search (DFS): Explores as far as possible along a branch before backtracking
3. Dijkstra's Algorithm: Finds shortest paths from a source vertex
4. Bellman-Ford Algorithm: Finds shortest paths and detects negative cycles
5. Floyd-Warshall Algorithm: Finds all-pairs shortest paths
6. Kruskal's Algorithm: Finds minimum spanning tree
7. Prim's Algorithm: Finds minimum spanning tree
8. Topological Sort: Linear ordering of vertices in a directed acyclic graph
"""

# Example: Graph representation and traversal
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        """Add an edge from u to v"""
        self.adj_list[u].append(v)
    
    def bfs(self, start):
        """Breadth-First Search traversal"""
        visited = [False] * self.vertices
        result = []
        
        # Create a queue for BFS
        queue = [start]
        visited[start] = True
        
        while queue:
            # Dequeue a vertex from queue
            vertex = queue.pop(0)
            result.append(vertex)
            
            # Get all adjacent vertices
            for neighbor in self.adj_list[vertex]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        
        return result
    
    def dfs(self, start):
        """Depth-First Search traversal"""
        visited = [False] * self.vertices
        result = []
        
        def dfs_util(vertex):
            # Mark the current node as visited
            visited[vertex] = True
            result.append(vertex)
            
            # Recur for all adjacent vertices
            for neighbor in self.adj_list[vertex]:
                if not visited[neighbor]:
                    dfs_util(neighbor)
        
        dfs_util(start)
        return result

# Create a graph
g = Graph(6)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
g.add_edge(2, 4)
g.add_edge(3, 4)
g.add_edge(3, 5)
g.add_edge(4, 5)

print(f"BFS traversal starting from vertex 0: {g.bfs(0)}")
print(f"DFS traversal starting from vertex 0: {g.dfs(0)}")

# ===== BREAKING DOWN COMPLEX PROBLEMS =====
print("\n===== BREAKING DOWN COMPLEX PROBLEMS =====")
"""
Strategies for breaking down complex problems:

1. Understand the Problem
   - Identify inputs, outputs, and constraints
   - Clarify ambiguities
   - Work through examples

2. Simplify the Problem
   - Solve a simpler version first
   - Identify special cases
   - Remove constraints temporarily

3. Pattern Recognition
   - Look for similarities to known problems
   - Identify applicable algorithms or data structures
   - Recognize common patterns (two pointers, sliding window, etc.)

4. Divide and Conquer
   - Break the problem into smaller subproblems
   - Solve each subproblem independently
   - Combine the solutions

5. Build Incrementally
   - Start with a basic solution
   - Add complexity step by step
   - Test each step

6. Use Visualization
   - Draw diagrams
   - Trace through examples
   - Use tables or trees to organize information

7. Work Backwards
   - Start from the desired output
   - Determine what steps would lead to that output
   - Reverse-engineer the solution
"""

# ===== LOGIC BUILDING EXERCISES =====
print("\n===== LOGIC BUILDING EXERCISES =====")
"""
To improve your logic building skills, practice these types of problems:

1. Array Manipulation
   - Finding elements (min, max, kth largest)
   - Searching and sorting
   - Subarrays with specific properties
   - Two pointers and sliding window problems

2. String Processing
   - Substring problems
   - Anagrams and palindromes
   - String matching and pattern recognition
   - String manipulation (reverse, replace, etc.)

3. Linked Lists
   - Traversal and modification
   - Detecting cycles
   - Merging and partitioning
   - Reversing sections

4. Trees and Graphs
   - Traversal algorithms (BFS, DFS)
   - Path finding
   - Tree construction and modification
   - Graph algorithms (shortest path, MST)

5. Dynamic Programming
   - Optimization problems
   - Counting problems
   - Sequence problems
   - Grid-based problems

6. Recursion and Backtracking
   - Permutations and combinations
   - Constraint satisfaction problems
   - Divide and conquer algorithms
   - Recursive data structure manipulation

7. Bit Manipulation
   - Bit operations
   - Bit counting and manipulation
   - Using bits to optimize solutions
"""

# Example: Logic building with a medium-difficulty problem
def longest_palindromic_substring(s):
    """
    Find the longest palindromic substring in a string
    
    Args:
        s: Input string
    
    Returns:
        Longest palindromic substring
    """
    if not s:
        return ""
    
    start = 0  # Start index of longest palindrome
    max_len = 1  # Length of longest palindrome
    
    # Helper function to expand around center
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1  # Length of palindrome
    
    for i in range(len(s)):
        # Expand around center i (odd length palindromes)
        len1 = expand_around_center(i, i)
        
        # Expand around center i, i+1 (even length palindromes)
        len2 = expand_around_center(i, i + 1)
        
        # Update longest palindrome if needed
        curr_len = max(len1, len2)
        if curr_len > max_len:
            max_len = curr_len
            # Calculate start index based on length
            start = i - (curr_len - 1) // 2
    
    return s[start:start + max_len]

# Test longest palindromic substring
test_string = "babad"
print(f"Longest palindromic substring of '{test_string}': {longest_palindromic_substring(test_string)}")

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Mastering key programming concepts and developing strong logic building skills are essential for solving interview problems effectively.

Remember these key points:
1. Understand the fundamental concepts (time/space complexity, recursion, etc.)
2. Learn to recognize common patterns and algorithms
3. Practice breaking down complex problems into manageable parts
4. Build solutions incrementally, starting with simple approaches
5. Analyze and optimize your solutions
6. Practice regularly with a variety of problem types

With consistent practice and a solid understanding of these concepts, you'll be well-prepared to tackle even the most challenging interview problems.
""")

print("\n===== END OF KEY PROGRAMMING CONCEPTS =====")