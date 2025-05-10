"""
Data Science Fundamentals: NumPy Basics
"""

# Import NumPy
# In a real environment, you would need to install NumPy first: pip install numpy
print("Note: This code assumes NumPy is installed. If you get an ImportError, install it with: pip install numpy")

try:
    import numpy as np
    print("NumPy successfully imported! Version:", np.__version__)
except ImportError:
    print("NumPy is not installed. Please install it with: pip install numpy")
    # Exit gracefully if NumPy is not installed
    import sys
    sys.exit(1)

# ===== CREATING NUMPY ARRAYS =====
print("\n===== CREATING NUMPY ARRAYS =====")

# Creating arrays from Python lists
print("Creating arrays from Python lists:")
list1d = [1, 2, 3, 4, 5]
array1d = np.array(list1d)
print(f"1D array: {array1d}")

list2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array2d = np.array(list2d)
print(f"2D array:\n{array2d}")

# Creating arrays with special values
print("\nCreating arrays with special values:")
zeros = np.zeros((3, 4))  # 3x4 array of zeros
print(f"Zeros array:\n{zeros}")

ones = np.ones((2, 3))  # 2x3 array of ones
print(f"Ones array:\n{ones}")

identity = np.eye(3)  # 3x3 identity matrix
print(f"Identity matrix:\n{identity}")

# Creating arrays with sequences
print("\nCreating arrays with sequences:")
range_array = np.arange(0, 10, 2)  # Start, stop, step
print(f"Range array: {range_array}")

linspace = np.linspace(0, 1, 5)  # Start, stop, num_points
print(f"Linspace array: {linspace}")

# Creating arrays with random values
print("\nCreating arrays with random values:")
np.random.seed(42)  # For reproducibility

random_array = np.random.random((2, 2))  # Uniform distribution [0, 1)
print(f"Random array [0, 1):\n{random_array}")

normal_array = np.random.normal(0, 1, (2, 2))  # Normal distribution (mean=0, std=1)
print(f"Normal distribution array:\n{normal_array}")

randint_array = np.random.randint(1, 10, (2, 3))  # Random integers [1, 10)
print(f"Random integers array:\n{randint_array}")

# ===== ARRAY ATTRIBUTES =====
print("\n===== ARRAY ATTRIBUTES =====")

sample_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Sample array:\n{sample_array}")
print(f"Shape: {sample_array.shape}")  # Dimensions
print(f"Size: {sample_array.size}")    # Total number of elements
print(f"Dimensions: {sample_array.ndim}")  # Number of dimensions
print(f"Data type: {sample_array.dtype}")  # Data type of elements
print(f"Item size: {sample_array.itemsize} bytes")  # Size of each element
print(f"Total memory: {sample_array.nbytes} bytes")  # Total memory used

# ===== ARRAY INDEXING AND SLICING =====
print("\n===== ARRAY INDEXING AND SLICING =====")

# 1D array indexing
array1d = np.array([10, 20, 30, 40, 50])
print(f"1D array: {array1d}")
print(f"First element: {array1d[0]}")
print(f"Last element: {array1d[-1]}")

# 1D array slicing
print(f"First three elements: {array1d[:3]}")
print(f"Every other element: {array1d[::2]}")
print(f"Reversed array: {array1d[::-1]}")

# 2D array indexing
array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{array2d}")
print(f"Element at row 1, column 2: {array2d[1, 2]}")  # Same as array2d[1][2]
print(f"Last row: {array2d[-1]}")

# 2D array slicing
print(f"First two rows:\n{array2d[:2]}")
print(f"First column: {array2d[:, 0]}")
print(f"Submatrix (first 2 rows, last 2 columns):\n{array2d[:2, 1:]}")

# Boolean indexing
print("\nBoolean indexing:")
bool_idx = array2d > 5
print(f"Boolean mask (elements > 5):\n{bool_idx}")
print(f"Elements greater than 5: {array2d[bool_idx]}")
print(f"Elements greater than 5 (direct): {array2d[array2d > 5]}")

# Fancy indexing
print("\nFancy indexing:")
indices = np.array([0, 2])  # Select rows 0 and 2
print(f"Rows 0 and 2:\n{array2d[indices]}")

col_indices = np.array([0, 2])  # Select columns 0 and 2
print(f"Columns 0 and 2:\n{array2d[:, col_indices]}")

# ===== ARRAY OPERATIONS =====
print("\n===== ARRAY OPERATIONS =====")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"a: {a}")
print(f"b: {b}")

# Element-wise operations
print("\nElement-wise operations:")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")  # Element-wise power

# Broadcasting
print("\nBroadcasting:")
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([10, 20, 30])
print(f"c:\n{c}")
print(f"d: {d}")
print(f"c + d:\n{c + d}")  # d is broadcast to match c's shape

# Matrix operations
print("\nMatrix operations:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A:\n{A}")
print(f"B:\n{B}")

print(f"Matrix multiplication (A @ B):\n{A @ B}")  # Same as np.matmul(A, B)
print(f"Element-wise multiplication (A * B):\n{A * B}")

# Dot product
print("\nDot product:")
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"Dot product (v1 · v2): {np.dot(v1, v2)}")  # Same as v1 @ v2

# ===== ARRAY MANIPULATION =====
print("\n===== ARRAY MANIPULATION =====")

# Reshaping arrays
print("Reshaping arrays:")
arr = np.arange(12)
print(f"Original array: {arr}")

reshaped = arr.reshape(3, 4)
print(f"Reshaped to 3x4:\n{reshaped}")

reshaped_alt = arr.reshape(4, 3)
print(f"Reshaped to 4x3:\n{reshaped_alt}")

# Flattening and raveling
print("\nFlattening arrays:")
print(f"Flattened array: {reshaped.flatten()}")  # Creates a copy
print(f"Raveled array: {reshaped.ravel()}")  # Returns a view when possible

# Transposing
print("\nTransposing arrays:")
transposed = reshaped.T
print(f"Original:\n{reshaped}")
print(f"Transposed:\n{transposed}")

# Stacking arrays
print("\nStacking arrays:")
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vertical_stack = np.vstack((a, b))
print(f"Vertical stack:\n{vertical_stack}")

horizontal_stack = np.hstack((a, b))
print(f"Horizontal stack:\n{horizontal_stack}")

# Splitting arrays
print("\nSplitting arrays:")
arr = np.arange(16).reshape(4, 4)
print(f"Original array:\n{arr}")

# Split horizontally
h_splits = np.hsplit(arr, 2)
print(f"Horizontal split (first part):\n{h_splits[0]}")
print(f"Horizontal split (second part):\n{h_splits[1]}")

# Split vertically
v_splits = np.vsplit(arr, 2)
print(f"Vertical split (first part):\n{v_splits[0]}")
print(f"Vertical split (second part):\n{v_splits[1]}")

# ===== ARRAY FUNCTIONS =====
print("\n===== ARRAY FUNCTIONS =====")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Sample array:\n{arr}")

# Statistical functions
print("\nStatistical functions:")
print(f"Sum of all elements: {np.sum(arr)}")
print(f"Sum along rows (axis=0): {np.sum(arr, axis=0)}")
print(f"Sum along columns (axis=1): {np.sum(arr, axis=1)}")
print(f"Mean of all elements: {np.mean(arr)}")
print(f"Median of all elements: {np.median(arr)}")
print(f"Standard deviation: {np.std(arr)}")
print(f"Minimum value: {np.min(arr)}")
print(f"Maximum value: {np.max(arr)}")

# Finding indices
print("\nFinding indices:")
print(f"Index of minimum value: {np.argmin(arr)}")
print(f"Index of maximum value: {np.argmax(arr)}")

# Sorting
print("\nSorting:")
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Unsorted array: {unsorted}")
print(f"Sorted array: {np.sort(unsorted)}")

# Sorting 2D array
print(f"Original 2D array:\n{arr}")
print(f"Sorted along rows (axis=1):\n{np.sort(arr, axis=1)}")
print(f"Sorted along columns (axis=0):\n{np.sort(arr, axis=0)}")

# Unique values
print("\nUnique values:")
repeated = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(f"Array with repeated values: {repeated}")
print(f"Unique values: {np.unique(repeated)}")

# Counting occurrences
unique, counts = np.unique(repeated, return_counts=True)
print(f"Value counts: {dict(zip(unique, counts))}")

# ===== BROADCASTING RULES =====
print("\n===== BROADCASTING RULES =====")

# Broadcasting allows NumPy to work with arrays of different shapes
print("Broadcasting example:")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 20, 30])
print(f"a (3x3):\n{a}")
print(f"b (1D):\n{b}")
print(f"a + b (b is broadcast):\n{a + b}")

# Broadcasting with scalar
print("\nBroadcasting with scalar:")
print(f"a * 2:\n{a * 2}")

# More complex broadcasting
print("\nMore complex broadcasting:")
c = np.array([[10], [20], [30]])  # Column vector
print(f"c (column vector):\n{c}")
print(f"a + c (c is broadcast):\n{a + c}")

# ===== PRACTICAL EXAMPLES =====
print("\n===== PRACTICAL EXAMPLES =====")

# Example 1: Computing distances between points
print("Example 1: Computing distances between points")
points = np.random.random((5, 2))  # 5 random points in 2D
print(f"Random points:\n{points}")

# Calculate pairwise distances
distances = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        # Euclidean distance
        distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

print(f"Distance matrix:\n{distances}")

# Example 2: Image processing (simulated)
print("\nExample 2: Image processing (simulated)")
# Create a simple 5x5 "image"
image = np.random.randint(0, 256, (5, 5))
print(f"Original 'image':\n{image}")

# Apply a filter (e.g., increase brightness)
brightened = np.clip(image + 50, 0, 255)  # Add 50 to brightness, clip to valid range
print(f"Brightened 'image':\n{brightened}")

# Example 3: Financial calculations
print("\nExample 3: Financial calculations")
# Monthly returns for 3 stocks over 5 months (in percentage)
returns = np.array([
    [1.2, 0.8, 1.5, -0.5, 1.1],  # Stock A
    [0.9, 1.3, 0.3, 0.7, 0.8],   # Stock B
    [1.8, -0.5, 0.4, 1.2, 1.0]   # Stock C
])
print(f"Monthly returns (%):\n{returns}")

# Calculate average monthly return for each stock
avg_returns = np.mean(returns, axis=1)
print(f"Average monthly returns: {avg_returns}")

# Calculate cumulative returns
cumulative_returns = np.cumprod(1 + returns/100, axis=1) - 1
print(f"Cumulative returns:\n{cumulative_returns}")

# Calculate correlation matrix
correlation = np.corrcoef(returns)
print(f"Correlation matrix:\n{correlation}")

print("\n===== END OF NUMPY BASICS =====")