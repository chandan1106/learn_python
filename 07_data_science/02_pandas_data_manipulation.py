"""
Data Science Fundamentals: Pandas for Data Manipulation
"""

# Import pandas
# In a real environment, you would need to install pandas first: pip install pandas
print("Note: This code assumes pandas is installed. If you get an ImportError, install it with: pip install pandas")

try:
    import pandas as pd
    import numpy as np
    print("Pandas successfully imported! Version:", pd.__version__)
except ImportError:
    print("Pandas is not installed. Please install it with: pip install pandas")
    # Exit gracefully if pandas is not installed
    import sys
    sys.exit(1)

# ===== CREATING PANDAS OBJECTS =====
print("\n===== CREATING PANDAS OBJECTS =====")

# Creating Series (1D array with labels)
print("Creating Series:")
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Creating Series with custom index
s_custom = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print("\nSeries with custom index:")
print(s_custom)

# Creating DataFrame (2D table with labeled rows and columns)
print("\nCreating DataFrame from dictionary:")
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)

# Creating DataFrame from NumPy array
print("\nCreating DataFrame from NumPy array:")
dates = pd.date_range('20230101', periods=6)
df_dates = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df_dates)

# Creating DataFrame from CSV file (commented out as we don't have a file)
# df_csv = pd.read_csv('data.csv')
# print(df_csv)

# ===== VIEWING DATA =====
print("\n===== VIEWING DATA =====")

# Basic DataFrame information
print("Basic DataFrame information:")
print(f"Shape: {df.shape}")
print(f"Dimensions: {df.ndim}")
print(f"Size: {df.size}")
print(f"Data types:\n{df.dtypes}")

# Viewing data
print("\nHead (first 2 rows):")
print(df.head(2))

print("\nTail (last 2 rows):")
print(df.tail(2))

print("\nSample (random rows):")
print(df.sample(2))

print("\nDataFrame description (statistics):")
# Create a DataFrame with numeric data for demonstration
df_numeric = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})
print(df_numeric.describe())

# ===== SELECTING DATA =====
print("\n===== SELECTING DATA =====")

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston'],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'Department': ['HR', 'IT', 'Finance', 'Marketing', 'IT']
}
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)

# Selecting columns
print("\nSelecting a single column (returns Series):")
print(df['Name'])

print("\nSelecting multiple columns (returns DataFrame):")
print(df[['Name', 'Age']])

# Selecting rows by position
print("\nSelecting rows by position:")
print("First 2 rows:")
print(df[:2])

# Selecting rows by label (index)
print("\nSelecting rows by label (after setting index):")
df_indexed = df.set_index('Name')
print(df_indexed)
print("\nSelecting row with label 'Alice':")
print(df_indexed.loc['Alice'])

# Selecting rows by condition
print("\nSelecting rows by condition:")
print("People older than 35:")
print(df[df['Age'] > 35])

print("\nPeople in IT department:")
print(df[df['Department'] == 'IT'])

print("\nComplex condition (Age > 30 AND Salary >= 70000):")
print(df[(df['Age'] > 30) & (df['Salary'] >= 70000)])

# Selecting specific cells
print("\nSelecting specific cells:")
print("Using loc (label-based):")
df_indexed = df.set_index('Name')
print(df_indexed.loc['Alice', 'Age'])

print("\nUsing iloc (position-based):")
print(df.iloc[0, 1])  # Row 0, Column 1

print("\nSelecting a subset of rows and columns:")
print("Using loc:")
print(df_indexed.loc[['Alice', 'Bob'], ['Age', 'City']])

print("\nUsing iloc:")
print(df.iloc[0:2, 1:3])  # First 2 rows, columns 1 and 2

# ===== DATA MANIPULATION =====
print("\n===== DATA MANIPULATION =====")

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston'],
    'Department': ['HR', 'IT', 'Finance', 'Marketing', 'IT']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Adding a new column
print("\nAdding a new column:")
df['Salary'] = [50000, 60000, 70000, 80000, 90000]
print(df)

# Adding a column with calculation
print("\nAdding a column with calculation:")
df['Salary_Monthly'] = df['Salary'] / 12
print(df)

# Modifying values
print("\nModifying values:")
df.loc[0, 'Age'] = 26  # Change Alice's age
print(df)

# Applying functions to columns
print("\nApplying functions to columns:")
df['Age_in_Months'] = df['Age'].apply(lambda x: x * 12)
print(df)

# Applying functions to rows
print("\nApplying functions to rows:")
df['Name_Length'] = df.apply(lambda row: len(row['Name']), axis=1)
print(df)

# Renaming columns
print("\nRenaming columns:")
df = df.rename(columns={'Name': 'Full_Name', 'Age': 'Age_Years'})
print(df)

# Dropping columns
print("\nDropping columns:")
df = df.drop(['Name_Length', 'Age_in_Months'], axis=1)
print(df)

# Dropping rows
print("\nDropping rows:")
df = df.drop([1, 3], axis=0)  # Drop Bob and David
print(df)

# ===== HANDLING MISSING DATA =====
print("\n===== HANDLING MISSING DATA =====")

# Create a DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
}
df_missing = pd.DataFrame(data)
print("DataFrame with missing values:")
print(df_missing)

# Checking for missing values
print("\nChecking for missing values:")
print(f"Any missing values? {df_missing.isna().any().any()}")
print(f"Missing values by column:\n{df_missing.isna().sum()}")

# Dropping rows with missing values
print("\nDropping rows with missing values:")
print(df_missing.dropna())

# Dropping columns with missing values
print("\nDropping columns with missing values:")
print(df_missing.dropna(axis=1))

# Filling missing values
print("\nFilling missing values with 0:")
print(df_missing.fillna(0))

print("\nFilling missing values with column mean:")
print(df_missing.fillna(df_missing.mean()))

print("\nFilling missing values with forward fill (previous value):")
print(df_missing.fillna(method='ffill'))

print("\nFilling missing values with backward fill (next value):")
print(df_missing.fillna(method='bfill'))

# ===== DATA AGGREGATION =====
print("\n===== DATA AGGREGATION =====")

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Value1': [10, 20, 30, 40, 50, 60, 70, 80, 90],
    'Value2': [100, 200, 300, 400, 500, 600, 700, 800, 900]
}
df_agg = pd.DataFrame(data)
print("Sample DataFrame for aggregation:")
print(df_agg)

# Basic aggregation
print("\nBasic aggregation:")
print(f"Sum: {df_agg['Value1'].sum()}")
print(f"Mean: {df_agg['Value1'].mean()}")
print(f"Median: {df_agg['Value1'].median()}")
print(f"Min: {df_agg['Value1'].min()}")
print(f"Max: {df_agg['Value1'].max()}")

# Groupby aggregation
print("\nGroupby aggregation:")
grouped = df_agg.groupby('Category')
print(grouped['Value1'].sum())

print("\nMultiple aggregations:")
print(grouped['Value1'].agg(['sum', 'mean', 'min', 'max']))

print("\nDifferent aggregations for different columns:")
print(grouped.agg({'Value1': 'sum', 'Value2': 'mean'}))

print("\nMultiple aggregations for multiple columns:")
print(grouped.agg({
    'Value1': ['sum', 'mean'],
    'Value2': ['min', 'max']
}))

# ===== MERGING AND JOINING DATA =====
print("\n===== MERGING AND JOINING DATA =====")

# Create sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'IT', 'Finance', 'Marketing']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3, 5],
    'Salary': [50000, 60000, 70000, 90000],
    'Bonus': [5000, 6000, 7000, 9000]
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Inner join (only matching rows)
print("\nInner join (only matching rows):")
inner_join = pd.merge(df1, df2, on='ID', how='inner')
print(inner_join)

# Left join (all rows from left, matching from right)
print("\nLeft join (all rows from left, matching from right):")
left_join = pd.merge(df1, df2, on='ID', how='left')
print(left_join)

# Right join (all rows from right, matching from left)
print("\nRight join (all rows from right, matching from left):")
right_join = pd.merge(df1, df2, on='ID', how='right')
print(right_join)

# Outer join (all rows from both)
print("\nOuter join (all rows from both):")
outer_join = pd.merge(df1, df2, on='ID', how='outer')
print(outer_join)

# Concatenation
print("\nConcatenation (stacking vertically):")
df3 = pd.DataFrame({
    'ID': [6, 7],
    'Name': ['Eva', 'Frank'],
    'Department': ['IT', 'HR']
})
concat_vertical = pd.concat([df1, df3])
print(concat_vertical)

print("\nConcatenation (stacking horizontally):")
df4 = pd.DataFrame({
    'Location': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'Remote': [False, True, True, False]
}, index=[1, 2, 3, 4])
concat_horizontal = pd.concat([df1, df4], axis=1)
print(concat_horizontal)

# ===== DATA TRANSFORMATION =====
print("\n===== DATA TRANSFORMATION =====")

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'Department': ['HR', 'IT', 'Finance', 'Marketing', 'IT']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Sorting
print("\nSorting by Age (ascending):")
print(df.sort_values('Age'))

print("\nSorting by Age (descending):")
print(df.sort_values('Age', ascending=False))

print("\nSorting by multiple columns:")
print(df.sort_values(['Department', 'Salary'], ascending=[True, False]))

# Ranking
print("\nRanking by Salary:")
print(df['Salary'].rank())

# One-hot encoding
print("\nOne-hot encoding for Department:")
dummies = pd.get_dummies(df['Department'], prefix='Dept')
df_encoded = pd.concat([df, dummies], axis=1)
print(df_encoded)

# Binning
print("\nBinning Ages into categories:")
bins = [20, 30, 40, 50]
labels = ['Young', 'Middle', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
print(df)

# Applying custom transformations
print("\nApplying custom transformations:")
df['Salary_Category'] = df['Salary'].apply(
    lambda x: 'High' if x >= 70000 else 'Medium' if x >= 60000 else 'Low'
)
print(df)

# ===== TIME SERIES DATA =====
print("\n===== TIME SERIES DATA =====")

# Create a time series
print("Creating a time series:")
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
ts = pd.Series(np.random.randn(10), index=dates)
print(ts)

# Time series operations
print("\nTime series operations:")
print(f"Shifting forward 2 periods:\n{ts.shift(2)}")
print(f"Shifting backward 1 period:\n{ts.shift(-1)}")
print(f"Rolling mean (window=3):\n{ts.rolling(window=3).mean()}")
print(f"Expanding mean:\n{ts.expanding().mean()}")

# Resampling
print("\nResampling:")
print(f"Daily to 3-day sum:\n{ts.resample('3D').sum()}")

# Date functionality
print("\nDate functionality:")
print(f"Month: {dates[0].month}")
print(f"Year: {dates[0].year}")
print(f"Day of week: {dates[0].day_of_week}")
print(f"Is month end? {dates[0].is_month_end}")

# ===== PRACTICAL EXAMPLES =====
print("\n===== PRACTICAL EXAMPLES =====")

# Example 1: Sales Analysis
print("Example 1: Sales Analysis")
sales_data = {
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Units': [10, 15, 12, 8, 20, 25, 15, 10, 18, 12],
    'Price': [100, 200, 100, 150, 200, 100, 150, 200, 100, 150]
}
sales_df = pd.DataFrame(sales_data)
sales_df['Revenue'] = sales_df['Units'] * sales_df['Price']
print(sales_df)

print("\nTotal revenue by product:")
print(sales_df.groupby('Product')['Revenue'].sum())

print("\nDaily revenue:")
print(sales_df.groupby('Date')['Revenue'].sum())

# Example 2: Customer Analysis
print("\nExample 2: Customer Analysis")
customer_data = {
    'CustomerID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Purchase': [100, 200, 150, 300, 250, 120, 220, 180, 320, 270]
}
customer_df = pd.DataFrame(customer_data)
print(customer_df)

print("\nTotal purchases by customer:")
print(customer_df.groupby('CustomerID')['Purchase'].sum())

print("\nAverage purchase by customer:")
print(customer_df.groupby('CustomerID')['Purchase'].mean())

print("\nCustomer purchase frequency:")
print(customer_df.groupby('CustomerID').size())

# Example 3: Data Cleaning and Preparation
print("\nExample 3: Data Cleaning and Preparation")
messy_data = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', np.nan, 'David', 'Eva'],
    'Age': [25, np.nan, 35, 40, 45],
    'Salary': [50000, 60000, 70000, np.nan, 90000],
    'Department': ['HR', 'IT', 'Finance', 'Marketing', 'IT']
}
messy_df = pd.DataFrame(messy_data)
print("Messy DataFrame:")
print(messy_df)

# Clean the data
print("\nCleaning the data:")
# Fill missing names with 'Unknown'
messy_df['Name'] = messy_df['Name'].fillna('Unknown')
# Fill missing ages with the mean age
messy_df['Age'] = messy_df['Age'].fillna(messy_df['Age'].mean())
# Fill missing salaries with the median salary by department
messy_df['Salary'] = messy_df.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.median())
)
print(messy_df)

print("\n===== END OF PANDAS TUTORIAL =====")