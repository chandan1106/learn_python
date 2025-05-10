"""
Python Control Flow

This module covers conditional statements and loops in Python.
"""

# Conditional Statements
# if, elif, else

print("CONDITIONAL STATEMENTS")
print("-" * 30)

# Simple if statement
age = 20
if age >= 18:
    print("You are an adult.")

# if-else statement
temperature = 15
if temperature > 25:
    print("It's warm outside.")
else:
    print("It's cool outside.")

# if-elif-else statement
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Score: {score}, Grade: {grade}")

# Nested if statements
num = 15
if num > 0:
    if num % 2 == 0:
        print(f"{num} is a positive even number.")
    else:
        print(f"{num} is a positive odd number.")
elif num < 0:
    print(f"{num} is a negative number.")
else:
    print(f"{num} is zero.")

# Conditional expressions (ternary operator)
age = 20
status = "adult" if age >= 18 else "minor"
print(f"Status: {status}")

# Loops
print("\nLOOPS")
print("-" * 30)

# For loop with a list
print("Iterating through a list:")
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"- {fruit}")

# For loop with range
print("\nUsing range():")
for i in range(5):  # 0 to 4
    print(f"Number: {i}")

# For loop with range (start, stop)
print("\nUsing range(start, stop):")
for i in range(2, 6):  # 2 to 5
    print(f"Number: {i}")

# For loop with range (start, stop, step)
print("\nUsing range(start, stop, step):")
for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
    print(f"Number: {i}")

# While loop
print("\nWhile loop:")
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Break statement
print("\nBreak statement:")
for i in range(10):
    if i == 5:
        break
    print(f"Number: {i}")

# Continue statement
print("\nContinue statement:")
for i in range(10):
    if i % 2 == 0:
        continue
    print(f"Odd number: {i}")

# Else clause with loops
print("\nElse clause with for loop:")
for i in range(5):
    print(f"Number: {i}")
else:
    print("Loop completed successfully!")

# Else clause with break
print("\nElse clause with break:")
for i in range(5):
    if i == 3:
        break
    print(f"Number: {i}")
else:
    print("This won't be printed because the loop was broken!")

# Nested loops
print("\nNested loops:")
for i in range(1, 4):
    for j in range(1, 4):
        print(f"({i}, {j})", end=" ")
    print()  # New line after each inner loop completes

# EXERCISE 1: FizzBuzz
# Write a program that prints numbers from 1 to 20
# For multiples of 3, print "Fizz" instead of the number
# For multiples of 5, print "Buzz" instead of the number
# For multiples of both 3 and 5, print "FizzBuzz"
print("\nEXERCISE 1: FizzBuzz")
# Your code here:
# Example solution:
for num in range(1, 21):
    if num % 3 == 0 and num % 5 == 0:
        print("FizzBuzz")
    elif num % 3 == 0:
        print("Fizz")
    elif num % 5 == 0:
        print("Buzz")
    else:
        print(num)

# EXERCISE 2: Pattern Printing
# Write a program to print the following pattern:
# *
# **
# ***
# ****
# *****
print("\nEXERCISE 2: Pattern Printing")
# Your code here:
# Example solution:
for i in range(1, 6):
    print("*" * i)