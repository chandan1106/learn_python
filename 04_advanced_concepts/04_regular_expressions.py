"""
Advanced Python Concepts: Regular Expressions
"""

import re

# ===== REGULAR EXPRESSIONS BASICS =====
print("===== REGULAR EXPRESSIONS BASICS =====")

# Simple pattern matching
text = "Hello, my phone number is 555-123-4567 and my email is example@email.com"

# Find a phone number pattern
phone_pattern = r'\d{3}-\d{3}-\d{4}'
phone_match = re.search(phone_pattern, text)
if phone_match:
    print(f"Found phone number: {phone_match.group()}")

# Find an email pattern
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
email_match = re.search(email_pattern, text)
if email_match:
    print(f"Found email: {email_match.group()}")
print()

# ===== REGEX SPECIAL CHARACTERS =====
print("===== REGEX SPECIAL CHARACTERS =====")

# . (dot) - matches any character except newline
print("Dot pattern example:")
dot_pattern = r'h.t'
dot_text = "The hit hat hot hut"
dot_matches = re.findall(dot_pattern, dot_text)
print(f"Matches for '{dot_pattern}': {dot_matches}")

# ^ (caret) - matches start of string
print("\nCaret pattern example:")
caret_pattern = r'^The'
caret_text1 = "The beginning of a line"
caret_text2 = "Not the beginning of a line"
print(f"'{caret_pattern}' matches '{caret_text1}': {bool(re.match(caret_pattern, caret_text1))}")
print(f"'{caret_pattern}' matches '{caret_text2}': {bool(re.match(caret_pattern, caret_text2))}")

# $ (dollar) - matches end of string
print("\nDollar pattern example:")
dollar_pattern = r'end$'
dollar_text1 = "This is the end"
dollar_text2 = "The end is near"
print(f"'{dollar_pattern}' matches '{dollar_text1}': {bool(re.search(dollar_pattern, dollar_text1))}")
print(f"'{dollar_pattern}' matches '{dollar_text2}': {bool(re.search(dollar_pattern, dollar_text2))}")

# * (asterisk) - matches 0 or more repetitions
print("\nAsterisk pattern example:")
asterisk_pattern = r'ab*c'
asterisk_text = "ac abc abbc abbbc"
asterisk_matches = re.findall(asterisk_pattern, asterisk_text)
print(f"Matches for '{asterisk_pattern}': {asterisk_matches}")

# + (plus) - matches 1 or more repetitions
print("\nPlus pattern example:")
plus_pattern = r'ab+c'
plus_text = "ac abc abbc abbbc"
plus_matches = re.findall(plus_pattern, plus_text)
print(f"Matches for '{plus_pattern}': {plus_matches}")

# ? (question mark) - matches 0 or 1 repetition
print("\nQuestion mark pattern example:")
question_pattern = r'ab?c'
question_text = "ac abc abbc"
question_matches = re.findall(question_pattern, question_text)
print(f"Matches for '{question_pattern}': {question_matches}")

# {} (braces) - matches specific number of repetitions
print("\nBraces pattern example:")
braces_pattern = r'a{2,3}'
braces_text = "a aa aaa aaaa"
braces_matches = re.findall(braces_pattern, braces_text)
print(f"Matches for '{braces_pattern}': {braces_matches}")

# [] (square brackets) - matches any character in the brackets
print("\nSquare brackets pattern example:")
brackets_pattern = r'[aeiou]'
brackets_text = "Hello World"
brackets_matches = re.findall(brackets_pattern, brackets_text)
print(f"Matches for '{brackets_pattern}': {brackets_matches}")

# | (pipe) - matches either expression
print("\nPipe pattern example:")
pipe_pattern = r'cat|dog'
pipe_text = "I have a cat and a dog"
pipe_matches = re.findall(pipe_pattern, pipe_text)
print(f"Matches for '{pipe_pattern}': {pipe_matches}")

# () (parentheses) - groups expressions
print("\nParentheses pattern example:")
group_pattern = r'(ab)+'
group_text = "ab abab ababab"
group_matches = re.findall(group_pattern, group_text)
print(f"Matches for '{group_pattern}': {group_matches}")
print()

# ===== CHARACTER CLASSES =====
print("===== CHARACTER CLASSES =====")

# \d - matches any digit
digit_pattern = r'\d+'
digit_text = "I have 2 apples and 3 oranges"
digit_matches = re.findall(digit_pattern, digit_text)
print(f"Digit matches: {digit_matches}")

# \w - matches any alphanumeric character and underscore
word_pattern = r'\w+'
word_text = "Hello_world! Python_3.9"
word_matches = re.findall(word_pattern, word_text)
print(f"Word matches: {word_matches}")

# \s - matches any whitespace character
space_pattern = r'\s+'
space_text = "Hello   world\t!\nPython"
space_matches = re.findall(space_pattern, space_text)
print(f"Whitespace matches: {len(space_matches)} whitespace groups")

# \D, \W, \S - negated versions of the above
print("\nNegated character classes:")
not_digit_pattern = r'\D+'
not_digit_text = "abc123def456"
not_digit_matches = re.findall(not_digit_pattern, not_digit_text)
print(f"Non-digit matches: {not_digit_matches}")
print()

# ===== REGEX FUNCTIONS =====
print("===== REGEX FUNCTIONS =====")

sample_text = "Python was created in 1991 by Guido van Rossum. Python 2.0 was released in 2000, and Python 3.0 was released in 2008."

# re.search() - find first match
print("re.search() example:")
search_result = re.search(r'Python \d\.\d', sample_text)
if search_result:
    print(f"First match: {search_result.group()}")
    print(f"Start position: {search_result.start()}")
    print(f"End position: {search_result.end()}")

# re.findall() - find all matches
print("\nre.findall() example:")
findall_result = re.findall(r'Python \d\.\d', sample_text)
print(f"All matches: {findall_result}")

# re.finditer() - find all matches as iterator
print("\nre.finditer() example:")
finditer_result = re.finditer(r'Python \d\.\d', sample_text)
for match in finditer_result:
    print(f"Match: {match.group()} at position {match.start()}-{match.end()}")

# re.sub() - substitute matches
print("\nre.sub() example:")
sub_result = re.sub(r'Python (\d\.\d)', r'Python version \1', sample_text)
print(f"After substitution: {sub_result}")

# re.split() - split string by pattern
print("\nre.split() example:")
split_result = re.split(r'\.', sample_text)
print(f"After splitting by period: {split_result}")
print()

# ===== CAPTURING GROUPS =====
print("===== CAPTURING GROUPS =====")

# Basic capturing groups
date_text = "Today is 2023-11-15 and tomorrow is 2023-11-16"
date_pattern = r'(\d{4})-(\d{2})-(\d{2})'

print("Basic capturing groups:")
for match in re.finditer(date_pattern, date_text):
    print(f"Full date: {match.group(0)}")
    print(f"Year: {match.group(1)}")
    print(f"Month: {match.group(2)}")
    print(f"Day: {match.group(3)}")

# Named capturing groups
print("\nNamed capturing groups:")
named_pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
for match in re.finditer(named_pattern, date_text):
    print(f"Full date: {match.group(0)}")
    print(f"Year: {match.group('year')}")
    print(f"Month: {match.group('month')}")
    print(f"Day: {match.group('day')}")

# Non-capturing groups
print("\nNon-capturing groups:")
non_capturing_pattern = r'(?:\d{4})-(\d{2})-(\d{2})'
for match in re.finditer(non_capturing_pattern, date_text):
    print(f"Full date: {match.group(0)}")
    print(f"Month: {match.group(1)}")  # First group is now month
    print(f"Day: {match.group(2)}")    # Second group is now day
print()

# ===== LOOKAHEAD AND LOOKBEHIND =====
print("===== LOOKAHEAD AND LOOKBEHIND =====")

# Positive lookahead
print("Positive lookahead:")
pos_lookahead_text = "apple orange banana grape"
pos_lookahead_pattern = r'\w+(?= orange)'  # Word followed by ' orange'
pos_lookahead_match = re.search(pos_lookahead_pattern, pos_lookahead_text)
if pos_lookahead_match:
    print(f"Word before 'orange': {pos_lookahead_match.group()}")

# Negative lookahead
print("\nNegative lookahead:")
neg_lookahead_text = "apple123 orange456 banana789"
neg_lookahead_pattern = r'\w+(?!\d)'  # Word not followed by a digit
neg_lookahead_matches = re.findall(neg_lookahead_pattern, neg_lookahead_text)
print(f"Words not followed by digits: {neg_lookahead_matches}")

# Positive lookbehind
print("\nPositive lookbehind:")
pos_lookbehind_text = "price: $100, cost: $50, value: $200"
pos_lookbehind_pattern = r'(?<=\$)\d+'  # Digits preceded by $
pos_lookbehind_matches = re.findall(pos_lookbehind_pattern, pos_lookbehind_text)
print(f"Prices: {pos_lookbehind_matches}")

# Negative lookbehind
print("\nNegative lookbehind:")
neg_lookbehind_text = "price100 $200 cost50"
neg_lookbehind_pattern = r'(?<!\$)\d+'  # Digits not preceded by $
neg_lookbehind_matches = re.findall(neg_lookbehind_pattern, neg_lookbehind_text)
print(f"Numbers not preceded by $: {neg_lookbehind_matches}")
print()

# ===== PRACTICAL EXAMPLES =====
print("===== PRACTICAL EXAMPLES =====")

# Validating email addresses
def validate_email(email):
    """Validate an email address using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

print("Email validation:")
emails = ["user@example.com", "invalid@email", "another.valid@email.co.uk", "no_at_symbol.com"]
for email in emails:
    print(f"{email}: {'Valid' if validate_email(email) else 'Invalid'}")

# Extracting URLs from text
def extract_urls(text):
    """Extract URLs from text"""
    pattern = r'https?://[^\s]+'
    return re.findall(pattern, text)

print("\nURL extraction:")
url_text = "Check out these websites: https://www.example.com and http://another-example.org/page"
urls = extract_urls(url_text)
print(f"Found URLs: {urls}")

# Parsing log files
log_entry = '192.168.1.1 - - [25/Sep/2023:14:23:12 +0000] "GET /index.html HTTP/1.1" 200 1234'
log_pattern = r'(\d+\.\d+\.\d+\.\d+).*\[(\d+/\w+/\d+:\d+:\d+:\d+).*\] "(\w+) ([^"]*)" (\d+) (\d+)'

print("\nLog parsing:")
log_match = re.search(log_pattern, log_entry)
if log_match:
    print(f"IP Address: {log_match.group(1)}")
    print(f"Timestamp: {log_match.group(2)}")
    print(f"Method: {log_match.group(3)}")
    print(f"Path: {log_match.group(4)}")
    print(f"Status Code: {log_match.group(5)}")
    print(f"Response Size: {log_match.group(6)}")

# Password strength checker
def check_password_strength(password):
    """Check password strength using regex"""
    # At least 8 characters, one uppercase, one lowercase, one digit, one special character
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(pattern, password))

print("\nPassword strength checker:")
passwords = ["weak", "Stronger1", "StrongP@ssw0rd", "NoDigit@here"]
for password in passwords:
    print(f"{password}: {'Strong' if check_password_strength(password) else 'Weak'}")
print()

# ===== REGEX FLAGS =====
print("===== REGEX FLAGS =====")

# re.IGNORECASE or re.I
print("Case-insensitive matching:")
text = "Python is awesome. PYTHON is powerful."
pattern = r'python'
matches_case_sensitive = re.findall(pattern, text)
matches_case_insensitive = re.findall(pattern, text, re.IGNORECASE)
print(f"Case sensitive: {matches_case_sensitive}")
print(f"Case insensitive: {matches_case_insensitive}")

# re.MULTILINE or re.M
print("\nMultiline matching:")
multiline_text = "First line\nSecond line\nThird line"
start_pattern = r'^Second'
matches_single_line = re.findall(start_pattern, multiline_text)
matches_multi_line = re.findall(start_pattern, multiline_text, re.MULTILINE)
print(f"Single line mode: {matches_single_line}")
print(f"Multi line mode: {matches_multi_line}")

# re.DOTALL or re.S
print("\nDot matches newline:")
dotall_text = "Line 1\nLine 2"
dot_pattern = r'Line.*Line'
matches_normal = re.findall(dot_pattern, dotall_text)
matches_dotall = re.findall(dot_pattern, dotall_text, re.DOTALL)
print(f"Normal dot: {matches_normal}")
print(f"DOTALL dot: {matches_dotall}")

# re.VERBOSE or re.X
print("\nVerbose mode:")
phone_pattern_verbose = re.compile(r'''
    (\d{3})  # Area code
    -        # Separator
    (\d{3})  # Exchange code
    -        # Separator
    (\d{4})  # Subscriber number
''', re.VERBOSE)

phone_text = "Call me at 555-123-4567"
phone_match = phone_pattern_verbose.search(phone_text)
if phone_match:
    print(f"Phone number parts: {phone_match.groups()}")
print()

# Combining flags
print("Combining flags:")
combined_text = "Python\nJAVA\nC++"
combined_pattern = r'^p.*'
matches_combined = re.findall(combined_pattern, combined_text, re.IGNORECASE | re.MULTILINE)
print(f"With combined flags: {matches_combined}")
print()

print("===== END OF REGULAR EXPRESSIONS TUTORIAL =====")