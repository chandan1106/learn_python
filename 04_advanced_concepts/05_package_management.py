"""
Advanced Python Concepts: Package Management
"""

# This file contains information about Python package management
# including pip, virtual environments, and best practices

# ===== PIP BASICS =====
"""
PIP (Pip Installs Packages) is Python's package installer.

Basic pip commands:
------------------
1. Install a package:
   pip install package_name
   
2. Install a specific version:
   pip install package_name==1.2.3
   
3. Install minimum version:
   pip install package_name>=1.2.3
   
4. Upgrade a package:
   pip install --upgrade package_name
   
5. Uninstall a package:
   pip uninstall package_name
   
6. List installed packages:
   pip list
   
7. Show package details:
   pip show package_name
   
8. Search for packages:
   pip search package_name  # Note: This feature may be disabled in newer pip versions
   
9. Install from requirements file:
   pip install -r requirements.txt
"""

# ===== REQUIREMENTS FILES =====
"""
Requirements files allow you to specify exact package versions for your project.

Example requirements.txt:
------------------------
numpy==1.21.0
pandas>=1.3.0,<2.0.0
matplotlib~=3.4.2  # Compatible release (>= 3.4.2, < 3.5.0)
requests

Creating requirements.txt:
------------------------
1. Manually create and edit the file
2. Generate from current environment:
   pip freeze > requirements.txt
"""

# ===== VIRTUAL ENVIRONMENTS =====
"""
Virtual environments isolate project dependencies to avoid conflicts.

Using venv (built into Python 3.3+):
----------------------------------
1. Create a virtual environment:
   python -m venv myenv
   
2. Activate the virtual environment:
   - Windows: myenv\\Scripts\\activate
   - Unix/MacOS: source myenv/bin/activate
   
3. Deactivate the virtual environment:
   deactivate
   
4. Install packages in the virtual environment:
   pip install package_name
   
5. Create requirements.txt from virtual environment:
   pip freeze > requirements.txt
"""

# ===== VIRTUALENV =====
"""
virtualenv is an alternative to venv with more features.

Installation and usage:
---------------------
1. Install virtualenv:
   pip install virtualenv
   
2. Create a virtual environment:
   virtualenv myenv
   
3. Activate/deactivate same as venv
"""

# ===== PIPENV =====
"""
Pipenv combines pip and virtualenv into one tool.

Installation and usage:
---------------------
1. Install pipenv:
   pip install pipenv
   
2. Install packages:
   pipenv install package_name
   
3. Install dev dependencies:
   pipenv install package_name --dev
   
4. Activate the virtual environment:
   pipenv shell
   
5. Run a command in the virtual environment:
   pipenv run python script.py
   
6. Generate requirements.txt:
   pipenv lock -r > requirements.txt
"""

# ===== CONDA =====
"""
Conda is a package, dependency, and environment manager,
popular in data science and scientific computing.

Basic conda commands:
-------------------
1. Create an environment:
   conda create --name myenv python=3.9
   
2. Activate an environment:
   conda activate myenv
   
3. Deactivate an environment:
   conda deactivate
   
4. Install a package:
   conda install package_name
   
5. List installed packages:
   conda list
   
6. Export environment:
   conda env export > environment.yml
   
7. Create environment from file:
   conda env create -f environment.yml
"""

# ===== POETRY =====
"""
Poetry is a modern dependency management and packaging tool.

Installation and usage:
---------------------
1. Install Poetry:
   curl -sSL https://install.python-poetry.org | python3 -
   
2. Create a new project:
   poetry new project_name
   
3. Add dependencies:
   poetry add package_name
   
4. Add dev dependencies:
   poetry add --dev package_name
   
5. Install dependencies:
   poetry install
   
6. Run a command in the virtual environment:
   poetry run python script.py
   
7. Activate the virtual environment:
   poetry shell
"""

# ===== BEST PRACTICES =====
"""
Package Management Best Practices:
--------------------------------
1. Always use virtual environments for projects
2. Pin exact versions in requirements.txt for production
3. Use version ranges for libraries you're developing
4. Regularly update dependencies for security fixes
5. Use a .gitignore file to exclude virtual environments from version control
6. Document installation steps in README.md
7. Consider using dependency scanning tools (e.g., safety, pip-audit)
8. Use lock files when available (Pipfile.lock, poetry.lock)
9. Separate production and development dependencies
10. Audit your dependencies periodically for vulnerabilities
"""

# ===== CREATING PYTHON PACKAGES =====
"""
Creating Your Own Python Package:
-------------------------------
1. Basic structure:
   my_package/
   ├── my_package/
   │   ├── __init__.py
   │   └── module.py
   ├── tests/
   │   └── test_module.py
   ├── README.md
   ├── LICENSE
   └── setup.py
   
2. Minimal setup.py:
   from setuptools import setup, find_packages
   
   setup(
       name="my_package",
       version="0.1.0",
       packages=find_packages(),
       install_requires=[
           "dependency1>=1.0.0",
           "dependency2>=2.0.0",
       ],
   )
   
3. Build the package:
   python setup.py sdist bdist_wheel
   
4. Install in development mode:
   pip install -e .
   
5. Upload to PyPI:
   pip install twine
   twine upload dist/*
"""

# ===== PRACTICAL EXAMPLE =====
"""
Example: Setting up a data science project
-----------------------------------------
1. Create a virtual environment:
   python -m venv ds_env
   
2. Activate the environment:
   source ds_env/bin/activate  # Unix/MacOS
   ds_env\\Scripts\\activate    # Windows
   
3. Install core data science packages:
   pip install numpy pandas matplotlib scikit-learn jupyter
   
4. Create a requirements file:
   pip freeze > requirements.txt
   
5. Create a project structure:
   data_science_project/
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── notebooks/
   │   └── exploration.ipynb
   ├── src/
   │   ├── __init__.py
   │   ├── data_processing.py
   │   └── modeling.py
   ├── tests/
   │   └── test_processing.py
   ├── README.md
   └── requirements.txt
   
6. Document setup instructions in README.md
"""

# ===== DEMONSTRATION =====
# This is a demonstration of how to use a package after installation
# In a real environment, you would need to install these packages first

print("===== PACKAGE MANAGEMENT DEMONSTRATION =====")
print("Note: This is a demonstration. In a real environment, you would need to install these packages first.")

# Example of importing and using packages
try:
    import numpy as np
    print("\nNumPy demonstration:")
    array = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array: {array}")
    print(f"Array mean: {np.mean(array)}")
except ImportError:
    print("\nNumPy is not installed. To install it, run: pip install numpy")

try:
    import pandas as pd
    print("\nPandas demonstration:")
    data = {'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'San Francisco', 'Los Angeles']}
    df = pd.DataFrame(data)
    print("DataFrame:")
    print(df)
except ImportError:
    print("\nPandas is not installed. To install it, run: pip install pandas")

try:
    import matplotlib.pyplot as plt
    print("\nMatplotlib demonstration:")
    print("(Matplotlib would generate a plot here in a normal environment)")
    # In a normal environment with display capabilities:
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    # plt.title('Sample Plot')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.show()
except ImportError:
    print("\nMatplotlib is not installed. To install it, run: pip install matplotlib")

print("\n===== END OF PACKAGE MANAGEMENT DEMONSTRATION =====")

# ===== CONCLUSION =====
"""
Python's package management ecosystem offers multiple tools for different needs:
- pip + venv: Standard, built-in solution
- pipenv: Simplified workflow combining pip and virtualenv
- conda: Popular in scientific computing and data science
- poetry: Modern approach with dependency resolution

Choose the tool that best fits your project requirements and team preferences.
Always use virtual environments to isolate project dependencies.
"""