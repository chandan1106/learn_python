"""
Machine Learning Basics: Introduction to ML Concepts
"""

# This file provides an introduction to machine learning concepts
# In a real environment, you would need to install scikit-learn: pip install scikit-learn
print("Note: This code assumes scikit-learn is installed. If you get an ImportError, install it with: pip install scikit-learn")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets, metrics, model_selection
    from sklearn.linear_model import LogisticRegression
    print("Required libraries successfully imported!")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please install the required libraries with: pip install scikit-learn matplotlib numpy")
    import sys
    sys.exit(1)

# ===== INTRODUCTION TO MACHINE LEARNING =====
print("\n===== INTRODUCTION TO MACHINE LEARNING =====")
"""
Machine Learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed.

Key Concepts:
1. Training Data: The dataset used to train the model
2. Features: The input variables used for prediction
3. Target: The output variable we want to predict
4. Model: The algorithm that learns patterns from data
5. Training: The process of learning patterns from data
6. Prediction: Using the trained model to make predictions on new data
7. Evaluation: Assessing how well the model performs

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data
   - Classification: Predicting categorical labels
   - Regression: Predicting continuous values
   
2. Unsupervised Learning: Learning from unlabeled data
   - Clustering: Grouping similar data points
   - Dimensionality Reduction: Reducing the number of features
   
3. Reinforcement Learning: Learning through trial and error with rewards
"""

# ===== SUPERVISED LEARNING EXAMPLE =====
print("\n===== SUPERVISED LEARNING EXAMPLE =====")

# Load a dataset (Iris dataset)
print("Loading the Iris dataset...")
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train a simple model (Logistic Regression)
print("\nTraining a Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# ===== MACHINE LEARNING WORKFLOW =====
print("\n===== MACHINE LEARNING WORKFLOW =====")
"""
A typical machine learning workflow consists of the following steps:

1. Problem Definition:
   - Define the problem you're trying to solve
   - Determine if it's a classification, regression, clustering, etc.

2. Data Collection:
   - Gather relevant data for your problem
   - Ensure data quality and quantity

3. Data Preprocessing:
   - Handle missing values
   - Encode categorical variables
   - Scale/normalize numerical features
   - Split data into training and testing sets

4. Feature Engineering:
   - Create new features from existing ones
   - Select relevant features
   - Reduce dimensionality if needed

5. Model Selection:
   - Choose appropriate algorithms for your problem
   - Consider model complexity, interpretability, and performance

6. Model Training:
   - Train models on the training data
   - Tune hyperparameters

7. Model Evaluation:
   - Evaluate models on testing data
   - Use appropriate metrics (accuracy, precision, recall, F1, etc.)

8. Model Deployment:
   - Deploy the model to production
   - Monitor performance over time
   - Update as needed
"""

# ===== DATA PREPROCESSING =====
print("\n===== DATA PREPROCESSING =====")

# Create a sample dataset with missing values and categorical features
print("Creating a sample dataset with missing values and categorical features...")
np.random.seed(42)
n_samples = 1000
age = np.random.normal(35, 10, n_samples).astype(int)
age[np.random.choice(n_samples, 50, replace=False)] = np.nan  # Add missing values

income = np.random.normal(50000, 15000, n_samples)
income[np.random.choice(n_samples, 50, replace=False)] = np.nan  # Add missing values

gender = np.random.choice(['Male', 'Female'], n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)

# Create a target variable (e.g., loan approval)
loan_approved = (age > 25) & (income > 40000)
loan_approved = loan_approved.astype(int)
loan_approved[np.isnan(age) | np.isnan(income)] = np.nan  # Missing target for missing features

# Convert to pandas DataFrame for easier handling
import pandas as pd
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Gender': gender,
    'Education': education,
    'LoanApproved': loan_approved
})

print("Sample of the dataset:")
print(data.head())

print("\nMissing values:")
print(data.isna().sum())

# Handle missing values
print("\nHandling missing values...")
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Income'].fillna(data['Income'].mean(), inplace=True)
data['LoanApproved'].fillna(0, inplace=True)  # Conservative approach

# Encode categorical variables
print("\nEncoding categorical variables...")
data_encoded = pd.get_dummies(data, columns=['Gender', 'Education'], drop_first=True)
print("Sample of the encoded dataset:")
print(data_encoded.head())

# Scale numerical features
print("\nScaling numerical features...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_encoded[['Age', 'Income']] = scaler.fit_transform(data_encoded[['Age', 'Income']])
print("Sample of the scaled dataset:")
print(data_encoded.head())

# ===== FEATURE SELECTION =====
print("\n===== FEATURE SELECTION =====")

# Prepare data for feature selection
X = data_encoded.drop('LoanApproved', axis=1)
y = data_encoded['LoanApproved']

# Feature importance using a tree-based model
print("Feature importance using a tree-based model...")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature importance:")
print(feature_importance)

# ===== MODEL SELECTION =====
print("\n===== MODEL SELECTION =====")

# Split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train and evaluate multiple models
print("Training and evaluating multiple models...")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# Find the best model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")

# ===== HYPERPARAMETER TUNING =====
print("\n===== HYPERPARAMETER TUNING =====")

# Hyperparameter tuning for the best model
print(f"Performing hyperparameter tuning for {best_model}...")

if best_model == 'Logistic Regression':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    model = LogisticRegression(random_state=42)
elif best_model == 'Decision Tree':
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    model = DecisionTreeClassifier(random_state=42)
elif best_model == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=42)
elif best_model == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    model = SVC(random_state=42)
else:  # KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    model = KNeighborsClassifier()

# Perform grid search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Tuned model accuracy on test set: {accuracy:.4f}")

# ===== CROSS-VALIDATION =====
print("\n===== CROSS-VALIDATION =====")

# Perform k-fold cross-validation
print("Performing 5-fold cross-validation...")
cv_scores = model_selection.cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# ===== MODEL EVALUATION METRICS =====
print("\n===== MODEL EVALUATION METRICS =====")

# Train the best model on the full training set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Classification metrics
print("Classification metrics:")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {metrics.f1_score(y_test, y_pred):.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# ROC curve and AUC
print("\nROC Curve and AUC:")
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc:.4f}")
else:
    print("This model doesn't support probability predictions for ROC curve.")

# ===== BIAS-VARIANCE TRADEOFF =====
print("\n===== BIAS-VARIANCE TRADEOFF =====")
"""
Bias-Variance Tradeoff:

1. Bias: Error due to overly simplistic assumptions
   - High bias models tend to underfit the data
   - They have high error on both training and test data

2. Variance: Error due to sensitivity to small fluctuations in the training set
   - High variance models tend to overfit the data
   - They have low error on training data but high error on test data

3. Tradeoff: As you decrease bias, you increase variance and vice versa
   - The goal is to find the sweet spot with the right model complexity

4. Ways to manage the tradeoff:
   - Regularization: Add penalties to complex models
   - Cross-validation: Ensure model generalizes well
   - Ensemble methods: Combine multiple models
   - Feature selection: Use only relevant features
"""

# ===== COMMON MACHINE LEARNING ALGORITHMS =====
print("\n===== COMMON MACHINE LEARNING ALGORITHMS =====")
"""
1. Linear Regression:
   - Predicts a continuous target variable
   - Assumes a linear relationship between features and target
   - Pros: Simple, interpretable
   - Cons: Limited to linear relationships

2. Logistic Regression:
   - Predicts binary or multi-class outcomes
   - Outputs probabilities between 0 and 1
   - Pros: Interpretable, provides probabilities
   - Cons: Limited to linear decision boundaries

3. Decision Trees:
   - Creates a tree-like model of decisions
   - Can handle both classification and regression
   - Pros: Easy to understand, handles non-linear relationships
   - Cons: Prone to overfitting

4. Random Forest:
   - Ensemble of decision trees
   - Combines multiple trees to improve performance
   - Pros: Powerful, handles non-linearity, less overfitting
   - Cons: Less interpretable, computationally intensive

5. Support Vector Machines (SVM):
   - Finds the optimal hyperplane to separate classes
   - Can use kernels for non-linear boundaries
   - Pros: Effective in high-dimensional spaces
   - Cons: Computationally intensive, sensitive to parameters

6. K-Nearest Neighbors (KNN):
   - Classifies based on the k closest training examples
   - Simple, non-parametric method
   - Pros: Simple, no training required
   - Cons: Computationally intensive for large datasets

7. Naive Bayes:
   - Based on Bayes' theorem with independence assumptions
   - Commonly used for text classification
   - Pros: Fast, works well with high-dimensional data
   - Cons: Independence assumption often violated

8. Neural Networks:
   - Inspired by the human brain's structure
   - Can model complex non-linear relationships
   - Pros: Highly flexible, can approximate any function
   - Cons: Requires large data, computationally intensive

9. Gradient Boosting:
   - Builds trees sequentially, each correcting errors of previous ones
   - Examples: XGBoost, LightGBM, CatBoost
   - Pros: Often achieves state-of-the-art results
   - Cons: Can overfit, requires careful tuning
"""

# ===== PRACTICAL TIPS =====
print("\n===== PRACTICAL TIPS =====")
"""
1. Start Simple:
   - Begin with simple models before trying complex ones
   - Establish a baseline performance

2. Feature Engineering:
   - Often more important than algorithm selection
   - Domain knowledge is valuable here

3. Cross-Validation:
   - Always validate your model on unseen data
   - Use k-fold cross-validation for reliable estimates

4. Hyperparameter Tuning:
   - Use grid search or random search
   - Consider computational resources

5. Ensemble Methods:
   - Combine multiple models for better performance
   - Techniques: bagging, boosting, stacking

6. Handle Imbalanced Data:
   - Use appropriate metrics (not just accuracy)
   - Consider resampling techniques

7. Regularization:
   - Add penalties to prevent overfitting
   - Common techniques: L1, L2 regularization

8. Monitor Overfitting:
   - Watch the gap between training and validation performance
   - Use early stopping when appropriate

9. Data Leakage:
   - Ensure validation data doesn't influence training
   - Perform preprocessing within cross-validation

10. Interpretability:
    - Consider model interpretability for real-world applications
    - Use tools like SHAP values or LIME for complex models
"""

print("\n===== END OF MACHINE LEARNING BASICS =====")