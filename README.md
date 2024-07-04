# CODTECH-Task-2

Name:Yashodip Kamble
Company:CODTECH IT SOLUTIONS
ID: CT6DA543
Domain: Data Analytics
Duration: June to August 2024
Mentor: Neela Santosh Kumar

Overview Of Project on Boston Dataset

Project Overview: Predictive Modeling with Linear Regression using the Boston Dataset
1. Introduction

Linear regression is a fundamental machine learning technique used for predicting a continuous target variable based on one or more input features. This project involves implementing a simple linear regression model using the Boston Housing dataset. The goal is to predict the median value of owner-occupied homes (MEDV) based on various features of the houses.
2. Dataset Description

The Boston Housing dataset contains information about various attributes of houses in Boston. The primary target variable is MEDV (Median value of owner-occupied homes in $1000s). Key features include:

    CRIM: Per capita crime rate by town.
    ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
    INDUS: Proportion of non-retail business acres per town.
    CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    NOX: Nitric oxides concentration (parts per 10 million).
    RM: Average number of rooms per dwelling.
    AGE: Proportion of owner-occupied units built prior to 1940.
    DIS: Weighted distances to five Boston employment centers.
    RAD: Index of accessibility to radial highways.
    TAX: Full-value property tax rate per $10,000.
    PTRATIO: Pupil-teacher ratio by town.
    B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
    LSTAT: Percentage of lower status of the population.
    MEDV: Median value of owner-occupied homes in $1000s.

3. Objectives

    Perform exploratory data analysis (EDA) to understand the dataset.
    Implement a simple linear regression model.
    Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²).
    Visualize the regression line and actual vs. predicted values.

4. Steps
4.1. Exploratory Data Analysis (EDA)

    Load the Dataset: Load the Boston Housing dataset.
    Summary Statistics: Display summary statistics to understand the distribution of features.
    Correlation Analysis: Plot a correlation matrix to examine relationships between features.
    Distribution Plots: Visualize the distribution of the target variable (MEDV) and key features.
    Scatter Plots: Create scatter plots to explore relationships between MEDV and significant features.

4.2. Implementing Linear Regression

    Feature Selection: Select relevant features for the regression model. For simplicity, use LSTAT as the feature.
    Data Splitting: Split the data into training and testing sets.
    Model Training: Train the linear regression model using the training data.
    Model Evaluation: Evaluate the model's performance on the test set using MSE and R-squared.
    Visualization: Plot the regression line and actual vs. predicted values to assess model accuracy.

4.3. Code Implementation

Here's the complete code to achieve the above steps:

python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('path/to/your/boston.csv')

# Display the first few rows of the dataset
print(data.head())

# Exploratory Data Analysis (EDA)
# Summary statistics
print(data.describe())

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of the target variable 'MEDV'
plt.figure(figsize=(8, 6))
sns.histplot(data['MEDV'], kde=True, bins=30)
plt.title('Distribution of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()

# Scatter plot between 'LSTAT' and 'MEDV'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LSTAT', y='MEDV', data=data)
plt.title('LSTAT vs. MEDV')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

# Select features and target variable
# For simplicity, let's use 'LSTAT' as the feature and 'MEDV' as the target
X = data[['LSTAT']]
y = data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the regression line and actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Actual')
plt.scatter(X_test, y_pred, alpha=0.7, label='Predicted', color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

# Plot the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

Explanation

    EDA: This part involves understanding the data through summary statistics, correlation matrix, and visualization of distributions and relationships.
    Linear Regression: Implementing and evaluating a simple linear regression model.
    Visualization: Assessing the model’s performance visually through scatter plots and residual analysis.

By following these steps, you will be able to effectively perform EDA and implement a linear regression model to predict house prices using the Boston dataset. This project provides a solid foundation in understanding and applying linear regression for predictive modeling tasks.
![Screenshot from 2024-07-04 23-52-18](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/a31aa7d9-f36c-4ed1-82ae-0d82a112f647)
![Screenshot from 2024-07-04 23-52-24](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/948c9286-1182-40f7-806d-d75e6889c922)
![Screenshot from 2024-07-04 23-52-39](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/70f0a15e-ad86-4896-b23a-f1065c3811c1)
![Screenshot from 2024-07-04 23-52-48](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/5e4fe9c5-b8bf-4f41-9a93-46ca8e49a93f)
![Screenshot from 2024-07-04 23-52-55](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/45f5a6cc-5e57-4951-9d2b-cda490e86ffc)
![Screenshot from 2024-07-04 23-53-02](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/23de3357-66a2-49be-87e8-fc08ef1e9737)
![Screenshot from 2024-07-04 23-53-11](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/6583540b-2cb5-46fe-8594-fb764d8cf7c7)
![Screenshot from 2024-07-04 23-53-18](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/72210692-3b07-4624-a992-d16c67307e47)
![Screenshot from 2024-07-04 23-53-30](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/7f77482e-b1d0-4220-bff8-b38747e87054)
![Screenshot from 2024-07-04 23-53-39](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/d3cae995-4d7d-426e-8b42-5d8b1f266885)
![Screenshot from 2024-07-04 23-54-18](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/dec8f253-b0ae-4573-a47b-36bb3f078de1)
![Screenshot from 2024-07-04 23-54-27](https://github.com/yashodip05/CODTECH-Task-2/assets/132188351/cfb6c11c-e0fb-4975-acc3-4340a3e9da1e)
