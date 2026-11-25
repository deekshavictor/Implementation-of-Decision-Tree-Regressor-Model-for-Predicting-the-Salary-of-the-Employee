# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DEEKSHA VICTOR SELVAKUMAR SINDHUPRIYA
RegisterNumber: 212225230041
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [22, 25, 27, 29, 32, 35, 38, 40, 42, 45],
    'Test_Score': [60, 65, 70, 72, 75, 80, 85, 88, 90, 95],
    'Salary': [25000, 30000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
}

df = pd.DataFrame(data)
print("Employee Salary Data:\n", df, "\n")

X = df[['Experience_Years', 'Age', 'Test_Score']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.figure(figsize=(12, 6))
plot_tree(model,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()

new_emp = [[5, 30, 76]]  # Experience, Age, Test_Score
predicted_salary = model.predict(new_emp)
print("\nPredicted Salary for New Employee:", round(predicted_salary[0], 2))
```

## Output:
<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/336588e3-7b3a-4a5e-8dca-3d59b1940e3e" />
<img width="935" height="501" alt="image" src="https://github.com/user-attachments/assets/e56e252d-5c4b-47e7-94d1-40a513ed373a" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
