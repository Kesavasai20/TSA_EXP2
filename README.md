# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
## Date:19-08-2025
## Name: K KESAVA SAI
## Register Number: 212223230105
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Clean_Dataset.csv")

# Group by days_left and calculate average price
data_grouped = data.groupby("days_left")["price"].mean().reset_index()
data_grouped.rename(columns={"days_left": "DaysLeft", "price": "AvgPrice"}, inplace=True)

# Extract variables
X_vals = data_grouped["DaysLeft"].tolist()
Y_vals = data_grouped["AvgPrice"].tolist()

# Center X around the midpoint
X = [i - X_vals[len(X_vals) // 2] for i in X_vals]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, Y_vals)]

# Linear regression
n = len(X)
b = (n * sum(xy) - sum(Y_vals) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(Y_vals) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

# Polynomial regression (degree 2)
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, Y_vals)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(Y_vals), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

# Print equations
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add results to DataFrame
data_grouped["Linear Trend"] = linear_trend
data_grouped["Polynomial Trend"] = poly_trend

# Set DaysLeft as index
data_grouped.set_index("DaysLeft", inplace=True)

# Plot Linear Trend
data_grouped["AvgPrice"].plot(kind="line", color="blue", marker="o", label="Average Price")
data_grouped["Linear Trend"].plot(kind="line", color="black", linestyle="--", label="Linear Trend")
plt.legend()
plt.title("Linear Trend Estimation on Flight Price vs Days Left")
plt.xlabel("Days Left")
plt.ylabel("Average Price")
plt.show()

# Plot Polynomial Trend
data_grouped["AvgPrice"].plot(kind="line", color="blue", marker="o", label="Average Price")
data_grouped["Polynomial Trend"].plot(kind="line", color="red", marker="o", label="Polynomial Trend (Degree 2)")
plt.legend()
plt.title("Polynomial Trend Estimation on Flight Price vs Days Left")
plt.xlabel("Days Left")
plt.ylabel("Average Price")
plt.show()

```

### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="871" height="627" alt="image" src="https://github.com/user-attachments/assets/678006db-a59b-401f-810d-4038556e3a37" />

B- POLYNOMIAL TREND ESTIMATION
<img width="820" height="586" alt="image" src="https://github.com/user-attachments/assets/a09d919c-22b0-483d-a1f1-a8fa5dc523a9" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
