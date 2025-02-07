import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = np.linspace(-5, 5, 50).reshape(-1, 1)
y = 3*x**3 - 2*x**2 + 4*x + np.random.normal(0, 5, size=x.shape)
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression: Degree 3')
plt.legend()
plt.show()
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
