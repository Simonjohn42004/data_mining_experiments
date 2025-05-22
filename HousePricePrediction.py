#HOUSE PRICE
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create simple dataset
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [200000, 250000, 300000, 350000, 400000]
})

# Train model
model = LinearRegression()
model.fit(data[['area']], data['price'])

# Predict
pred = model.predict(data[['area']])

# Plot
plt.scatter(data['area'], data['price'], color='yellow', label='Actual')
plt.plot(data['area'], pred, color='pink', label='Prediction')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('House Price Prediction')
plt.legend()
plt.show()
