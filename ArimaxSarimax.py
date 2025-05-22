#ARIMA AND SARIMA
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
  
# Generate simple time series and save to CSV
np.random.seed(0)
date_rng = pd.date_range(start='2020-01', periods=100, freq='M')
data = pd.Series(np.random.randn(100).cumsum() + 100, index=date_rng)
data.to_csv("timeseries.csv", header=['value'])

# Load data
df = pd.read_csv("timeseries.csv", parse_dates=[0], index_col=0)

# ARIMA model
arima = ARIMA(df, order=(2, 1, 2)).fit()
arima_pred = arima.forecast(12)

# SARIMA model
sarima = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_pred = sarima.forecast(12)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df, label='Original')
plt.plot(arima_pred.index, arima_pred, label='ARIMA Forecast')
plt.plot(sarima_pred.index, sarima_pred, label='SARIMA Forecast')
plt.legend(); plt.title('ARIMA & SARIMA Forecast'); plt.grid(); plt.show()
