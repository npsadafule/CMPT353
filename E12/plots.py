import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn

seaborn.set()  # For nicer plots

# Load and process data
data = pd.read_csv('dog_rates_tweets.csv')
data['rating'] = pd.to_numeric(data['text'].str.extract(r'(\d+(\.\d+)?)/10')[0])
data['created_at'] = pd.to_datetime(data['created_at'])
data = data[(data['rating'] <= 25) & (data['rating'].notnull())]
data['timestamp'] = data['created_at'].apply(lambda x: x.timestamp())

# Linear regression
fit = linregress(data['timestamp'], data['rating'])
data['prediction'] = data['timestamp'] * fit.slope + fit.intercept

# Scatter plot with trendline
plt.figure(figsize=(10, 6))
plt.plot(data['created_at'], data['rating'], 'b.', alpha=0.5, label='Ratings')
plt.plot(data['created_at'], data['prediction'], 'r-', linewidth=2, label='Fit Line')
plt.xticks(rotation=25)
plt.xlabel('Date')
plt.ylabel('Rating')
plt.title('Dog Ratings Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Residuals histogram
residuals = data['rating'] - data['prediction']
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals from Linear Regression')
plt.tight_layout()
plt.show()
