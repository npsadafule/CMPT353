import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats

# Step 1: Load the data from the CSV into a DataFrame.
data_frame = pd.read_csv('dog_rates_tweets.csv', parse_dates=['created_at'])

# Step 2: Extract numeric ratings from tweets that contain an “n/10” rating
data_frame['Rating'] = pd.to_numeric(data_frame['text'].str.extract(r'(\d+(\.\d+)?)/10')[0])

# Step 3: Remove outliers (ratings greater than 25)
data_frame = data_frame[data_frame['Rating'].notnull() & (data_frame['Rating'] <= 25)]

# Step 4: Create a timestamp column by converting 'created_at' to timestamp
data_frame['timestamp'] = data_frame['created_at'].apply(lambda x: x.timestamp())

# Step 5: Linear regression to find the best-fit line
fit = scipy.stats.linregress(data_frame['timestamp'], data_frame['Rating'])
print(fit.slope, fit.intercept)

# Add a 'prediction' column to the DataFrame
data_frame['prediction'] = data_frame['timestamp'] * fit.slope + fit.intercept

# Step 6: Create a scatter plot of date vs rating, with the best-fit line
plt.figure(figsize=(10, 5))
plt.xticks(rotation=25)
plt.plot(data_frame['created_at'], data_frame['Rating'], 'b.', alpha=0.5, label='Actual Ratings')
plt.plot(data_frame['created_at'], data_frame['prediction'], 'r-', linewidth=3, label='Best Fit Line')
plt.xlabel('Date')
plt.ylabel('Rating')
plt.title('Dog Ratings Over Time with Best-Fit Line')
plt.legend()
plt.show()
