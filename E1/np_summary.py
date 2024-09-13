import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# 1. Which city had the lowest total precipitation over the year? 
precipitation_sum = np.sum(totals, axis=1)
lowest_precipitation_city = np.argmin(precipitation_sum)
print("Row with lowest total precipitation:")
print(lowest_precipitation_city)

# 2. Determine the average precipitation in these locations for each month
average_precipitation_per_month = np.sum(totals, axis=0) / np.sum(counts, axis=0)
print("Average precipitation in each month:")
print(average_precipitation_per_month)

# 3. Average precipitation per city
average_precipitation_per_city = np.sum(totals, axis=1) / np.sum(counts, axis=1)
print("Average precipitation in each city:")
print(average_precipitation_per_city)

# 4. Calculate the total precipitation for each quarter in each city
quarterly_precipitation = np.sum(totals.reshape(-1, 3), axis=1).reshape(-1, 4)
print("Quarterly precipitation totals:")
print(quarterly_precipitation) 