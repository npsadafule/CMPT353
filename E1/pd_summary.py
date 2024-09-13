import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

# 1. Find the city with the lowest total precipitation over the year
lowest_precipitation_city = totals.sum(axis=1).idxmin()
print("City with lowest total precipitation:")
print(lowest_precipitation_city)

# 2. Determine the average precipitation in these locations for each month
average_precipitation_per_month = totals.sum(axis=0) / counts.sum(axis=0)
print("Average precipitation in each month:")
print(average_precipitation_per_month)

# 3. Average precipitation per city
average_precipitation_per_city = totals.sum(axis=1) / counts.sum(axis=1)
print("Average precipitation in each city:")
print(average_precipitation_per_city)
