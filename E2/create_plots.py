import sys
import matplotlib.pyplot as plt
import pandas as pd

# Get filenames from command line arguments
file1 = sys.argv[1]
file2 = sys.argv[2]

# Read data from the first file
arg1 = pd.read_csv(file1, sep=' ', header=None, index_col=1,
                   names=['lang', 'page', 'views', 'bytes'])

# Sort the data by views in descending order
arg1.sort_values(by=['views'], inplace=True, ascending=False)

# Plot the sorted views from the first file
views = arg1['views'].values
plt.figure(figsize=(10, 5))

# First subplot: Distribution of Views
plt.subplot(1, 2, 1)
plt.title("Distribution of Views (File 1)")
plt.xlabel("Ranks")
plt.ylabel("Views")
plt.plot(views)

# Read data from the second file
arg2 = pd.read_csv(file2, sep=' ', header=None, index_col=1,
                   names=['lang', 'page', 'views', 'bytes'])

# Merge the two dataframes based on the page index
merged_data = arg1[['views']].merge(arg2[['views']], left_index=True, right_index=True, suffixes=('_file1', '_file2'))

# Second subplot: Scatter plot of views from day 1 vs day 2
plt.subplot(1, 2, 2)
plt.title("Daily Views Comparison")
plt.xlabel("Day 1 Views (log scale)")
plt.ylabel("Day 2 Views (log scale)")
plt.xscale('log')
plt.yscale('log')
plt.scatter(merged_data['views_file1'], merged_data['views_file2'])

# Save the plot as an image file
plt.tight_layout()
plt.savefig('wikipedia.png')
