import time
from implementations import all_implementations
import pandas as pd
import numpy as np

def main():
    # Array of sorting implementations to benchmark
    all_implementations_array = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
    
    # Creating a DataFrame to store execution times for each implementation
    data = pd.DataFrame(columns=all_implementations_array)
    
    # Lists to store the timing data for each sorting algorithm
    qs1 = []
    qs2 = []
    qs3 = []
    qs4 = []
    qs5 = []
    merge1 = []
    partition_sort = []

    # List of lists to group the results
    list_array = [qs1, qs2, qs3, qs4, qs5, merge1, partition_sort]
    
    # Run each sorting algorithm on 200 random arrays
    for i in range(200):
        random_array = np.random.randint(0, 2000, 200)
        z = 0
        for sort in all_implementations:
            st = time.time()
            res = sort(random_array)  # Sort the random array
            en = time.time()
            list_array[z].append(en - st)  # Record the time taken
            z += 1
    
    # Store the results in the DataFrame
    data['qs1'] = qs1
    data['qs2'] = qs2
    data['qs3'] = qs3
    data['qs4'] = qs4
    data['qs5'] = qs5
    data['merge1'] = merge1
    data['partition_sort'] = partition_sort

    # Save the DataFrame to a CSV file
    data.to_csv('data.csv', index=False)
    print("Data saved to data.csv")

if __name__ == '__main__':
    main()