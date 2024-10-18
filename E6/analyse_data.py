import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

def main():
    # Load the timing data from data.csv
    data_frame = pd.read_csv('data.csv')
    
    # Calculate the mean execution times for each sorting algorithm
    data_mean = data_frame.mean()
    print('Sorting implementations and respective mean of their speed:\n')
    print(data_mean)
    print('\n')
    
    # Rank the sorting implementations by their mean execution times
    ranked_data_mean = data_mean.rank(axis=0, ascending=True)
    sorted_rank = ranked_data_mean.sort_values(ascending=True)
    print('Sorting implementations and their respective rank (Ascending):\n')
    print(sorted_rank)
    
    # Perform ANOVA to determine if there is a statistically significant difference between the sorting implementations
    anova = stats.f_oneway(data_frame['qs1'], data_frame['qs2'], data_frame['qs3'], 
                           data_frame['qs4'], data_frame['qs5'], 
                           data_frame['merge1'], data_frame['partition_sort'])
    print('\nANOVA results:\n', anova)
    print('ANOVA p-value:', anova.pvalue, '\n')
    
    # If the ANOVA p-value is < 0.05, we reject the null hypothesis and conclude that at least one sorting algorithm is different
    # Perform Tukey's HSD test to determine which pairs of sorting algorithms have different execution times
    sort_implementation_data = pd.DataFrame({
        'qs1': data_frame['qs1'], 'qs2': data_frame['qs2'], 'qs3': data_frame['qs3'], 
        'qs4': data_frame['qs4'], 'qs5': data_frame['qs5'], 
        'merge1': data_frame['merge1'], 'partition_sort': data_frame['partition_sort']
    })
    
    # Melt the DataFrame for Tukey's HSD test
    sort_implementation_melt = pd.melt(sort_implementation_data)
    
    # Apply Tukey's HSD test
    posthoc = pairwise_tukeyhsd(
        sort_implementation_melt['value'], 
        sort_implementation_melt['variable'],
        alpha=0.05
    )
    print(posthoc)

if __name__ == '__main__':
    main()