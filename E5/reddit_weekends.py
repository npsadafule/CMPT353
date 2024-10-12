import sys
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date

# Output template as specified in the assignment
OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    # Step 1: Load the data from the provided reddit-counts.json.gz file
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)

    # Step 2: Filter data to only include comments from /r/canada and within the years 2012 and 2013
    counts['week_day'] = counts['date'].dt.weekday  # Extract the weekday (0-6)
    counts['year'] = counts['date'].dt.year         # Extract the year from the date
    counts_subreddit = counts[counts['subreddit'] == 'canada']  # Filter for only /r/canada
    counts_final = counts_subreddit[(counts_subreddit['year'] == 2012) | (counts_subreddit['year'] == 2013)]  # Limit to 2012 and 2013

    # Step 3: Separate weekdays (0-4) from weekends (5-6)
    weekends = counts_final[(counts_final['week_day'] == 5) | (counts_final['week_day'] == 6)]  # Saturday and Sunday
    weekdays = counts_final[~((counts_final['week_day'] == 5) | (counts_final['week_day'] == 6))]  # Monday to Friday

    # Step 4: Perform the initial T-test to check for differences in comment counts between weekdays and weekends
    initial_p = stats.ttest_ind(weekends['comment_count'], weekdays['comment_count']).pvalue

    # Step 5: Perform normality tests on the original data for both weekdays and weekends
    initial_weekday_n_p = stats.normaltest(weekdays['comment_count']).pvalue
    initial_weekend_n_p = stats.normaltest(weekends['comment_count']).pvalue

    # Step 6: Check for equal variance between the two groups using Levene's test
    initial_levene = stats.levene(weekends['comment_count'], weekdays['comment_count']).pvalue

    # Step 7: Apply transformations (square root) to the data to handle skewness
    # This is because the data might not be normally distributed, and transforming it can help
    transformed_weekdays = np.sqrt(weekdays['comment_count'])
    transformed_weekends = np.sqrt(weekends['comment_count'])

    # Step 8: Run normality tests again on the transformed data
    transformed_weekday_n_p = stats.normaltest(transformed_weekdays).pvalue
    transformed_weekend_n_p = stats.normaltest(transformed_weekends).pvalue

    # Step 9: Re-run Levene's test on the transformed data to check for equal variances again
    transformed_levene = stats.levene(transformed_weekends, transformed_weekdays).pvalue

    # Step 10: Central Limit Theorem - Aggregate data by year and week to analyze weekly comment counts
    # Group by year and week, then take the mean comment count for each week
    weekends_c = weekends.copy()
    weekdays_c = weekdays.copy()

    weekends_c = weekends_c.assign(
        year=weekends['date'].dt.isocalendar().year,  # Extract ISO calendar year
        week=weekends['date'].dt.isocalendar().week   # Extract ISO calendar week
    )
    weekdays_c = weekdays_c.assign(
        year=weekdays['date'].dt.isocalendar().year,  # Extract ISO calendar year
        week=weekdays['date'].dt.isocalendar().week   # Extract ISO calendar week
    )

    # Step 11: Group by year and week to get the mean comment count for each week
    weekends_grouped = weekends_c.groupby(['year', 'week'])[['comment_count']].mean()
    weekdays_grouped = weekdays_c.groupby(['year', 'week'])[['comment_count']].mean()

    # Step 12: Perform normality tests on the weekly mean data
    weekly_weekday_n_p = stats.normaltest(weekdays_grouped['comment_count']).pvalue
    weekly_weekend_n_p = stats.normaltest(weekends_grouped['comment_count']).pvalue

    # Step 13: Perform Levene's test on the weekly mean data to check for equal variances
    weekly_levene = stats.levene(weekends_grouped['comment_count'], weekdays_grouped['comment_count']).pvalue

    # Step 14: Perform a T-test on the weekly mean data to check for differences in comment counts between weekdays and weekends
    weekly_ttest = stats.ttest_ind(weekends_grouped['comment_count'], weekdays_grouped['comment_count']).pvalue

    # Step 15: Apply the Mann-Whitney U-test (a non-parametric test) to check for differences without assuming normal distribution
    Mann_utest = stats.mannwhitneyu(weekends['comment_count'], weekdays['comment_count'], alternative='two-sided').pvalue

    # Step 16: Output all the results in the specified format
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_p,
        initial_weekday_normality_p=initial_weekday_n_p,
        initial_weekend_normality_p=initial_weekend_n_p,
        initial_levene_p=initial_levene,
        transformed_weekday_normality_p=transformed_weekday_n_p,
        transformed_weekend_normality_p=transformed_weekend_n_p,
        transformed_levene_p=transformed_levene,
        weekly_weekday_normality_p=weekly_weekday_n_p,
        weekly_weekend_normality_p=weekly_weekend_n_p,
        weekly_levene_p=weekly_levene,
        weekly_ttest_p=weekly_ttest,
        utest_p=Mann_utest,
    ))


if __name__ == '__main__':
    main()
