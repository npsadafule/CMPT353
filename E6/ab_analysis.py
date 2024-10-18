import sys
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu  # From the instructions

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)

def main():
    # Step 1: Load the data
    searchdata_file = sys.argv[1]
    json_file = pd.read_json(searchdata_file, orient='records', lines=True)

    # Step 2: Split users into odd (treatment) and even (control) groups based on uid
    odd_users = json_file[json_file['uid'] % 2 != 0]
    even_users = json_file[json_file['uid'] % 2 == 0]

    # Getting rid of NaN values 
    odd_users = odd_users.dropna()
    even_users= even_users.dropna()

    # Step 3: Prepare contingency table for the chi-squared test (Did more users search?)
    odd_users_atleast_once = odd_users[odd_users['search_count'] > 0]
    even_users_atleast_once = even_users[even_users['search_count'] > 0]
    odd_users_never = odd_users[odd_users['search_count'] == 0]
    even_users_never = even_users[even_users['search_count'] == 0]

    contingency_table = [
        [len(odd_users_atleast_once), len(odd_users_never)],
        [len(even_users_atleast_once), len(even_users_never)]
    ]

    # Step 4: Chi-squared test (Did more users use the search feature?)
    chi2, more_users_pvalue, _, _ = chi2_contingency(contingency_table)

    # Step 5: Mann-Whitney U test (Did users search more often?)
    more_searches_pvalue = mannwhitneyu(odd_users['search_count'], even_users['search_count']).pvalue

    # Step 6: Filter for instructors only
    odd_instructors = odd_users[odd_users['is_instructor'] == True]
    even_instructors = even_users[even_users['is_instructor'] == True]

    # Step 7: Prepare contingency table for instructors (Did more instructors search?)
    odd_instructors_atleast_once = odd_instructors[odd_instructors['search_count'] > 0]
    even_instructors_atleast_once = even_instructors[even_instructors['search_count'] > 0]
    odd_instructors_never = odd_instructors[odd_instructors['search_count'] == 0]
    even_instructors_never = even_instructors[even_instructors['search_count'] == 0]

    contingency_table_instructors = [
        [len(odd_instructors_atleast_once), len(odd_instructors_never)],
        [len(even_instructors_atleast_once), len(even_instructors_never)]
    ]

    # Step 8: Chi-squared test for instructors
    chi2_instructors, more_instructors_pvalue, _, _ = chi2_contingency(contingency_table_instructors)

    # Step 9: Mann-Whitney U test for instructors (Did instructors search more often?)
    more_instr_searches_pvalue = mannwhitneyu(
        odd_instructors['search_count'], 
        even_instructors['search_count']
    ).pvalue

    # Step 10: Output the results
    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_pvalue,
        more_searches_p=more_searches_pvalue,
        more_instr_p=more_instructors_pvalue,
        more_instr_searches_p=more_instr_searches_pvalue
    ))

if __name__ == '__main__':
    main()
