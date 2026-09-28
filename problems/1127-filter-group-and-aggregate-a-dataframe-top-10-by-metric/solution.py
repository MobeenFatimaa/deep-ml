import pandas as pd

def solution(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter rows where status is "completed"
    filtered_df = df[df['status'] == 'completed']
    
    # 2. Group by region, sum the amount, reset index to keep 'region' as a column
    grouped_df = filtered_df.groupby('region', as_index=False)['amount'].sum()
    
    # 3. Sort by amount in descending order, take top 10, and reset integer index
    result = grouped_df.nlargest(10, 'amount').reset_index(drop=True)
    
    return result