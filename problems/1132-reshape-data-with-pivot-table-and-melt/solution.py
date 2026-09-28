import pandas as pd

def solution(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Pivot to wide format, summing sales and filling missing entries with 0
    wide_df = df.pivot_table(
        index='date',
        columns='product',
        values='sales',
        aggfunc='sum',
        fill_value=0
    )
    
    # Step 2: Reset index so 'date' becomes a standard column
    wide_df = wide_df.reset_index()
    wide_df.columns.name = None
    
    # Step 3: Melt wide table back into long format
    long_df = wide_df.melt(
        id_vars=['date'],
        var_name='product',
        value_name='sales'
    )
    
    # Step 4: Sort by date then product (ascending) and reset the index
    result = long_df.sort_values(by=['date', 'product']).reset_index(drop=True)
    
    return result