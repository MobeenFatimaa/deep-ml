import pandas as pd

def solution(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    # 1. Inner join on df1 and df2
    merged_df = df1.merge(df2, on='emp_id', how='inner')
    
    # 2. Left join with df3 to attach salary information
    result = merged_df.merge(df3, on='emp_id', how='left')
    
    # 3. Ensure columns are in the required order and reset index
    result = result[['emp_id', 'name', 'dept', 'salary']].reset_index(drop=True)
    
    return result