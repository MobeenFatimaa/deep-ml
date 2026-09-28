import pandas as pd

def solution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Calculate group mean and broadcast back to original row structure
    df['group_avg'] = df.groupby('group')['value'].transform('mean')
    
    return df