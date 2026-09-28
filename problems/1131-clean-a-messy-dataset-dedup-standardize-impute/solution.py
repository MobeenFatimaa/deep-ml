import pandas as pd

def solution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Step 1: Trim surrounding whitespace and convert name to title case
    df['name'] = df['name'].str.strip().str.title()
    
    # Step 2: Strip whitespace from date and convert to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'].str.strip()).dt.strftime('%Y-%m-%d')
    
    # Step 3: Remove duplicate rows (keeping first) and reset index
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    
    # Step 4: Fill remaining missing values in 'value' with the mean of non-missing values
    mean_val = df['value'].mean()
    df['value'] = df['value'].fillna(mean_val)
    
    return df