import pandas as pd
import numpy as np

def solution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Drop columns with > 50% missing values
    col_threshold = len(df) * 0.5
    # Keep columns that have at least col_threshold non-null values
    df = df.dropna(axis=1, thresh=col_threshold + 1e-9 if len(df) % 2 == 0 else col_threshold)
    # A cleaner alternative using direct condition:
    # valid_cols = [col for col in df.columns if df[col].isna().mean() <= 0.5]
    # df = df[valid_cols]

    # Expressing directly via boolean indexing for maximum clarity and correctness:
    df = df.loc[:, df.isna().mean() <= 0.5]
    
    # 2. Drop rows with > 50% missing values across remaining columns
    df = df.loc[df.isna().mean(axis=1) <= 0.5]
    
    # 3. Fill remaining missing values column-by-column
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                    
    # Return DataFrame with integer index reset
    return df.reset_index(drop=True)