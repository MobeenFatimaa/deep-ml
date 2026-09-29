import pandas as pd


def solution(df):
    # 1. Create the bucket column based on integer division by 10
    df["bucket"] = (df["score"] // 10) * 10

    # 2. Group by grade and bucket to get the counts for occurring buckets
    result = df.groupby(["grade", "bucket"]).size().reset_index(name="count")

    # 3. Calculate cumulative counts within each grade group
    result["cum_count"] = result.groupby("grade")["count"].cumsum()

    # 4. Calculate total counts for each grade group
    grade_totals = result.groupby("grade")["count"].transform("sum")

    # 5. Compute the cumulative percentage and round to 2 decimal places
    result["cum_pct"] = (result["cum_count"] / grade_totals * 100).round(2)

    # 6. Sort and clean up output columns
    result = result.sort_values(["grade", "bucket"]).reset_index(drop=True)

    return result[["grade", "bucket", "count", "cum_pct"]]