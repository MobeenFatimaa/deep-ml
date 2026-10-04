import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    i_vals = list(range(1, n + 1))
    
    # 1. Detrend using PI = 3.14159
    detrended_y = [
        y - 10 * math.sin(2 * PI * i / 10) 
        for i, y in zip(i_vals, consumption_data)
    ]
    
    # 2. Linear Regression (OLS)
    mean_x = sum(i_vals) / n
    mean_y = sum(detrended_y) / n
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(i_vals, detrended_y))
    denominator = sum((x - mean_x) ** 2 for x in i_vals)
    
    m = numerator / denominator
    c = mean_y - m * mean_x
    
    # 3. Predict day 15 base consumption
    pred_base = m * 15 + c
    
    # 4. Add day 15 fluctuation back
    pred_total = pred_base + 10 * math.sin(2 * PI * 15 / 10)
    
    # 5. Round base prediction first, then add 5% safety margin
    rounded_base = round(pred_total)
    final_val = math.ceil(rounded_base * 1.05)
    
    return int(final_val)