def tick_bars(ticks: list, bar_size: int) -> list:
    """
    Sample tick data into tick bars.
    
    Args:
        ticks: List of tuples (timestamp, price, volume) representing individual trades
        bar_size: Number of ticks per bar
    
    Returns:
        List of dictionaries with keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    if not ticks or bar_size <= 0:
        return []
    
    bars = []
    n = len(ticks)
    
    # Iterate through ticks in fixed chunks of size `bar_size`
    for i in range(0, n - bar_size + 1, bar_size):
        chunk = ticks[i : i + bar_size]
        
        # Extract components
        timestamps = [t[0] for t in chunk]
        prices = [t[1] for t in chunk]
        volumes = [t[2] for t in chunk]
        
        # Aggregate OHLCV statistics
        bar = {
            'timestamp': timestamps[-1],
            'open': round(float(prices[0]), 2),
            'high': round(float(max(prices)), 2),
            'low': round(float(min(prices)), 2),
            'close': round(float(prices[-1]), 2),
            'volume': round(float(sum(volumes)), 2)
        }
        
        bars.append(bar)
        
    return bars