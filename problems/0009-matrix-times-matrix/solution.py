def matrixmul(a: list[list[int | float]], 
              b: list[list[int | float]]) -> list[list[int | float]] | int:
    # Check for empty inputs
    if not a or not b or not a[0] or not b[0]:
        return -1
    
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    
    # Inner dimensions must match
    if cols_a != rows_b:
        return -1
    
    # Initialize output matrix of shape (rows_a x cols_b) with zeros
    c = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # Perform matrix multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                c[i][j] += a[i][k] * b[k][j]
                
    return c