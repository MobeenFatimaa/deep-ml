def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    if not dense_matrix or not dense_matrix[0]:
        return [], [], [0]
    
    n_rows = len(dense_matrix)
    n_cols = len(dense_matrix[0])
    
    vals = []
    row_idx = []
    col_ptr = [0]
    
    for c in range(n_cols):
        count = 0
        for r in range(n_rows):
            val = dense_matrix[r][c]
            if val != 0:
                vals.append(val)
                row_idx.append(r)
                count += 1
        col_ptr.append(col_ptr[-1] + count)
        
    return vals, row_idx, col_ptr