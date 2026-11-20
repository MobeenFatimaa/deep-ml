import numpy as np


def _solve_simplex_min(c: np.ndarray, A_ub: np.ndarray, b_ub: np.ndarray) -> np.ndarray:
    """
    Solve min c^T x subject to A_ub @ x <= b_ub, x >= 0 using Two-Phase Simplex
    with Bland's anti-cycling rule.
    """
    m_con, n_var = A_ub.shape
    # Convert A_ub @ x <= b_ub into equality constraints A_ub @ x + s = b_ub, s >= 0
    total_vars = n_var + m_con
    A_eq = np.hstack([A_ub, np.eye(m_con, dtype=np.float64)])
    b_eq = b_ub.astype(np.float64).copy()

    # Ensure non-negative RHS
    for r in range(m_con):
        if b_eq[r] < 0:
            A_eq[r] *= -1.0
            b_eq[r] *= -1.0

    eps = 1e-11

    def run_simplex(tableau: np.ndarray, basis: list[int]) -> np.ndarray:
        n_rows, n_cols_total = tableau.shape
        n_dec = n_cols_total - 1
        while True:
            # Reduced costs are in the last row, columns [0 .. n_dec - 1]
            obj_row = tableau[-1, :n_dec]
            entering_candidates = [j for j in range(n_dec) if obj_row[j] < -eps]
            if not entering_candidates:
                break
            enter_col = min(entering_candidates)  # Bland's rule: smallest index

            # Minimum ratio test
            pivot_row = -1
            min_ratio = float("inf")
            col_vals = tableau[:n_rows - 1, enter_col]
            rhs_vals = tableau[:n_rows - 1, -1]
            for r in range(n_rows - 1):
                if col_vals[r] > eps:
                    ratio = rhs_vals[r] / col_vals[r]
                    if ratio < min_ratio - eps or (
                        abs(ratio - min_ratio) <= eps
                        and pivot_row != -1
                        and basis[r] < basis[pivot_row]
                    ):
                        min_ratio = ratio
                        pivot_row = r

            if pivot_row == -1:
                raise RuntimeError("LP is unbounded.")

            # Pivot operation
            pivot_val = tableau[pivot_row, enter_col]
            tableau[pivot_row] /= pivot_val
            for r in range(n_rows):
                if r != pivot_row:
                    factor = tableau[r, enter_col]
                    if abs(factor) > 0.0:
                        tableau[r] -= factor * tableau[pivot_row]
            basis[pivot_row] = enter_col

        return tableau, basis

    # Phase I: minimize sum of artificial variables a_0 .. a_{m_con - 1}
    art_cols = np.eye(m_con, dtype=np.float64)
    phase1_tab = np.zeros((m_con + 1, total_vars + m_con + 1), dtype=np.float64)
    phase1_tab[:m_con, :total_vars] = A_eq
    phase1_tab[:m_con, total_vars : total_vars + m_con] = art_cols
    phase1_tab[:m_con, -1] = b_eq

    # Objective w = sum(a_r) = sum(b_eq_r - sum_j A_eq_{r,j} x_j)
    # In tableau form (row represents objective expression), reduced cost of x_j is -sum_r A_eq_{r,j}
    phase1_tab[-1, :total_vars] = -np.sum(A_eq, axis=0)
    phase1_tab[-1, -1] = -np.sum(b_eq)

    basis = list(range(total_vars, total_vars + m_con))
    phase1_tab, basis = run_simplex(phase1_tab, basis)

    if phase1_tab[-1, -1] < -1e-7:
        raise RuntimeError("LP is infeasible.")

    # Drive any remaining artificial variables out of the basis if possible
    for r in range(m_con):
        if basis[r] >= total_vars:
            for j in range(total_vars):
                if abs(phase1_tab[r, j]) > eps:
                    pivot_val = phase1_tab[r, j]
                    phase1_tab[r] /= pivot_val
                    for rr in range(m_con + 1):
                        if rr != r:
                            phase1_tab[rr] -= phase1_tab[rr, j] * phase1_tab[r]
                    basis[r] = j
                    break

    # Phase II: build tableau for original objective min c^T x
    phase2_tab = np.zeros((m_con + 1, total_vars + 1), dtype=np.float64)
    phase2_tab[:m_con, :total_vars] = phase1_tab[:m_con, :total_vars]
    phase2_tab[:m_con, -1] = phase1_tab[:m_con, -1]

    obj_coeffs = np.zeros(total_vars, dtype=np.float64)
    obj_coeffs[:n_var] = c
    phase2_tab[-1, :total_vars] = obj_coeffs

    # Zero out reduced costs for current basic variables
    for r, b_col in enumerate(basis):
        if b_col < total_vars:
            coeff = phase2_tab[-1, b_col]
            if abs(coeff) > 0.0:
                phase2_tab[-1] -= coeff * phase2_tab[r]

    phase2_tab, basis = run_simplex(phase2_tab, basis)

    sol = np.zeros(n_var, dtype=np.float64)
    for r, b_col in enumerate(basis):
        if b_col < n_var:
            sol[b_col] = max(0.0, phase2_tab[r, -1])
    return sol


def _optimal_row_strategy(payoff_matrix: np.ndarray) -> tuple[float, list[float]]:
    """Return (game_value, row_strategy) for row player on payoff_matrix (m x n)."""
    m, n = payoff_matrix.shape
    min_entry = float(np.min(payoff_matrix))
    shift = max(0.0, -min_entry) + 1.0
    shifted = payoff_matrix + shift

    # min sum(x_i) s.t. -(shifted^T) @ x <= -1, x >= 0
    c = np.ones(m, dtype=np.float64)
    A_ub = -shifted.T
    b_ub = -np.ones(n, dtype=np.float64)

    x = _solve_simplex_min(c, A_ub, b_ub)
    x_sum = float(np.sum(x))
    if x_sum <= 1e-12:
        p = np.full(m, 1.0 / m, dtype=np.float64)
        val_shifted = float(np.min(p @ shifted))
    else:
        val_shifted = 1.0 / x_sum
        p = x * val_shifted
        p = np.maximum(0.0, p)
        p_sum = float(np.sum(p))
        if p_sum > 0:
            p /= p_sum

    game_value = float(val_shifted - shift)
    return game_value, [float(val) for val in p]


def solve_zero_sum(payoff):
    """Return (value, row_strategy, col_strategy) of the zero-sum game with this payoff matrix."""
    A = np.asarray(payoff, dtype=np.float64)
    value, row_strategy = _optimal_row_strategy(A)
    neg_val, col_strategy = _optimal_row_strategy(-A.T)
    # Both value and -neg_val equal the true value v; average for numerical tidiness
    final_value = float(0.5 * (value - neg_val))
    return final_value, row_strategy, col_strategy