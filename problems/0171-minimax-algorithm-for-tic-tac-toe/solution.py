import numpy as np

def is_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    """
    Returns the optimal move (row, col) for the given player ('X' or 'O') on the current board using Minimax.
    Args:
        board: 3x3 NumPy array with entries 'X', 'O', or ''
        player: 'X' or 'O'
    Returns:
        Tuple (row, col) for the optimal move, or None if the game is over.
    """
    # Return None if the game is already over
    if is_winner(board, 'X') or is_winner(board, 'O') or is_full(board):
        return None
    
    opponent = 'O' if player == 'X' else 'X'
    
    def minimax(curr_board, current_turn):
        # Terminal check
        if is_winner(curr_board, player):
            return 1, None
        if is_winner(curr_board, opponent):
            return -1, None
        if is_full(curr_board):
            return 0, None
        
        is_maximizing = (current_turn == player)
        
        if is_maximizing:
            max_eval = -float('inf')
            best_move = None
            for move in get_available_moves(curr_board):
                curr_board[move] = current_turn
                eval_score, _ = minimax(curr_board, opponent)
                curr_board[move] = ''  # Undo move
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in get_available_moves(curr_board):
                curr_board[move] = current_turn
                eval_score, _ = minimax(curr_board, player)
                curr_board[move] = ''  # Undo move
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return min_eval, best_move

    # Run minimax starting with a copy of the board
    _, best_move = minimax(board.copy(), player)
    return best_move