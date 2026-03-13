"""Heuristic evaluation functions for non-terminal board states on larger boards."""

from board import Board


def evaluate_board(board, ai_player, opponent):
    """
    Heuristic evaluation for non-terminal states.

    Scores based on:
    - Number of lines (rows, cols, diagonals) with potential to win
    - Count of aligned symbols in each line
    - Blocking opportunities
    - Center and corner positional advantage
    """
    score = 0
    n = board.size

    lines = _get_all_lines(board, n)

    for line in lines:
        score += _evaluate_line(line, ai_player, opponent, n)

    # Positional bonus: center is most valuable
    center = n // 2
    if board.grid[center][center] == ai_player:
        score += 3
    elif board.grid[center][center] == opponent:
        score -= 3

    # Corner bonus
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    for r, c in corners:
        if board.grid[r][c] == ai_player:
            score += 1
        elif board.grid[r][c] == opponent:
            score -= 1

    return score


def _get_all_lines(board, n):
    """Extract all rows, columns, and diagonals as lists of cell values."""
    lines = []

    # Rows
    for r in range(n):
        lines.append([board.grid[r][c] for c in range(n)])

    # Columns
    for c in range(n):
        lines.append([board.grid[r][c] for r in range(n)])

    # Main diagonal
    lines.append([board.grid[i][i] for i in range(n)])

    # Anti-diagonal
    lines.append([board.grid[i][n - 1 - i] for i in range(n)])

    return lines


def _evaluate_line(line, ai_player, opponent, n):
    """
    Score a single line based on symbol counts.

    A line that contains both players' symbols has no value.
    Otherwise, more aligned symbols = higher score.
    """
    ai_count = line.count(ai_player)
    opp_count = line.count(opponent)

    # Line blocked by both players — no value
    if ai_count > 0 and opp_count > 0:
        return 0

    # AI-favorable line
    if ai_count > 0:
        if ai_count == n - 1:
            return 50  # One move from winning
        elif ai_count == n - 2:
            return 10
        else:
            return ai_count

    # Opponent-favorable line (threat)
    if opp_count > 0:
        if opp_count == n - 1:
            return -50  # Must block
        elif opp_count == n - 2:
            return -10
        else:
            return -opp_count

    # Empty line
    return 0
