"""AI player implementations: Minimax, Alpha-Beta pruning, and Random agent."""

import random
import time
from board import Board


class AIStats:
    """Tracks statistics for AI move computation."""

    def __init__(self):
        self.nodes_explored = 0
        self.elapsed_time = 0.0

    def reset(self):
        self.nodes_explored = 0
        self.elapsed_time = 0.0


class MinimaxAgent:
    """AI agent using plain Minimax (no pruning)."""

    def __init__(self, player_symbol, depth_limit=None, heuristic=None):
        self.player = player_symbol
        self.opponent = Board.O if player_symbol == Board.X else Board.X
        self.depth_limit = depth_limit
        self.heuristic = heuristic
        self.stats = AIStats()

    def get_move(self, board):
        """Select the best move using Minimax."""
        self.stats.reset()
        start = time.time()

        best_score = float('-inf')
        best_move = None
        is_maximizing_first = (board.current_player() == self.player)

        for row, col in board.get_empty_cells():
            board.make_move(row, col, board.current_player())
            if is_maximizing_first:
                score = self._minimax(board, 1, False)
            else:
                score = self._minimax(board, 1, True)
            board.undo_move(row, col)

            if score > best_score:
                best_score = score
                best_move = (row, col)

        self.stats.elapsed_time = time.time() - start
        return best_move

    def _minimax(self, board, depth, is_maximizing):
        self.stats.nodes_explored += 1

        winner = board.check_winner()
        if winner == self.player:
            return 10 - depth
        elif winner == self.opponent:
            return depth - 10
        elif board.is_full():
            return 0

        if self.depth_limit is not None and depth >= self.depth_limit:
            return self.heuristic(board, self.player, self.opponent)

        if is_maximizing:
            best = float('-inf')
            current = self.player if board.current_player() == self.player else self.opponent
            for row, col in board.get_empty_cells():
                board.make_move(row, col, board.current_player())
                best = max(best, self._minimax(board, depth + 1, False))
                board.undo_move(row, col)
            return best
        else:
            best = float('inf')
            for row, col in board.get_empty_cells():
                board.make_move(row, col, board.current_player())
                best = min(best, self._minimax(board, depth + 1, True))
                board.undo_move(row, col)
            return best


class AlphaBetaAgent:
    """AI agent using Minimax with Alpha-Beta pruning."""

    def __init__(self, player_symbol, depth_limit=None, heuristic=None):
        self.player = player_symbol
        self.opponent = Board.O if player_symbol == Board.X else Board.X
        self.depth_limit = depth_limit
        self.heuristic = heuristic
        self.stats = AIStats()

    def get_move(self, board):
        """Select the best move using Alpha-Beta pruning."""
        self.stats.reset()
        start = time.time()

        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        for row, col in board.get_empty_cells():
            board.make_move(row, col, board.current_player())
            score = self._alphabeta(board, 1, alpha, beta, False)
            board.undo_move(row, col)

            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)

        self.stats.elapsed_time = time.time() - start
        return best_move

    def _alphabeta(self, board, depth, alpha, beta, is_maximizing):
        self.stats.nodes_explored += 1

        winner = board.check_winner()
        if winner == self.player:
            return 10 - depth
        elif winner == self.opponent:
            return depth - 10
        elif board.is_full():
            return 0

        if self.depth_limit is not None and depth >= self.depth_limit:
            return self.heuristic(board, self.player, self.opponent)

        if is_maximizing:
            value = float('-inf')
            for row, col in board.get_empty_cells():
                board.make_move(row, col, board.current_player())
                value = max(value, self._alphabeta(board, depth + 1, alpha, beta, False))
                board.undo_move(row, col)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff
            return value
        else:
            value = float('inf')
            for row, col in board.get_empty_cells():
                board.make_move(row, col, board.current_player())
                value = min(value, self._alphabeta(board, depth + 1, alpha, beta, True))
                board.undo_move(row, col)
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Alpha cutoff
            return value


class RandomAgent:
    """AI agent that makes random moves (used for benchmarking)."""

    def __init__(self, player_symbol):
        self.player = player_symbol
        self.stats = AIStats()

    def get_move(self, board):
        """Select a random valid move."""
        self.stats.reset()
        start = time.time()
        empty = board.get_empty_cells()
        move = random.choice(empty) if empty else None
        self.stats.elapsed_time = time.time() - start
        self.stats.nodes_explored = 1
        return move
