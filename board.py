"""Board representation and game-state management for n×n Tic-Tac-Toe."""

import copy


class Board:
    """Represents an n×n Tic-Tac-Toe board."""

    EMPTY = '.'
    X = 'X'
    O = 'O'

    def __init__(self, size=3):
        if size < 3:
            raise ValueError("Board size must be at least 3")
        self.size = size
        self.grid = [[self.EMPTY] * size for _ in range(size)]
        self.move_count = 0

    def copy(self):
        """Return a deep copy of the board."""
        new_board = Board.__new__(Board)
        new_board.size = self.size
        new_board.grid = [row[:] for row in self.grid]
        new_board.move_count = self.move_count
        return new_board

    def make_move(self, row, col, player):
        """Place a player's symbol at (row, col). Returns True if valid."""
        if not self.is_valid_move(row, col):
            return False
        self.grid[row][col] = player
        self.move_count += 1
        return True

    def undo_move(self, row, col):
        """Remove the symbol at (row, col)."""
        self.grid[row][col] = self.EMPTY
        self.move_count -= 1

    def is_valid_move(self, row, col):
        """Check if a move is within bounds and the cell is empty."""
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == self.EMPTY

    def get_empty_cells(self):
        """Return a list of (row, col) tuples for empty cells."""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r][c] == self.EMPTY]

    def is_full(self):
        """Check if the board has no empty cells."""
        return self.move_count == self.size * self.size

    def check_winner(self):
        """Check if there is a winner. Returns the winning symbol or None."""
        n = self.size

        # Check rows
        for r in range(n):
            if self.grid[r][0] != self.EMPTY and all(self.grid[r][c] == self.grid[r][0] for c in range(n)):
                return self.grid[r][0]

        # Check columns
        for c in range(n):
            if self.grid[0][c] != self.EMPTY and all(self.grid[r][c] == self.grid[0][c] for r in range(n)):
                return self.grid[0][c]

        # Check main diagonal
        if self.grid[0][0] != self.EMPTY and all(self.grid[i][i] == self.grid[0][0] for i in range(n)):
            return self.grid[0][0]

        # Check anti-diagonal
        if self.grid[0][n - 1] != self.EMPTY and all(self.grid[i][n - 1 - i] == self.grid[0][n - 1] for i in range(n)):
            return self.grid[0][n - 1]

        return None

    def is_terminal(self):
        """Check if the game is over (win or draw)."""
        return self.check_winner() is not None or self.is_full()

    def current_player(self):
        """Return whose turn it is (X always goes first)."""
        return self.X if self.move_count % 2 == 0 else self.O

    def display(self):
        """Return a string representation of the board."""
        col_header = "  " + " ".join(str(c) for c in range(self.size))
        rows = []
        for r in range(self.size):
            rows.append(f"{r} " + " ".join(self.grid[r]))
        return col_header + "\n" + "\n".join(rows)

    def __str__(self):
        return self.display()
