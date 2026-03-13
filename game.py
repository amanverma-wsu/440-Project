"""Command-line interface for playing n×n Tic-Tac-Toe against the AI."""

import sys
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board


def get_ai_agent(player_symbol, board_size, algorithm):
    """Create an AI agent based on board size and selected algorithm."""
    if board_size <= 3:
        # Full-depth search for 3×3
        if algorithm == "minimax":
            return MinimaxAgent(player_symbol)
        else:
            return AlphaBetaAgent(player_symbol)
    else:
        # Depth-limited search with heuristic for larger boards
        depth = 6 if board_size == 4 else 4
        if algorithm == "minimax":
            return MinimaxAgent(player_symbol, depth_limit=depth, heuristic=evaluate_board)
        else:
            return AlphaBetaAgent(player_symbol, depth_limit=depth, heuristic=evaluate_board)


def play_game():
    """Main game loop for human vs AI play."""
    print("=" * 40)
    print("  Tic-Tac-Toe AI (Minimax + Alpha-Beta)")
    print("=" * 40)

    # Board size selection
    while True:
        try:
            size = int(input("\nEnter board size (3-5, default 3): ") or "3")
            if 3 <= size <= 5:
                break
            print("Please enter a size between 3 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Algorithm selection
    print("\nSelect AI algorithm:")
    print("  1. Minimax (no pruning)")
    print("  2. Alpha-Beta pruning (recommended)")
    while True:
        choice = input("Choice (1/2, default 2): ") or "2"
        if choice in ("1", "2"):
            break
        print("Invalid choice.")
    algorithm = "minimax" if choice == "1" else "alphabeta"

    # Player order selection
    print("\nDo you want to play as X (first) or O (second)?")
    while True:
        symbol = input("Choice (X/O, default X): ").upper() or "X"
        if symbol in ("X", "O"):
            break
        print("Invalid choice.")

    board = Board(size)
    human_symbol = symbol
    ai_symbol = Board.O if human_symbol == Board.X else Board.X
    ai = get_ai_agent(ai_symbol, size, algorithm)

    print(f"\nYou are '{human_symbol}', AI is '{ai_symbol}'")
    print(f"Algorithm: {'Minimax' if algorithm == 'minimax' else 'Alpha-Beta Pruning'}")
    if ai.depth_limit:
        print(f"Depth limit: {ai.depth_limit}")
    print(f"Board size: {size}×{size}\n")

    while not board.is_terminal():
        print(board)
        print()

        current = board.current_player()

        if current == human_symbol:
            # Human turn
            while True:
                try:
                    move_input = input(f"Your move ({human_symbol}) - enter row col: ")
                    parts = move_input.strip().split()
                    if len(parts) != 2:
                        print("Enter two numbers separated by a space (e.g., '1 2').")
                        continue
                    row, col = int(parts[0]), int(parts[1])
                    if board.is_valid_move(row, col):
                        board.make_move(row, col, human_symbol)
                        break
                    else:
                        print("Invalid move. Cell is occupied or out of bounds.")
                except (ValueError, IndexError):
                    print("Invalid input. Enter row and column as numbers.")
        else:
            # AI turn
            print(f"AI ({ai_symbol}) is thinking...")
            move = ai.get_move(board)
            if move:
                board.make_move(move[0], move[1], ai_symbol)
                print(f"AI plays: {move[0]} {move[1]}")
                print(f"  Nodes explored: {ai.stats.nodes_explored}")
                print(f"  Time: {ai.stats.elapsed_time:.4f}s")
            print()

    # Game over
    print(board)
    print()
    winner = board.check_winner()
    if winner == human_symbol:
        print("You win!")
    elif winner == ai_symbol:
        print("AI wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    try:
        play_game()
    except (KeyboardInterrupt, EOFError):
        print("\nGame ended.")
        sys.exit(0)
