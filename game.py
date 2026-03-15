"""Command-line interface for playing n×n Tic-Tac-Toe against the AI."""

import sys
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from qlearning import QLearningAgent
from nn_heuristic import make_nn_heuristic


def get_ai_agent(player_symbol, board_size, algorithm):
    """Create an AI agent based on board size and selected algorithm."""
    if algorithm == "qlearning":
        agent = QLearningAgent(player_symbol)
        qtable_path = "qtable.json"
        if agent.load_qtable(qtable_path):
            print("Loaded pre-trained Q-table from qtable.json")
        else:
            print("No saved Q-table found. Training Q-Learning agent (50,000 episodes)...")
            agent.train(num_episodes=50000, board_size=board_size)
            agent.save_qtable(qtable_path)
            print("Training complete. Q-table saved to qtable.json")
        return agent

    if algorithm == "nn":
        nn_heuristic = make_nn_heuristic(board_size)
        depth = None if board_size <= 3 else (6 if board_size == 4 else 4)
        return AlphaBetaAgent(player_symbol, depth_limit=depth, heuristic=nn_heuristic)

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
    print("=" * 45)
    print("  Tic-Tac-Toe AI (Minimax + Alpha-Beta + RL + NN)")
    print("=" * 45)

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
    print("  3. Q-Learning (reinforcement learning, 3×3 only)")
    print("  4. Neural Network heuristic (learned evaluation)")
    while True:
        choice = input("Choice (1/2/3/4, default 2): ") or "2"
        if choice in ("1", "2", "3", "4"):
            break
        print("Invalid choice.")

    if choice == "3" and size != 3:
        print("Q-Learning is only supported for 3×3 boards. Falling back to Alpha-Beta.")
        choice = "2"

    algorithm = {"1": "minimax", "2": "alphabeta", "3": "qlearning", "4": "nn"}[choice]

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
    algo_names = {"minimax": "Minimax", "alphabeta": "Alpha-Beta Pruning",
                  "qlearning": "Q-Learning", "nn": "Neural Network Heuristic"}
    print(f"Algorithm: {algo_names[algorithm]}")
    if hasattr(ai, 'depth_limit') and ai.depth_limit:
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
