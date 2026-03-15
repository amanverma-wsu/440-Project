"""Benchmarking and performance analysis: Minimax vs Alpha-Beta vs Q-Learning."""

import time
import random
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from qlearning import QLearningAgent


def run_ai_vs_random(ai_class, board_size, num_games=100, depth_limit=None):
    """
    Run AI vs Random agent for a number of games.
    Returns win/draw/loss counts and average stats.
    """
    ai_symbol = Board.X
    random_symbol = Board.O

    heuristic = evaluate_board if depth_limit else None
    wins, draws, losses = 0, 0, 0
    total_nodes = 0
    total_time = 0.0
    total_moves = 0

    for _ in range(num_games):
        board = Board(board_size)
        ai = ai_class(ai_symbol, depth_limit=depth_limit, heuristic=heuristic) if depth_limit else ai_class(ai_symbol)
        rand_agent = RandomAgent(random_symbol)

        while not board.is_terminal():
            current = board.current_player()
            if current == ai_symbol:
                move = ai.get_move(board)
                total_nodes += ai.stats.nodes_explored
                total_time += ai.stats.elapsed_time
                total_moves += 1
            else:
                move = rand_agent.get_move(board)

            if move:
                board.make_move(move[0], move[1], current)

        winner = board.check_winner()
        if winner == ai_symbol:
            wins += 1
        elif winner == random_symbol:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_nodes_per_move": total_nodes / max(total_moves, 1),
        "avg_time_per_move": total_time / max(total_moves, 1),
        "total_moves": total_moves,
    }


def run_qlearning_vs_random(trained_agent, board_size=3, num_games=100):
    """
    Run a pre-trained Q-Learning agent vs Random.
    Returns the same result dict format as run_ai_vs_random.
    """
    ai_symbol = trained_agent.player
    random_symbol = trained_agent.opponent

    wins, draws, losses = 0, 0, 0
    total_time = 0.0
    total_moves = 0

    for _ in range(num_games):
        board = Board(board_size)
        rand_agent = RandomAgent(random_symbol)

        while not board.is_terminal():
            current = board.current_player()
            if current == ai_symbol:
                move = trained_agent.get_move(board)
                total_time += trained_agent.stats.elapsed_time
                total_moves += 1
            else:
                move = rand_agent.get_move(board)

            if move:
                board.make_move(move[0], move[1], current)

        winner = board.check_winner()
        if winner == ai_symbol:
            wins += 1
        elif winner == random_symbol:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_nodes_per_move": 1.0,  # Q-learning is always a single table lookup
        "avg_time_per_move": total_time / max(total_moves, 1),
        "total_moves": total_moves,
    }


def compare_algorithms(board_size=3, num_games=50):
    """Compare Minimax vs Alpha-Beta vs Q-Learning on the same board size."""
    print(f"\n{'=' * 60}")
    print(f"  Comparison: Minimax vs Alpha-Beta vs Q-Learning ({board_size}×{board_size})")
    print(f"  Running {num_games} games each against a random opponent")
    print(f"{'=' * 60}\n")

    depth_limit = None if board_size <= 3 else (6 if board_size == 4 else 4)

    results = {}
    for name, agent_class in [("Minimax", MinimaxAgent), ("Alpha-Beta", AlphaBetaAgent)]:
        print(f"Running {name}...")
        result = run_ai_vs_random(agent_class, board_size, num_games, depth_limit)
        results[name] = result

        print(f"  Wins: {result['wins']}, Draws: {result['draws']}, Losses: {result['losses']}")
        print(f"  Avg nodes/move: {result['avg_nodes_per_move']:.1f}")
        print(f"  Avg time/move:  {result['avg_time_per_move']:.6f}s")
        print()

    # Q-Learning (only for 3×3 where Q-table is practical)
    if board_size == 3:
        print("Running Q-Learning (training 50,000 episodes)...")
        ql_agent = QLearningAgent(Board.X, epsilon=0.3)
        ql_agent.train(num_episodes=50000, board_size=board_size)
        ql_result = run_qlearning_vs_random(ql_agent, board_size, num_games)
        results["Q-Learning"] = ql_result

        print(f"  Wins: {ql_result['wins']}, Draws: {ql_result['draws']}, Losses: {ql_result['losses']}")
        print(f"  Avg nodes/move: {ql_result['avg_nodes_per_move']:.1f} (table lookup)")
        print(f"  Avg time/move:  {ql_result['avg_time_per_move']:.6f}s")
        print()

    # Node reduction
    mm_nodes = results["Minimax"]["avg_nodes_per_move"]
    ab_nodes = results["Alpha-Beta"]["avg_nodes_per_move"]
    if mm_nodes > 0:
        reduction = (1 - ab_nodes / mm_nodes) * 100
        print(f"Alpha-Beta node reduction: {reduction:.1f}%")
        print(f"Alpha-Beta speedup: {results['Minimax']['avg_time_per_move'] / max(results['Alpha-Beta']['avg_time_per_move'], 1e-9):.2f}x")

    return results


def scalability_analysis():
    """Analyze performance across different board sizes."""
    print(f"\n{'=' * 60}")
    print(f"  Scalability Analysis: Alpha-Beta across board sizes")
    print(f"{'=' * 60}\n")

    configs = [
        (3, None, 20),
        (4, 6, 10),
        (5, 4, 5),
    ]

    print(f"{'Size':<6} {'Depth':<7} {'Wins':<6} {'Draws':<7} {'Losses':<8} {'Avg Nodes':<12} {'Avg Time':<12}")
    print("-" * 58)

    for size, depth, games in configs:
        result = run_ai_vs_random(AlphaBetaAgent, size, games, depth)
        depth_str = str(depth) if depth else "full"
        print(f"{size}×{size:<4} {depth_str:<7} {result['wins']:<6} {result['draws']:<7} "
              f"{result['losses']:<8} {result['avg_nodes_per_move']:<12.1f} {result['avg_time_per_move']:<12.6f}s")


def run_nn_vs_handcrafted(board_size=4, num_games=50, depth_limit=4):
    """Run AlphaBeta+NN vs AlphaBeta+Handcrafted head-to-head."""
    try:
        from nn_heuristic import make_nn_heuristic
    except ImportError:
        print("NN heuristic not available.")
        return None

    nn_heuristic = make_nn_heuristic(board_size)
    nn_wins, hc_wins, draws = 0, 0, 0

    for game_num in range(num_games):
        board = Board(board_size)
        if game_num % 2 == 0:
            nn_sym, hc_sym = Board.X, Board.O
        else:
            nn_sym, hc_sym = Board.O, Board.X

        nn_agent = AlphaBetaAgent(nn_sym, depth_limit=depth_limit,
                                   heuristic=nn_heuristic)
        hc_agent = AlphaBetaAgent(hc_sym, depth_limit=depth_limit,
                                   heuristic=evaluate_board)

        while not board.is_terminal():
            current = board.current_player()
            if current == nn_sym:
                move = nn_agent.get_move(board)
            else:
                move = hc_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], current)

        winner = board.check_winner()
        if winner == nn_sym:
            nn_wins += 1
        elif winner == hc_sym:
            hc_wins += 1
        else:
            draws += 1

    return {"nn_wins": nn_wins, "hc_wins": hc_wins, "draws": draws}


if __name__ == "__main__":
    random.seed(42)

    compare_algorithms(board_size=3, num_games=50)
    print()
    scalability_analysis()
