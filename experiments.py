"""Experimental runner and result visualization for Tic-Tac-Toe AI analysis."""

import random
import json
import os
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from benchmark import run_ai_vs_random, compare_algorithms

# Try to import matplotlib; fall back gracefully if unavailable
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def experiment_correctness_3x3(num_games=200):
    """
    Verify that AlphaBeta AI never loses on 3×3 board.
    Tests both as X (first player) and O (second player).
    """
    print("\n" + "=" * 60)
    print("  Experiment 1: Correctness Verification (3×3)")
    print("=" * 60)

    for ai_sym, opp_sym, label in [(Board.X, Board.O, "AI as X (first)"),
                                    (Board.O, Board.X, "AI as O (second)")]:
        wins, draws, losses = 0, 0, 0
        for _ in range(num_games):
            board = Board(3)
            ai = AlphaBetaAgent(ai_sym)
            rand = RandomAgent(opp_sym)

            while not board.is_terminal():
                current = board.current_player()
                if current == ai_sym:
                    move = ai.get_move(board)
                else:
                    move = rand.get_move(board)
                if move:
                    board.make_move(move[0], move[1], current)

            winner = board.check_winner()
            if winner == ai_sym:
                wins += 1
            elif winner == opp_sym:
                losses += 1
            else:
                draws += 1

        print(f"\n  {label} ({num_games} games vs random):")
        print(f"    Wins: {wins}, Draws: {draws}, Losses: {losses}")
        if losses == 0:
            print("    PASS: AI never lost")
        else:
            print("    FAIL: AI lost some games!")


def experiment_node_comparison(num_games=30):
    """
    Compare node expansions between Minimax and Alpha-Beta on 3×3.
    """
    print("\n" + "=" * 60)
    print("  Experiment 2: Node Expansion Comparison (3×3)")
    print("=" * 60)

    results = compare_algorithms(board_size=3, num_games=num_games)
    return results


def experiment_depth_limit_impact(num_games=10):
    """
    Analyze the impact of different depth limits on 4×4 board performance.
    """
    print("\n" + "=" * 60)
    print("  Experiment 3: Depth Limit Impact (4×4)")
    print("=" * 60)

    depth_limits = [2, 4, 6, 8]
    results = []

    print(f"\n{'Depth':<7} {'Wins':<6} {'Draws':<7} {'Losses':<8} {'Avg Nodes':<12} {'Avg Time (s)':<14}")
    print("-" * 54)

    for depth in depth_limits:
        r = run_ai_vs_random(AlphaBetaAgent, 4, num_games, depth)
        results.append({"depth": depth, **r})
        print(f"{depth:<7} {r['wins']:<6} {r['draws']:<7} {r['losses']:<8} "
              f"{r['avg_nodes_per_move']:<12.1f} {r['avg_time_per_move']:<14.6f}")

    return results


def experiment_board_size_scaling(num_games=10):
    """
    Measure AI performance as board size increases.
    """
    print("\n" + "=" * 60)
    print("  Experiment 4: Board Size Scaling Analysis")
    print("=" * 60)

    configs = [
        (3, None),
        (4, 6),
        (5, 4),
    ]
    results = []

    print(f"\n{'Size':<6} {'Depth':<7} {'Wins':<6} {'Draws':<7} {'Losses':<8} {'Avg Nodes':<12} {'Avg Time (s)':<14}")
    print("-" * 60)

    for size, depth in configs:
        r = run_ai_vs_random(AlphaBetaAgent, size, num_games, depth)
        depth_str = str(depth) if depth else "full"
        results.append({"size": size, "depth": depth_str, **r})
        print(f"{size}×{size:<4} {depth_str:<7} {r['wins']:<6} {r['draws']:<7} {r['losses']:<8} "
              f"{r['avg_nodes_per_move']:<12.1f} {r['avg_time_per_move']:<14.6f}")

    return results


def generate_plots(node_results, depth_results, scaling_results):
    """Generate visualization plots if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Skipping plot generation.")
        print("Install it with: pip install matplotlib")
        return

    os.makedirs("results", exist_ok=True)

    # Plot 1: Minimax vs Alpha-Beta node comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    algorithms = list(node_results.keys())
    nodes = [node_results[a]["avg_nodes_per_move"] for a in algorithms]
    times = [node_results[a]["avg_time_per_move"] for a in algorithms]

    x = range(len(algorithms))
    bars = ax.bar(x, nodes, color=["#e74c3c", "#2ecc71"])
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("Average Nodes Explored per Move")
    ax.set_title("Minimax vs Alpha-Beta: Node Expansions (3×3)")
    for bar, val in zip(bars, nodes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.0f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/node_comparison.png", dpi=150)
    print("Saved: results/node_comparison.png")
    plt.close()

    # Plot 2: Depth limit impact
    if depth_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        depths = [r["depth"] for r in depth_results]
        avg_nodes = [r["avg_nodes_per_move"] for r in depth_results]
        avg_times = [r["avg_time_per_move"] for r in depth_results]
        win_rates = [r["wins"] / max(r["wins"] + r["draws"] + r["losses"], 1) * 100 for r in depth_results]

        ax1.plot(depths, avg_nodes, "o-", color="#3498db", linewidth=2)
        ax1.set_xlabel("Depth Limit")
        ax1.set_ylabel("Avg Nodes per Move")
        ax1.set_title("Depth Limit vs Node Expansions (4×4)")
        ax1.grid(True, alpha=0.3)

        ax2.bar(depths, win_rates, color="#9b59b6", alpha=0.8)
        ax2.set_xlabel("Depth Limit")
        ax2.set_ylabel("Win Rate (%)")
        ax2.set_title("Depth Limit vs Win Rate (4×4)")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("results/depth_impact.png", dpi=150)
        print("Saved: results/depth_impact.png")
        plt.close()

    # Plot 3: Scaling analysis
    if scaling_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sizes = [f"{r['size']}×{r['size']}" for r in scaling_results]
        avg_nodes = [r["avg_nodes_per_move"] for r in scaling_results]
        avg_times = [r["avg_time_per_move"] for r in scaling_results]

        ax1.bar(sizes, avg_nodes, color="#e67e22", alpha=0.8)
        ax1.set_xlabel("Board Size")
        ax1.set_ylabel("Avg Nodes per Move")
        ax1.set_title("Board Size vs Node Expansions")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(sizes, avg_times, color="#1abc9c", alpha=0.8)
        ax2.set_xlabel("Board Size")
        ax2.set_ylabel("Avg Time per Move (s)")
        ax2.set_title("Board Size vs Computation Time")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("results/scaling_analysis.png", dpi=150)
        print("Saved: results/scaling_analysis.png")
        plt.close()


def run_all_experiments():
    """Run all experiments and generate results."""
    random.seed(42)

    # Experiment 1: Correctness
    experiment_correctness_3x3(num_games=100)

    # Experiment 2: Node comparison
    node_results = experiment_node_comparison(num_games=30)

    # Experiment 3: Depth limit impact
    depth_results = experiment_depth_limit_impact(num_games=10)

    # Experiment 4: Scaling
    scaling_results = experiment_board_size_scaling(num_games=10)

    # Generate plots
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)
    generate_plots(node_results, depth_results, scaling_results)

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
