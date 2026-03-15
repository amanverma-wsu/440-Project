"""Experimental runner and result visualization for Tic-Tac-Toe AI analysis."""

import random
import json
import os
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from benchmark import run_ai_vs_random, run_qlearning_vs_random, compare_algorithms
from qlearning import QLearningAgent

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


def experiment_qlearning_training(num_episodes=50000, board_size=3):
    """
    Train a Q-Learning agent and record its learning curve.
    Returns the reward history for plotting.
    """
    print("\n" + "=" * 60)
    print("  Experiment 5: Q-Learning Training Curve (3×3)")
    print("=" * 60)

    agent = QLearningAgent(Board.X, learning_rate=0.1, discount_factor=0.9,
                           epsilon=0.3)
    print(f"\n  Training for {num_episodes} episodes...")
    rewards = agent.train(num_episodes=num_episodes, board_size=board_size)

    # Compute windowed win-rate at checkpoints
    window = max(1, num_episodes // 20)
    checkpoints = list(range(window, num_episodes + 1, window))
    win_rates = []
    for cp in checkpoints:
        if cp >= 2:
            delta = rewards[cp - 1] - rewards[cp - window - 1] if cp - window - 1 >= 0 else rewards[cp - 1]
        else:
            delta = rewards[cp - 1]
        win_rates.append(delta / window * 100)

    print(f"\n  {'Episode':<10} {'Win Rate (last {window} eps)':<30}")
    print("  " + "-" * 38)
    for cp, wr in zip(checkpoints, win_rates):
        print(f"  {cp:<10} {wr:<30.1f}%")

    print(f"\n  Q-table size: {len(agent.q_table)} entries")
    return rewards, agent


def experiment_qlearning_vs_others(trained_agent, num_games=200):
    """
    Compare a trained Q-Learning agent against Random and Alpha-Beta agents.
    """
    print("\n" + "=" * 60)
    print("  Experiment 6: Q-Learning vs Alpha-Beta vs Random (3×3)")
    print("=" * 60)

    results = {}

    # Q-Learning vs Random
    qr = run_qlearning_vs_random(trained_agent, board_size=3, num_games=num_games)
    results["Q-Learning"] = qr
    print(f"\n  Q-Learning vs Random ({num_games} games):")
    print(f"    Wins: {qr['wins']}, Draws: {qr['draws']}, Losses: {qr['losses']}")

    # Alpha-Beta vs Random
    abr = run_ai_vs_random(AlphaBetaAgent, 3, num_games)
    results["Alpha-Beta"] = abr
    print(f"\n  Alpha-Beta vs Random ({num_games} games):")
    print(f"    Wins: {abr['wins']}, Draws: {abr['draws']}, Losses: {abr['losses']}")

    # Speed comparison
    print(f"\n  Avg time per move:")
    print(f"    Q-Learning:  {qr['avg_time_per_move']:.6f}s (table lookup)")
    print(f"    Alpha-Beta:  {abr['avg_time_per_move']:.6f}s (tree search)")
    if abr['avg_time_per_move'] > 0:
        speedup = abr['avg_time_per_move'] / max(qr['avg_time_per_move'], 1e-9)
        print(f"    Q-Learning speedup: {speedup:.1f}x faster at inference")

    return results


def generate_plots(node_results, depth_results, scaling_results,
                   ql_rewards=None, ql_comparison=None):
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

    # Plot 4: Q-Learning training curve
    if ql_rewards:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Cumulative reward over episodes
        ax1.plot(range(1, len(ql_rewards) + 1), ql_rewards, color="#3498db",
                 linewidth=0.5, alpha=0.7)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("Q-Learning Training: Cumulative Reward")
        ax1.grid(True, alpha=0.3)

        # Smoothed per-episode reward (moving average)
        window = 500
        if len(ql_rewards) > window:
            per_episode = [ql_rewards[0]] + [ql_rewards[i] - ql_rewards[i - 1]
                                              for i in range(1, len(ql_rewards))]
            smoothed = []
            for i in range(len(per_episode)):
                start = max(0, i - window + 1)
                smoothed.append(sum(per_episode[start:i + 1]) / (i - start + 1))
            ax2.plot(range(1, len(smoothed) + 1), smoothed, color="#e74c3c",
                     linewidth=0.8)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Avg Reward (moving window)")
            ax2.set_title(f"Q-Learning Training: Smoothed Reward (window={window})")
            ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
                        label="Draw baseline")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/qlearning_training.png", dpi=150)
        print("Saved: results/qlearning_training.png")
        plt.close()

    # Plot 5: Q-Learning vs Alpha-Beta vs Random comparison
    if ql_comparison:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        agents = list(ql_comparison.keys())
        wins = [ql_comparison[a]["wins"] for a in agents]
        draws = [ql_comparison[a]["draws"] for a in agents]
        losses = [ql_comparison[a]["losses"] for a in agents]

        x = range(len(agents))
        width = 0.25
        ax1.bar([i - width for i in x], wins, width, label="Wins", color="#2ecc71")
        ax1.bar(x, draws, width, label="Draws", color="#f39c12")
        ax1.bar([i + width for i in x], losses, width, label="Losses", color="#e74c3c")
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents)
        ax1.set_ylabel("Games")
        ax1.set_title("Agent Performance vs Random (3×3)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Inference speed comparison
        times = [ql_comparison[a]["avg_time_per_move"] for a in agents]
        bars = ax2.bar(x, times, color=["#3498db", "#9b59b6"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents)
        ax2.set_ylabel("Avg Time per Move (s)")
        ax2.set_title("Inference Speed Comparison (3×3)")
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.6f}s", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig("results/qlearning_comparison.png", dpi=150)
        print("Saved: results/qlearning_comparison.png")
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

    # Experiment 5: Q-Learning training curve
    ql_rewards, trained_agent = experiment_qlearning_training(
        num_episodes=50000, board_size=3)

    # Experiment 6: Q-Learning vs others
    ql_comparison = experiment_qlearning_vs_others(trained_agent, num_games=200)

    # Generate plots
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)
    generate_plots(node_results, depth_results, scaling_results,
                   ql_rewards, ql_comparison)

    # Neural Network experiments
    try:
        from nn_experiments import run_nn_experiments
        run_nn_experiments()
    except ImportError:
        print("\nNN experiments not available (missing nn_heuristic.py or PyTorch)")
    except Exception as e:
        print(f"\nNN experiments failed: {e}")

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
