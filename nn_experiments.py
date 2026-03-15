"""Experiments comparing Neural Network heuristic vs handcrafted heuristic."""

import os
import random
import time
import numpy as np
import torch

from board import Board
from ai import AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from benchmark import run_ai_vs_random
from nn_heuristic import (
    BoardNet, encode_board, generate_training_data, train_nn,
    make_nn_heuristic, nn_evaluate_board,
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Experiment 7: NN Training
# ---------------------------------------------------------------------------

def experiment_nn_training(board_size=3, epochs=500):
    """Train the neural network and return loss history."""
    print("\n" + "=" * 60)
    print("  Experiment 7: Neural Network Training")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)
    model, train_losses, val_losses = train_nn(
        board_size=board_size, epochs=epochs, verbose=True
    )

    print(f"\n  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss:   {val_losses[-1]:.6f}")

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Experiment 8: NN Accuracy vs Handcrafted Heuristic
# ---------------------------------------------------------------------------

def experiment_nn_accuracy(board_size=3):
    """Compare NN predictions vs handcrafted heuristic against ground truth."""
    print("\n" + "=" * 60)
    print("  Experiment 8: NN vs Handcrafted Accuracy (3×3)")
    print("=" * 60)

    X_data, y_true = generate_training_data(board_size)
    ai_player, opponent = Board.X, Board.O

    # NN predictions
    input_size = board_size * board_size + 1
    model = BoardNet(input_size=input_size)
    weight_path = f"results/nn_weights_{board_size}x{board_size}.pt"
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        nn_preds = model(torch.tensor(X_data)).squeeze().numpy()

    # Handcrafted heuristic predictions (need to reconstruct boards)
    hc_preds = []
    for encoded in X_data:
        board = _decode_board(encoded, board_size, ai_player, opponent)
        score = evaluate_board(board, ai_player, opponent)
        hc_preds.append(score / 10.0)  # Normalize same as y_true
    hc_preds = np.array(hc_preds)

    # Metrics
    nn_mse = np.mean((nn_preds - y_true) ** 2)
    hc_mse = np.mean((hc_preds - y_true) ** 2)
    nn_corr = np.corrcoef(nn_preds, y_true)[0, 1]
    hc_corr = np.corrcoef(hc_preds, y_true)[0, 1]

    # Sign agreement (does the heuristic agree on who's winning?)
    nn_sign = np.mean(np.sign(nn_preds) == np.sign(y_true))
    hc_sign = np.mean(np.sign(hc_preds) == np.sign(y_true))

    print(f"\n  {'Metric':<25} {'Neural Net':<15} {'Handcrafted':<15}")
    print("  " + "-" * 55)
    print(f"  {'MSE':<25} {nn_mse:<15.6f} {hc_mse:<15.6f}")
    print(f"  {'Correlation':<25} {nn_corr:<15.4f} {hc_corr:<15.4f}")
    print(f"  {'Sign Agreement':<25} {nn_sign:<15.2%} {hc_sign:<15.2%}")

    return {
        "y_true": y_true, "nn_preds": nn_preds, "hc_preds": hc_preds,
        "nn_mse": nn_mse, "hc_mse": hc_mse,
        "nn_corr": nn_corr, "hc_corr": hc_corr,
        "nn_sign": nn_sign, "hc_sign": hc_sign,
    }


def _decode_board(encoded, board_size, ai_player, opponent):
    """Reconstruct a Board object from encoded features."""
    board = Board(board_size)
    idx = 0
    x_count, o_count = 0, 0
    for r in range(board_size):
        for c in range(board_size):
            val = encoded[idx]
            if val > 0.5:
                board.grid[r][c] = ai_player
                if ai_player == Board.X:
                    x_count += 1
                else:
                    o_count += 1
            elif val < -0.5:
                board.grid[r][c] = opponent
                if opponent == Board.X:
                    x_count += 1
                else:
                    o_count += 1
            idx += 1
    board.move_count = x_count + o_count
    return board


# ---------------------------------------------------------------------------
# Experiment 9: NN vs Handcrafted Head-to-Head
# ---------------------------------------------------------------------------

def experiment_nn_vs_handcrafted(num_games=100):
    """AlphaBeta+NN vs AlphaBeta+Handcrafted on 4×4 board."""
    print("\n" + "=" * 60)
    print("  Experiment 9: NN vs Handcrafted Head-to-Head (4×4)")
    print("=" * 60)

    # First train NN for 4x4 if not already done
    weight_path = "results/nn_weights_4x4.pt"
    if not os.path.exists(weight_path):
        print("\n  Training NN for 4×4 board first...")
        train_nn(board_size=4, epochs=300, verbose=True)

    nn_heuristic = make_nn_heuristic(4)
    results = {}

    for depth in [2, 4, 6]:
        nn_wins, hc_wins, draws = 0, 0, 0

        for game_num in range(num_games):
            board = Board(4)
            # Alternate who goes first
            if game_num % 2 == 0:
                nn_sym, hc_sym = Board.X, Board.O
            else:
                nn_sym, hc_sym = Board.O, Board.X

            nn_agent = AlphaBetaAgent(nn_sym, depth_limit=depth,
                                       heuristic=nn_heuristic)
            hc_agent = AlphaBetaAgent(hc_sym, depth_limit=depth,
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

        results[depth] = {"nn_wins": nn_wins, "hc_wins": hc_wins, "draws": draws}
        print(f"\n  Depth {depth}: NN wins={nn_wins}, HC wins={hc_wins}, Draws={draws}")

    return results


# ---------------------------------------------------------------------------
# Experiment 10: Speed Comparison
# ---------------------------------------------------------------------------

def experiment_nn_speed(num_games=50):
    """Compare evaluation speed: NN heuristic vs handcrafted."""
    print("\n" + "=" * 60)
    print("  Experiment 10: NN vs Handcrafted Speed (4×4)")
    print("=" * 60)

    nn_heuristic = make_nn_heuristic(4)
    depth = 4

    # NN-powered AlphaBeta
    print("\n  Running NN-powered Alpha-Beta...")
    nn_result = run_ai_vs_random(AlphaBetaAgent, 4, num_games, depth)

    # We need to pass the NN heuristic — run manually
    nn_total_time = 0.0
    nn_total_moves = 0
    nn_wins = 0
    for _ in range(num_games):
        board = Board(4)
        ai = AlphaBetaAgent(Board.X, depth_limit=depth, heuristic=nn_heuristic)
        rand = RandomAgent(Board.O)
        while not board.is_terminal():
            current = board.current_player()
            if current == Board.X:
                move = ai.get_move(board)
                nn_total_time += ai.stats.elapsed_time
                nn_total_moves += 1
            else:
                move = rand.get_move(board)
            if move:
                board.make_move(move[0], move[1], current)
        if board.check_winner() == Board.X:
            nn_wins += 1

    # Handcrafted AlphaBeta
    print("  Running Handcrafted Alpha-Beta...")
    hc_total_time = 0.0
    hc_total_moves = 0
    hc_wins = 0
    for _ in range(num_games):
        board = Board(4)
        ai = AlphaBetaAgent(Board.X, depth_limit=depth, heuristic=evaluate_board)
        rand = RandomAgent(Board.O)
        while not board.is_terminal():
            current = board.current_player()
            if current == Board.X:
                move = ai.get_move(board)
                hc_total_time += ai.stats.elapsed_time
                hc_total_moves += 1
            else:
                move = rand.get_move(board)
            if move:
                board.make_move(move[0], move[1], current)
        if board.check_winner() == Board.X:
            hc_wins += 1

    nn_avg = nn_total_time / max(nn_total_moves, 1)
    hc_avg = hc_total_time / max(hc_total_moves, 1)

    print(f"\n  {'Method':<20} {'Avg Time/Move':<18} {'Wins/{num_games}':<15}")
    print("  " + "-" * 50)
    print(f"  {'Neural Net':<20} {nn_avg:<18.6f} {nn_wins:<15}")
    print(f"  {'Handcrafted':<20} {hc_avg:<18.6f} {hc_wins:<15}")
    if nn_avg > 0:
        print(f"\n  Speed ratio (HC/NN): {hc_avg / nn_avg:.2f}x")

    return {
        "nn_avg_time": nn_avg, "hc_avg_time": hc_avg,
        "nn_wins": nn_wins, "hc_wins": hc_wins,
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_nn_plots(train_losses, val_losses, accuracy_results=None,
                      h2h_results=None, speed_results=None):
    """Generate visualization plots for NN experiments."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Skipping NN plot generation.")
        return

    os.makedirs("results", exist_ok=True)

    # Plot 1: Training loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train", color="#3498db", linewidth=1)
    ax1.plot(val_losses, label="Validation", color="#e74c3c", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Neural Network Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_losses[10:], label="Train", color="#3498db", linewidth=1)
    ax2.plot(val_losses[10:], label="Validation", color="#e74c3c", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Loss (after epoch 10)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/nn_training_loss.png", dpi=150)
    print("Saved: results/nn_training_loss.png")
    plt.close()

    # Plot 2: Accuracy comparison
    if accuracy_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        y_true = accuracy_results["y_true"]
        nn_preds = accuracy_results["nn_preds"]
        hc_preds = accuracy_results["hc_preds"]

        ax1.scatter(y_true, nn_preds, alpha=0.3, s=10, color="#3498db")
        ax1.plot([-1, 1], [-1, 1], "r--", linewidth=1)
        ax1.set_xlabel("True Minimax Score")
        ax1.set_ylabel("NN Predicted Score")
        ax1.set_title(f"Neural Net (MSE={accuracy_results['nn_mse']:.4f})")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_true, hc_preds, alpha=0.3, s=10, color="#e67e22")
        ax2.plot([-1, 1], [-1, 1], "r--", linewidth=1)
        ax2.set_xlabel("True Minimax Score")
        ax2.set_ylabel("Handcrafted Predicted Score")
        ax2.set_title(f"Handcrafted (MSE={accuracy_results['hc_mse']:.4f})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/nn_accuracy.png", dpi=150)
        print("Saved: results/nn_accuracy.png")
        plt.close()

    # Plot 3: Head-to-head comparison
    if h2h_results:
        fig, ax = plt.subplots(figsize=(8, 5))

        depths = sorted(h2h_results.keys())
        nn_wins = [h2h_results[d]["nn_wins"] for d in depths]
        hc_wins = [h2h_results[d]["hc_wins"] for d in depths]
        draws = [h2h_results[d]["draws"] for d in depths]

        x = range(len(depths))
        width = 0.25
        ax.bar([i - width for i in x], nn_wins, width, label="NN Wins", color="#3498db")
        ax.bar(x, draws, width, label="Draws", color="#f39c12")
        ax.bar([i + width for i in x], hc_wins, width, label="HC Wins", color="#e74c3c")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Depth {d}" for d in depths])
        ax.set_ylabel("Games")
        ax.set_title("NN Heuristic vs Handcrafted Heuristic (4×4)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("results/nn_comparison.png", dpi=150)
        print("Saved: results/nn_comparison.png")
        plt.close()

    # Plot 4: Speed comparison
    if speed_results:
        fig, ax = plt.subplots(figsize=(6, 5))

        methods = ["Neural Net", "Handcrafted"]
        times = [speed_results["nn_avg_time"], speed_results["hc_avg_time"]]
        bars = ax.bar(methods, times, color=["#3498db", "#e67e22"])
        ax.set_ylabel("Avg Time per Move (s)")
        ax.set_title("Evaluation Speed: NN vs Handcrafted (4×4)")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}s", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig("results/nn_speed.png", dpi=150)
        print("Saved: results/nn_speed.png")
        plt.close()


# ---------------------------------------------------------------------------
# Run all NN experiments
# ---------------------------------------------------------------------------

def run_nn_experiments():
    """Run all neural network experiments."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Experiment 7: Training
    model, train_losses, val_losses = experiment_nn_training(board_size=3)

    # Experiment 8: Accuracy
    accuracy_results = experiment_nn_accuracy(board_size=3)

    # Experiment 9: Head-to-head
    h2h_results = experiment_nn_vs_handcrafted(num_games=50)

    # Experiment 10: Speed
    speed_results = experiment_nn_speed(num_games=30)

    # Generate plots
    print("\n" + "=" * 60)
    print("  Generating NN Visualizations")
    print("=" * 60)
    generate_nn_plots(train_losses, val_losses, accuracy_results,
                      h2h_results, speed_results)

    print("\n" + "=" * 60)
    print("  All NN experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_nn_experiments()
