"""Neural network heuristic evaluation for Tic-Tac-Toe board positions.

Uses a PyTorch feedforward network trained on ground-truth minimax scores
to replace the handcrafted heuristic in depth-limited Alpha-Beta search.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from board import Board


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

class BoardNet(nn.Module):
    """Feedforward network for board evaluation.

    Input:  n*n + 1 features (flattened board + turn indicator)
    Output: single scalar score in [-1, 1] (tanh)
    """

    def __init__(self, input_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board, ai_player, opponent):
    """Convert a Board to a flat numpy array.

    Encoding (relative to ai_player perspective):
        ai_player  -> +1
        opponent   -> -1
        empty      ->  0
    Plus a turn indicator: +1 if it is ai_player's turn, else -1.
    """
    mapping = {ai_player: 1.0, opponent: -1.0, Board.EMPTY: 0.0}
    flat = [mapping[board.grid[r][c]]
            for r in range(board.size) for c in range(board.size)]
    turn = 1.0 if board.current_player() == ai_player else -1.0
    flat.append(turn)
    return np.array(flat, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training data generation (ground-truth minimax on 3x3)
# ---------------------------------------------------------------------------

def _minimax_score(board, ai_player, opponent, depth=0):
    """Full-depth minimax returning the true score for *ai_player*."""
    winner = board.check_winner()
    if winner == ai_player:
        return 10 - depth
    elif winner == opponent:
        return depth - 10
    elif board.is_full():
        return 0

    current = board.current_player()
    is_maximizing = (current == ai_player)

    if is_maximizing:
        best = float('-inf')
        for r, c in board.get_empty_cells():
            board.make_move(r, c, current)
            best = max(best, _minimax_score(board, ai_player, opponent, depth + 1))
            board.undo_move(r, c)
        return best
    else:
        best = float('inf')
        for r, c in board.get_empty_cells():
            board.make_move(r, c, current)
            best = min(best, _minimax_score(board, ai_player, opponent, depth + 1))
            board.undo_move(r, c)
        return best


def generate_training_data(board_size=3):
    """Enumerate all reachable board states and label them with minimax scores.

    Returns (X_data, y_data) as numpy arrays.
    """
    ai_player = Board.X
    opponent = Board.O
    visited = set()
    data_x = []
    data_y = []

    def _enumerate(board):
        key = tuple(board.grid[r][c] for r in range(board.size)
                    for c in range(board.size))
        if key in visited:
            return
        visited.add(key)

        if board.is_terminal():
            return  # Terminal states don't need heuristic evaluation

        score = _minimax_score(board, ai_player, opponent)
        data_x.append(encode_board(board, ai_player, opponent))
        data_y.append(score / 10.0)  # Normalize to [-1, 1]

        current = board.current_player()
        for r, c in board.get_empty_cells():
            board.make_move(r, c, current)
            _enumerate(board)
            board.undo_move(r, c)

    board = Board(board_size)
    _enumerate(board)
    return np.array(data_x), np.array(data_y, dtype=np.float32)


def generate_training_data_sampled(board_size, num_samples=10000, depth_limit=8):
    """Generate training data for larger boards by sampling random game states
    and scoring them with depth-limited Alpha-Beta + handcrafted heuristic.
    """
    from ai import AlphaBetaAgent
    from heuristic import evaluate_board

    ai_player = Board.X
    opponent = Board.O
    data_x = []
    data_y = []
    visited = set()

    for _ in range(num_samples):
        board = Board(board_size)
        # Play random moves to a random depth
        num_moves = random.randint(0, board_size * board_size - 2)
        for _ in range(num_moves):
            cells = board.get_empty_cells()
            if not cells or board.is_terminal():
                break
            r, c = random.choice(cells)
            board.make_move(r, c, board.current_player())

        if board.is_terminal():
            continue

        key = tuple(board.grid[r][c] for r in range(board.size)
                    for c in range(board.size))
        if key in visited:
            continue
        visited.add(key)

        # Score with depth-limited alpha-beta
        agent = AlphaBetaAgent(ai_player, depth_limit=depth_limit,
                               heuristic=evaluate_board)
        score = _minimax_score_limited(board, agent, depth_limit)
        data_x.append(encode_board(board, ai_player, opponent))
        data_y.append(max(-1.0, min(1.0, score / 10.0)))

    return np.array(data_x), np.array(data_y, dtype=np.float32)


def _minimax_score_limited(board, agent, depth_limit):
    """Quick depth-limited minimax score using the agent's heuristic."""
    from heuristic import evaluate_board
    ai = agent.player
    opp = agent.opponent

    def _search(b, depth, is_max):
        w = b.check_winner()
        if w == ai:
            return 10 - depth
        elif w == opp:
            return depth - 10
        elif b.is_full():
            return 0
        if depth >= depth_limit:
            return evaluate_board(b, ai, opp)

        if is_max:
            best = float('-inf')
            for r, c in b.get_empty_cells():
                b.make_move(r, c, b.current_player())
                best = max(best, _search(b, depth + 1, False))
                b.undo_move(r, c)
            return best
        else:
            best = float('inf')
            for r, c in b.get_empty_cells():
                b.make_move(r, c, b.current_player())
                best = min(best, _search(b, depth + 1, True))
                b.undo_move(r, c)
            return best

    return _search(board.copy(), 0, board.current_player() == ai)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nn(board_size=3, epochs=500, lr=0.001, batch_size=64, verbose=True):
    """Train the neural network on minimax-labeled board positions.

    Returns (model, train_losses, val_losses).
    """
    if verbose:
        print(f"Generating training data for {board_size}x{board_size}...")

    if board_size <= 3:
        X_data, y_data = generate_training_data(board_size)
    else:
        X_data, y_data = generate_training_data_sampled(board_size)

    if verbose:
        print(f"  {len(X_data)} board states collected")

    # Train/validation split (80/20)
    n = len(X_data)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X_data[train_idx])
    y_train = torch.tensor(y_data[train_idx]).unsqueeze(1)
    X_val = torch.tensor(X_data[val_idx])
    y_val = torch.tensor(y_data[val_idx]).unsqueeze(1)

    input_size = board_size * board_size + 1
    model = BoardNet(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    if verbose:
        print(f"  Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} — train loss: {epoch_loss:.6f}, val loss: {val_loss:.6f}")

    # Save weights
    os.makedirs("results", exist_ok=True)
    save_path = f"results/nn_weights_{board_size}x{board_size}.pt"
    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"  Model saved to {save_path}")

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Heuristic wrapper (drop-in replacement for evaluate_board)
# ---------------------------------------------------------------------------

_cached_models = {}


def _load_numpy_weights(board_size):
    """Load PyTorch weights and convert to NumPy arrays for fast inference."""
    input_size = board_size * board_size + 1
    weight_path = f"results/nn_weights_{board_size}x{board_size}.pt"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"No trained model found at {weight_path}. "
            f"Run train_nn(board_size={board_size}) first."
        )
    state = torch.load(weight_path, weights_only=True)
    # Extract weights as NumPy: net.0=Linear(in,64), net.2=Linear(64,32), net.4=Linear(32,1)
    w1 = state['net.0.weight'].numpy()
    b1 = state['net.0.bias'].numpy()
    w2 = state['net.2.weight'].numpy()
    b2 = state['net.2.bias'].numpy()
    w3 = state['net.4.weight'].numpy()
    b3 = state['net.4.bias'].numpy()
    return (w1, b1, w2, b2, w3, b3)


def _numpy_forward(weights, x):
    """Fast NumPy forward pass: Linear->ReLU->Linear->ReLU->Linear->Tanh."""
    w1, b1, w2, b2, w3, b3 = weights
    h = np.maximum(0, x @ w1.T + b1)       # ReLU
    h = np.maximum(0, h @ w2.T + b2)       # ReLU
    out = h @ w3.T + b3                     # Linear
    return np.tanh(out)[0]                  # Tanh, scalar


def make_nn_heuristic(board_size=3):
    """Return a heuristic function that uses the trained neural network.

    The returned function has signature: (board, ai_player, opponent) -> float
    matching the existing evaluate_board interface.
    Uses pure NumPy inference for speed.
    """

    def nn_evaluate(board, ai_player, opponent):
        if board_size not in _cached_models:
            _cached_models[board_size] = _load_numpy_weights(board_size)

        weights = _cached_models[board_size]
        encoded = encode_board(board, ai_player, opponent)
        score = _numpy_forward(weights, encoded)
        return float(score) * 10.0

    return nn_evaluate


# Convenience wrapper for 3x3
def nn_evaluate_board(board, ai_player, opponent):
    """Neural network evaluation matching evaluate_board signature."""
    heuristic_fn = make_nn_heuristic(board.size)
    return heuristic_fn(board, ai_player, opponent)


# ---------------------------------------------------------------------------
# Standalone training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    model, train_losses, val_losses = train_nn(board_size=3, epochs=500)
    print(f"\nFinal train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss:   {val_losses[-1]:.6f}")
