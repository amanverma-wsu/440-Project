# CLAUDE.md - Project Context for Claude Code

## Project Overview

Configurable n×n Tic-Tac-Toe AI for CptS 440 (Artificial Intelligence) at WSU. Implements multiple game-playing algorithms with experiments comparing their performance, plus a web GUI for interactive play.

**Author:** Aman Verma (aman.verma@wsu.edu)

## Architecture

```
440-Project/
├── Core Game Engine
│   ├── board.py              Board state: n×n grid, moves, win detection
│   ├── ai.py                 MinimaxAgent, AlphaBetaAgent, RandomAgent
│   ├── heuristic.py          Handcrafted board evaluation (line scoring)
│   ├── nn_heuristic.py       PyTorch neural net heuristic + NumPy inference
│   └── qlearning.py          Q-Learning agent with epsilon-greedy exploration
│
├── Interfaces
│   ├── game.py               CLI: choose board size, algorithm, play vs AI
│   ├── web_app.py            Flask web app (REST API on port 8080)
│   ├── gui.py                Pygame desktop GUI
│   └── templates/index.html  Web frontend (HTML/CSS/JS)
│
├── Experiments & Analysis
│   ├── experiments.py         6 experiments: correctness, node comparison, depth, scaling, Q-learning
│   ├── nn_experiments.py      4 NN experiments: training, accuracy, head-to-head, speed
│   └── benchmark.py           Benchmark utilities for all agent types
│
├── Notebooks (Google Colab)
│   ├── tic_tac_toe_notebook.ipynb   Main notebook: all experiments + NN section
│   └── nn_colab.ipynb               Dedicated NN training & evaluation notebook
│
└── results/                   Pre-trained weights (.pt), experiment plots (.png)
```

## Key Abstractions

### Board (`board.py`)
- `Board(size)` — creates n×n board (3-5)
- `Board.X`, `Board.O`, `Board.EMPTY` — cell constants
- `make_move(row, col, player)` / `undo_move(row, col)` — used by search
- `check_winner()` → symbol or None
- `is_terminal()` → bool (win or full)
- `get_empty_cells()` → list of (row, col)
- `current_player()` → X if even moves, O if odd

### AI Agents (`ai.py`)
All agents share: `get_move(board) → (row, col)` and `stats.nodes_explored`, `stats.elapsed_time`

- **MinimaxAgent(player, depth_limit=None, heuristic=None)** — plain minimax
- **AlphaBetaAgent(player, depth_limit=None, heuristic=None)** — alpha-beta pruning
- **RandomAgent(player)** — random valid move (for testing)

The `heuristic` parameter accepts any function with signature `(board, ai_player, opponent) → float`. This is how both `evaluate_board` (handcrafted) and `make_nn_heuristic()` (neural net) plug in.

### Heuristics
- `heuristic.py:evaluate_board(board, ai_player, opponent)` — scores lines, center/corner control
- `nn_heuristic.py:make_nn_heuristic(board_size)` — returns a closure with same signature, uses cached NumPy weights for fast inference (~9μs per eval)

### Neural Network (`nn_heuristic.py`)
- **BoardNet**: Input(n²+1) → Linear(64, ReLU) → Linear(32, ReLU) → Linear(1, Tanh)
- **Encoding**: board cells mapped to +1 (AI), -1 (opponent), 0 (empty) + turn indicator
- **Training**: ground-truth minimax scores for 3×3 (~5,478 states), handcrafted heuristic labels for 4×4/5×5
- **Inference**: Pure NumPy forward pass (no PyTorch at runtime)
- **Pre-trained weights**: `results/nn_weights_{size}x{size}.pt`

### Q-Learning (`qlearning.py`)
- State = tuple of grid, Action = (row, col)
- Rewards: win=1.0, loss=-1.0, draw=0.5
- Epsilon-greedy with decay, trains via self-play
- Q-table saved/loaded as JSON
- Only supports 3×3

## Depth Limits by Board Size

| Size | Default Depth | Heuristic |
|------|--------------|-----------|
| 3×3  | None (full)  | Not needed |
| 4×4  | 6 (standard) / 4 (NN) | evaluate_board or NN |
| 5×5  | 4 (standard) / 2 (NN) | evaluate_board or NN |

## Web App (`web_app.py`)

Flask REST API:
- `GET /` — serves `templates/index.html`
- `POST /api/new` — body: `{size, algorithm, human}` → creates game, returns grid
- `POST /api/move` — body: `{row, col}` → human move + AI response

Run: `python3 web_app.py` (port 8080) or `python3 web_app.py --host` (LAN access)

## Experiments

| # | Name | File | What it tests |
|---|------|------|---------------|
| 1 | Correctness | experiments.py | AI never loses on 3×3 vs random |
| 2 | Minimax vs Alpha-Beta | experiments.py | Node reduction from pruning |
| 3 | Depth Limit Impact | experiments.py | Win rate vs compute at depths 2-8 on 4×4 |
| 4 | Board Size Scaling | experiments.py | Performance across 3×3, 4×4, 5×5 |
| 5 | Q-Learning Training | experiments.py | Learning curve over 50k episodes |
| 6 | Q-Learning Comparison | experiments.py | Q-Learning vs Alpha-Beta vs Random |
| 7 | NN Training | nn_experiments.py | Loss convergence on minimax data |
| 8 | NN Accuracy | nn_experiments.py | MSE, correlation vs ground truth |
| 9 | NN vs Handcrafted | nn_experiments.py | Head-to-head on 4×4 at various depths |
| 10 | NN Speed | nn_experiments.py | Evaluation time comparison |

## Notebook Sections (tic_tac_toe_notebook.ipynb)

1. Setup
2. Board Representation
3. Play Against the AI
4. Experiment 1: Correctness Verification
5. Experiment 2: Minimax vs Alpha-Beta
6. Experiment 3: Depth Limit Impact
7. Experiment 4: Board Size Scaling
8. (Reserved)
9. Neural Network Heuristic Evaluation
10. Summary

## Dependencies

```
numpy          # array operations, NN inference
torch          # NN training only
matplotlib     # experiment plots
flask          # web app
pygame         # desktop GUI (optional)
```

## Running

```bash
# CLI game
python3 game.py

# Web GUI (recommended)
pip install flask numpy torch
python3 web_app.py
# Open http://localhost:8080

# Run all experiments
python3 experiments.py
python3 nn_experiments.py

# Train NN from scratch
python3 nn_heuristic.py
```

## Git

- **Branch**: `claude/tic-tac-toe-ai-game-8TPHb` (development)
- **Main**: push dev branch to main via `git push origin claude/...:main`
- **Commits**: authored as `Aman Verma <aman.verma@wsu.edu>`
