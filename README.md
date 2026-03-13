# Configurable n×n Tic-Tac-Toe AI

An AI opponent for n×n Tic-Tac-Toe using the **Minimax algorithm** with **Alpha-Beta pruning**.

## Project Structure

| File | Description |
|------|-------------|
| `board.py` | Board representation and game-state management |
| `ai.py` | AI agents: Minimax, Alpha-Beta, and Random |
| `heuristic.py` | Heuristic evaluation for larger boards |
| `game.py` | CLI interface for human vs AI gameplay |
| `benchmark.py` | Performance comparison tools |
| `experiments.py` | Full experimental suite with visualization |

## How to Play

```bash
python game.py
```

You can configure:
- **Board size** (3×3 to 5×5)
- **Algorithm** (Minimax or Alpha-Beta)
- **Player order** (X first or O second)

## Running Experiments

```bash
python experiments.py
```

Runs all experiments:
1. Correctness verification (AI never loses on 3×3)
2. Node expansion comparison (Minimax vs Alpha-Beta)
3. Depth limit impact analysis (4×4)
4. Board size scaling analysis

## Running Benchmarks

```bash
python benchmark.py
```

## Requirements

- Python 3.7+
- `matplotlib` (optional, for generating plots)

## Algorithm Details

- **3×3 boards**: Full-depth Minimax guarantees optimal play
- **4×4 boards**: Depth-limited (depth=6) with heuristic evaluation
- **5×5 boards**: Depth-limited (depth=4) with heuristic evaluation

The heuristic evaluates non-terminal states based on:
- Winning line potential and symbol alignment
- Blocking opportunities
- Center and corner positional advantage
