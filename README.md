# Configurable n×n Tic-Tac-Toe AI

CPTS 440 Project  
Washington State University  

---

## Overview

This project implements a configurable **n×n Tic-Tac-Toe AI system** using both classical adversarial search and modern learning-based techniques.

The system demonstrates:

- **Minimax search** for optimal decision-making  
- **Alpha-Beta pruning** for efficient search  
- **Depth-limited search with heuristics** for scalability  
- **Q-Learning** (reinforcement learning)  
- **Neural Network heuristic** (learning-based evaluation)  

The AI supports **3×3, 4×4, and 5×5 boards**.

---

## Key Results

- Minimax: ~179,000 nodes per move  
- Alpha-Beta: ~6,800 nodes per move  
- **~96% reduction in node exploration**  
- **~25× speed improvement**  

---

## Features

- Configurable board sizes: **3×3, 4×4, 5×5**
- Multiple AI agents
- CLI, GUI, and Web App
- Benchmarking and experiments
- Guided Colab demo notebook

---

## Installation

```bash
git clone https://github.com/amanverma-wsu/440-Project.git
cd 440-Project
pip install -r requirements.txt
```

---

## Run

### CLI
```bash
python game.py
```

### Web App
```bash
python web_app.py
```

Open:
http://127.0.0.1:5000/

---

## Benchmark

```bash
python benchmark.py
```

---

## Experiments

```bash
python experiments.py
```

---

## Demo Notebook

Guided_Demo.ipynb

---

## Presentation

440_AI_TicTacToe.pptx

---

## Authors

Aman Verma  
Nicholas Vendeland  
Jason Lu  
