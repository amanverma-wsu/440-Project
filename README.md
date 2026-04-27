# Configurable n×n Tic-Tac-Toe AI

CPTS 440 Project  
Washington State University  

## Overview

This project implements a configurable **n×n Tic-Tac-Toe AI system** using adversarial search techniques from artificial intelligence. The primary focus is on **Minimax search**, **Alpha-Beta pruning**, **depth-limited search**, and **heuristic evaluation**.

The system supports **3×3, 4×4, and 5×5** boards. For 3×3 boards, the AI performs full-depth search to guarantee optimal play. For larger boards, where exhaustive search becomes infeasible, the system uses depth-limited search with a heuristic evaluation function.

The project also includes experimental extensions using **Q-Learning** and a **Neural Network heuristic**. These are used for comparison, while the most reliable method remains **Alpha-Beta pruning**.

---

## Features

- Configurable board sizes: **3×3, 4×4, 5×5**
- Multiple AI agents (Minimax, Alpha-Beta, Q-Learning, Neural Net)
- CLI, GUI (Pygame), and Web App (Flask)
- Benchmarking and experiments
- Visualization outputs

---

## Installation

```bash
git clone https://github.com/amanverma-wsu/440-Project.git
cd 440-Project
pip install -r requirements.txt
```

---

## Run

CLI:
```bash
python game.py
```

Web App:
```bash
python webapp.py
```
Open: http://127.0.0.1:5000/

---

## Experiments

```bash
python experiments.py
```
## Benchmarking
```
python benchmark.py
```
---

## Authors

Aman Verma  
Nicholas Vendeland  
Jason Lu
