"""Q-Learning agent for n×n Tic-Tac-Toe."""

import random
import time
import json
import os
from collections import defaultdict
from board import Board
from ai import AIStats, RandomAgent


class QLearningAgent:
    """AI agent that learns to play Tic-Tac-Toe through Q-Learning."""

    def __init__(self, player_symbol, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.1):
        self.player = player_symbol
        self.opponent = Board.O if player_symbol == Board.X else Board.X
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.stats = AIStats()
        self.training_history = []

    # ── State / action helpers ──────────────────────────────────────────

    @staticmethod
    def _board_key(board):
        """Convert board grid to a hashable tuple-of-tuples."""
        return tuple(tuple(row) for row in board.grid)

    def _q(self, state, action):
        return self.q_table[(state, action)]

    def _best_action(self, board):
        """Return the action with the highest Q-value for the current state."""
        state = self._board_key(board)
        empty = board.get_empty_cells()
        if not empty:
            return None
        best_val = float('-inf')
        best_move = empty[0]
        for move in empty:
            val = self._q(state, move)
            if val > best_val:
                best_val = val
                best_move = move
        return best_move

    def _epsilon_greedy(self, board):
        """Pick a random move with probability epsilon, else best move."""
        if random.random() < self.epsilon:
            empty = board.get_empty_cells()
            return random.choice(empty) if empty else None
        return self._best_action(board)

    # ── Training ────────────────────────────────────────────────────────

    def train(self, num_episodes=50000, board_size=3, opponent="random",
              decay_epsilon=True):
        """
        Train the agent via self-play or vs a random opponent.

        Returns a list of cumulative reward values (one per episode) for
        plotting training curves.
        """
        rewards_history = []
        cumulative_reward = 0.0

        original_epsilon = self.epsilon

        for episode in range(num_episodes):
            if decay_epsilon:
                # Linear decay from original epsilon to 0.01
                self.epsilon = max(0.01,
                                   original_epsilon * (1 - episode / num_episodes))

            board = Board(board_size)

            # Decide who goes first randomly to diversify experience
            agent_first = random.random() < 0.5
            if agent_first:
                agent_sym, opp_sym = Board.X, Board.O
            else:
                agent_sym, opp_sym = Board.O, Board.X

            # Temporarily set symbols for this episode
            saved_player, saved_opponent = self.player, self.opponent
            self.player, self.opponent = agent_sym, opp_sym

            opp = RandomAgent(opp_sym)
            trajectory = []  # list of (state, action) for the agent

            while not board.is_terminal():
                current = board.current_player()
                if current == agent_sym:
                    state = self._board_key(board)
                    action = self._epsilon_greedy(board)
                    if action is None:
                        break
                    trajectory.append((state, action))
                    board.make_move(action[0], action[1], current)
                else:
                    move = opp.get_move(board)
                    if move:
                        board.make_move(move[0], move[1], current)

            # Determine reward
            winner = board.check_winner()
            if winner == agent_sym:
                reward = 1.0
            elif winner == opp_sym:
                reward = -1.0
            else:
                reward = 0.5  # draw

            cumulative_reward += reward

            # Update Q-values backwards along the trajectory
            future_q = 0.0
            for state, action in reversed(trajectory):
                old_q = self.q_table[(state, action)]
                self.q_table[(state, action)] = (
                    old_q + self.lr * (reward + self.gamma * future_q - old_q)
                )
                future_q = self.q_table[(state, action)]

            rewards_history.append(cumulative_reward)

        # Restore original settings
        self.epsilon = original_epsilon
        self.player, self.opponent = saved_player, saved_opponent
        self.training_history = rewards_history
        return rewards_history

    # ── Inference (same interface as MinimaxAgent / AlphaBetaAgent) ─────

    def get_move(self, board):
        """Select the best learned move (greedy, no exploration)."""
        self.stats.reset()
        start = time.time()
        self.stats.nodes_explored = 1  # table lookup
        move = self._best_action(board)
        self.stats.elapsed_time = time.time() - start
        return move

    # ── Persistence ─────────────────────────────────────────────────────

    def save_qtable(self, path="qtable.json"):
        """Save the Q-table to a JSON file."""
        serializable = {}
        for (state, action), value in self.q_table.items():
            key = json.dumps({"state": state, "action": action})
            serializable[key] = value
        with open(path, "w") as f:
            json.dump(serializable, f)

    def load_qtable(self, path="qtable.json"):
        """Load a Q-table from a JSON file."""
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = defaultdict(float)
        for key, value in data.items():
            parsed = json.loads(key)
            state = tuple(tuple(row) for row in parsed["state"])
            action = tuple(parsed["action"])
            self.q_table[(state, action)] = value
        return True
