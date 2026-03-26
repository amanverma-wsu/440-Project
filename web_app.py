"""Flask web app for Tic-Tac-Toe AI."""

from flask import Flask, render_template, jsonify, request
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent
from heuristic import evaluate_board
from nn_heuristic import make_nn_heuristic

app = Flask(__name__, template_folder="templates", static_folder="static")

# Store game state in memory (single-player app)
games = {}


def _create_ai(ai_symbol, board_size, algorithm):
    if algorithm == "nn":
        nn_heuristic = make_nn_heuristic(board_size)
        depth = None if board_size <= 3 else (4 if board_size == 4 else 2)
        return AlphaBetaAgent(ai_symbol, depth_limit=depth, heuristic=nn_heuristic)
    if board_size <= 3:
        if algorithm == "minimax":
            return MinimaxAgent(ai_symbol)
        return AlphaBetaAgent(ai_symbol)
    else:
        depth = 6 if board_size == 4 else 4
        if algorithm == "minimax":
            return MinimaxAgent(ai_symbol, depth_limit=depth, heuristic=evaluate_board)
        return AlphaBetaAgent(ai_symbol, depth_limit=depth, heuristic=evaluate_board)


def _board_to_list(board):
    return [[board.grid[r][c] for c in range(board.size)] for r in range(board.size)]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    data = request.json
    size = data.get("size", 3)
    algorithm = data.get("algorithm", "alphabeta")
    human = data.get("human", "X")

    board = Board(size)
    ai_symbol = Board.O if human == "X" else Board.X
    ai = _create_ai(ai_symbol, size, algorithm)

    game_id = "game"
    games[game_id] = {"board": board, "ai": ai, "human": human, "ai_symbol": ai_symbol}

    result = {
        "grid": _board_to_list(board),
        "size": size,
        "current": board.current_player(),
        "human": human,
        "winner": None,
        "draw": False,
        "ai_move": None,
        "stats": None,
    }

    # If AI goes first, play center immediately (optimal opening)
    if board.current_player() == ai_symbol:
        center = size // 2
        move = (center, center)
        board.make_move(move[0], move[1], ai_symbol)
        result["grid"] = _board_to_list(board)
        result["current"] = board.current_player()
        result["ai_move"] = [move[0], move[1]]
        result["stats"] = {
            "nodes": ai.stats.nodes_explored,
            "time": round(ai.stats.elapsed_time, 4),
        }

    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.json
    row, col = data["row"], data["col"]

    game = games.get("game")
    if not game:
        return jsonify({"error": "No active game"}), 400

    board = game["board"]
    ai = game["ai"]
    human = game["human"]
    ai_symbol = game["ai_symbol"]

    # Human move
    if not board.make_move(row, col, human):
        return jsonify({"error": "Invalid move"}), 400

    result = {
        "grid": _board_to_list(board),
        "current": board.current_player(),
        "human": human,
        "winner": None,
        "draw": False,
        "ai_move": None,
        "stats": None,
        "winning_line": None,
    }

    # Check if game over after human move
    winner = board.check_winner()
    if winner:
        result["winner"] = winner
        result["winning_line"] = _find_winning_line(board, winner)
        return jsonify(result)
    if board.is_full():
        result["draw"] = True
        return jsonify(result)

    # AI move
    move = ai.get_move(board)
    if move:
        board.make_move(move[0], move[1], ai_symbol)
        result["grid"] = _board_to_list(board)
        result["current"] = board.current_player()
        result["ai_move"] = [move[0], move[1]]
        result["stats"] = {
            "nodes": ai.stats.nodes_explored,
            "time": round(ai.stats.elapsed_time, 4),
        }

    # Check if game over after AI move
    winner = board.check_winner()
    if winner:
        result["winner"] = winner
        result["winning_line"] = _find_winning_line(board, winner)
    elif board.is_full():
        result["draw"] = True

    return jsonify(result)


def _find_winning_line(board, symbol):
    n = board.size
    for r in range(n):
        if all(board.grid[r][c] == symbol for c in range(n)):
            return [[r, c] for c in range(n)]
    for c in range(n):
        if all(board.grid[r][c] == symbol for r in range(n)):
            return [[r, c] for r in range(n)]
    if all(board.grid[i][i] == symbol for i in range(n)):
        return [[i, i] for i in range(n)]
    if all(board.grid[i][n - 1 - i] == symbol for i in range(n)):
        return [[i, n - 1 - i] for i in range(n)]
    return None


if __name__ == "__main__":
    print("\n  Open http://localhost:8080 in your browser\n")
    app.run(debug=False, port=8080)
