"""Pygame GUI for n×n Tic-Tac-Toe with AI opponents."""

import sys
import threading
import pygame
from board import Board
from ai import MinimaxAgent, AlphaBetaAgent, RandomAgent
from heuristic import evaluate_board
from qlearning import QLearningAgent
from nn_heuristic import make_nn_heuristic

# Colors
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
BG_COLOR = (240, 240, 240)
GRID_COLOR = (60, 60, 60)
X_COLOR = (231, 76, 60)
O_COLOR = (52, 152, 219)
HOVER_COLOR = (200, 200, 200)
WIN_COLOR = (46, 204, 113)
DRAW_COLOR = (241, 196, 15)
BUTTON_COLOR = (52, 73, 94)
BUTTON_HOVER = (44, 62, 80)
BUTTON_TEXT = (255, 255, 255)
STATUS_BG = (44, 62, 80)
MENU_BG = (236, 240, 241)

# Layout
CELL_SIZE = 120
LINE_WIDTH = 4
SYMBOL_WIDTH = 6
MARGIN = 8
STATUS_HEIGHT = 60
MENU_WIDTH = 500
MENU_HEIGHT = 550


class TicTacToeGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tic-Tac-Toe AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_title = pygame.font.SysFont("Arial", 42, bold=True)

        # Game state
        self.board = None
        self.ai = None
        self.human_symbol = Board.X
        self.ai_symbol = Board.O
        self.algorithm = "alphabeta"
        self.board_size = 3
        self.game_active = False
        self.ai_thinking = False
        self.ai_stats = ""
        self.winner = None
        self.winning_line = None
        self.hover_cell = None

        # Menu state
        self.in_menu = True
        self.menu_board_size = 3
        self.menu_algorithm = 1  # 0=minimax, 1=alphabeta, 2=qlearning, 3=nn
        self.menu_player = 0  # 0=X, 1=O

        # Set up menu screen
        self.screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

    def _create_ai(self):
        algos = ["minimax", "alphabeta", "qlearning", "nn"]
        algo = algos[self.menu_algorithm]
        self.algorithm = algo
        size = self.board_size

        if algo == "qlearning":
            agent = QLearningAgent(self.ai_symbol)
            if not agent.load_qtable("qtable.json"):
                agent.train(num_episodes=50000, board_size=size)
                agent.save_qtable("qtable.json")
            return agent

        if algo == "nn":
            nn_heuristic = make_nn_heuristic(size)
            depth = None if size <= 3 else (6 if size == 4 else 4)
            return AlphaBetaAgent(self.ai_symbol, depth_limit=depth, heuristic=nn_heuristic)

        if size <= 3:
            if algo == "minimax":
                return MinimaxAgent(self.ai_symbol)
            return AlphaBetaAgent(self.ai_symbol)
        else:
            depth = 6 if size == 4 else 4
            if algo == "minimax":
                return MinimaxAgent(self.ai_symbol, depth_limit=depth, heuristic=evaluate_board)
            return AlphaBetaAgent(self.ai_symbol, depth_limit=depth, heuristic=evaluate_board)

    def _start_game(self):
        self.board_size = self.menu_board_size
        self.human_symbol = Board.X if self.menu_player == 0 else Board.O
        self.ai_symbol = Board.O if self.menu_player == 0 else Board.X
        self.board = Board(self.board_size)
        self.ai = self._create_ai()
        self.game_active = True
        self.ai_thinking = False
        self.ai_stats = ""
        self.winner = None
        self.winning_line = None
        self.hover_cell = None
        self.in_menu = False

        board_px = self.board_size * CELL_SIZE + MARGIN * 2
        self.screen = pygame.display.set_mode((board_px, board_px + STATUS_HEIGHT))

        # If AI goes first, trigger AI move
        if self.board.current_player() == self.ai_symbol:
            self._trigger_ai_move()

    def _get_board_origin(self):
        return MARGIN, MARGIN

    def _cell_from_pos(self, pos):
        ox, oy = self._get_board_origin()
        x, y = pos[0] - ox, pos[1] - oy
        board_px = self.board_size * CELL_SIZE
        if 0 <= x < board_px and 0 <= y < board_px:
            col = x // CELL_SIZE
            row = y // CELL_SIZE
            return int(row), int(col)
        return None

    def _trigger_ai_move(self):
        self.ai_thinking = True
        self.ai_stats = "AI thinking..."
        thread = threading.Thread(target=self._ai_move_thread, daemon=True)
        thread.start()

    def _ai_move_thread(self):
        move = self.ai.get_move(self.board)
        if move:
            self.board.make_move(move[0], move[1], self.ai_symbol)
            nodes = self.ai.stats.nodes_explored
            t = self.ai.stats.elapsed_time
            self.ai_stats = f"AI: ({move[0]},{move[1]})  nodes={nodes}  {t:.3f}s"
        self.ai_thinking = False
        self._check_game_over()

    def _check_game_over(self):
        w = self.board.check_winner()
        if w:
            self.winner = w
            self.game_active = False
            self.winning_line = self._find_winning_line(w)
        elif self.board.is_full():
            self.winner = "draw"
            self.game_active = False

    def _find_winning_line(self, symbol):
        n = self.board.size
        for r in range(n):
            if all(self.board.grid[r][c] == symbol for c in range(n)):
                return [(r, c) for c in range(n)]
        for c in range(n):
            if all(self.board.grid[r][c] == symbol for r in range(n)):
                return [(r, c) for r in range(n)]
        if all(self.board.grid[i][i] == symbol for i in range(n)):
            return [(i, i) for i in range(n)]
        if all(self.board.grid[i][n - 1 - i] == symbol for i in range(n)):
            return [(i, n - 1 - i) for i in range(n)]
        return None

    def _draw_x(self, surface, cx, cy, size, color=X_COLOR):
        pad = size * 0.22
        pygame.draw.line(surface, color, (cx - size + pad, cy - size + pad),
                         (cx + size - pad, cy + size - pad), SYMBOL_WIDTH)
        pygame.draw.line(surface, color, (cx + size - pad, cy - size + pad),
                         (cx - size + pad, cy + size - pad), SYMBOL_WIDTH)

    def _draw_o(self, surface, cx, cy, size, color=O_COLOR):
        pad = size * 0.22
        pygame.draw.circle(surface, color, (int(cx), int(cy)), int(size - pad), SYMBOL_WIDTH)

    def _draw_board(self):
        self.screen.fill(BG_COLOR)
        ox, oy = self._get_board_origin()
        n = self.board.size
        board_px = n * CELL_SIZE

        # Hover highlight
        if self.hover_cell and self.game_active and not self.ai_thinking:
            r, c = self.hover_cell
            if self.board.is_valid_move(r, c):
                rect = pygame.Rect(ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, HOVER_COLOR, rect)

        # Winning cells highlight
        if self.winning_line:
            for r, c in self.winning_line:
                rect = pygame.Rect(ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (*WIN_COLOR, 80), rect)

        # Grid lines
        for i in range(1, n):
            x = ox + i * CELL_SIZE
            pygame.draw.line(self.screen, GRID_COLOR, (x, oy), (x, oy + board_px), LINE_WIDTH)
        for i in range(1, n):
            y = oy + i * CELL_SIZE
            pygame.draw.line(self.screen, GRID_COLOR, (ox, y), (ox + board_px, y), LINE_WIDTH)

        # Border
        pygame.draw.rect(self.screen, GRID_COLOR, (ox, oy, board_px, board_px), LINE_WIDTH)

        # Symbols
        half = CELL_SIZE // 2
        for r in range(n):
            for c in range(n):
                cx = ox + c * CELL_SIZE + half
                cy = oy + r * CELL_SIZE + half
                cell = self.board.grid[r][c]
                if cell == Board.X:
                    color = X_COLOR
                    if self.winning_line and (r, c) in self.winning_line:
                        color = (192, 57, 43)
                    self._draw_x(self.screen, cx, cy, half, color)
                elif cell == Board.O:
                    color = O_COLOR
                    if self.winning_line and (r, c) in self.winning_line:
                        color = (41, 128, 185)
                    self._draw_o(self.screen, cx, cy, half, color)

        # Status bar
        status_y = oy + board_px + MARGIN
        status_rect = pygame.Rect(0, status_y, self.screen.get_width(), STATUS_HEIGHT)
        pygame.draw.rect(self.screen, STATUS_BG, status_rect)

        if self.winner:
            if self.winner == "draw":
                text = "Draw!"
                color = DRAW_COLOR
            elif self.winner == self.human_symbol:
                text = "You win!"
                color = WIN_COLOR
            else:
                text = "AI wins!"
                color = X_COLOR
            surf = self.font_large.render(text, True, color)
            self.screen.blit(surf, (10, status_y + 12))
            restart = self.font_small.render("Click to play again  |  ESC for menu", True, WHITE)
            self.screen.blit(restart, (self.screen.get_width() - restart.get_width() - 10, status_y + 20))
        elif self.ai_thinking:
            surf = self.font.render("AI is thinking...", True, WHITE)
            self.screen.blit(surf, (10, status_y + 15))
        else:
            turn = "Your turn" if self.board.current_player() == self.human_symbol else "AI turn"
            surf = self.font.render(turn, True, WHITE)
            self.screen.blit(surf, (10, status_y + 5))
            if self.ai_stats:
                stats_surf = self.font_small.render(self.ai_stats, True, (180, 180, 180))
                self.screen.blit(stats_surf, (10, status_y + 32))

    def _draw_menu(self):
        self.screen.fill(MENU_BG)

        # Title
        title = self.font_title.render("Tic-Tac-Toe AI", True, BLACK)
        self.screen.blit(title, (MENU_WIDTH // 2 - title.get_width() // 2, 30))

        subtitle = self.font_small.render("Minimax  |  Alpha-Beta  |  Q-Learning  |  Neural Net", True, (120, 120, 120))
        self.screen.blit(subtitle, (MENU_WIDTH // 2 - subtitle.get_width() // 2, 80))

        y = 130

        # Board size
        self._draw_section_label("Board Size", y)
        y += 35
        for i, size in enumerate([3, 4, 5]):
            selected = self.menu_board_size == size
            label = f"{size}x{size}"
            self.menu_size_buttons[i] = self._draw_option_button(label, 50 + i * 155, y, 140, 40, selected)
        y += 65

        # Algorithm
        self._draw_section_label("AI Algorithm", y)
        y += 35
        algo_labels = ["Minimax", "Alpha-Beta", "Q-Learning", "Neural Net"]
        for i, label in enumerate(algo_labels):
            selected = self.menu_algorithm == i
            col = i % 2
            row = i // 2
            bx = 50 + col * 210
            by = y + row * 50
            self.menu_algo_buttons[i] = self._draw_option_button(label, bx, by, 195, 40, selected)
        y += 120

        # Player choice
        self._draw_section_label("Play As", y)
        y += 35
        for i, label in enumerate(["X (First)", "O (Second)"]):
            selected = self.menu_player == i
            self.menu_player_buttons[i] = self._draw_option_button(label, 50 + i * 210, y, 195, 40, selected)
        y += 65

        # Warning for Q-Learning + non-3x3
        if self.menu_algorithm == 2 and self.menu_board_size != 3:
            warn = self.font_small.render("Q-Learning only supports 3x3", True, X_COLOR)
            self.screen.blit(warn, (MENU_WIDTH // 2 - warn.get_width() // 2, y))
            y += 30

        # Start button
        y += 10
        self.start_button = self._draw_start_button("Start Game", MENU_WIDTH // 2 - 100, y, 200, 50)

    def _draw_section_label(self, text, y):
        surf = self.font.render(text, True, BLACK)
        self.screen.blit(surf, (50, y))

    def _draw_option_button(self, text, x, y, w, h, selected):
        rect = pygame.Rect(x, y, w, h)
        mouse = pygame.mouse.get_pos()
        hovering = rect.collidepoint(mouse)

        if selected:
            pygame.draw.rect(self.screen, BUTTON_COLOR, rect, border_radius=6)
            color = WHITE
        elif hovering:
            pygame.draw.rect(self.screen, (180, 180, 180), rect, border_radius=6)
            color = BLACK
        else:
            pygame.draw.rect(self.screen, (210, 210, 210), rect, border_radius=6)
            color = BLACK

        surf = self.font_small.render(text, True, color)
        self.screen.blit(surf, (x + w // 2 - surf.get_width() // 2, y + h // 2 - surf.get_height() // 2))
        return rect

    def _draw_start_button(self, text, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        mouse = pygame.mouse.get_pos()
        hovering = rect.collidepoint(mouse)
        color = BUTTON_HOVER if hovering else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        surf = self.font.render(text, True, WHITE)
        self.screen.blit(surf, (x + w // 2 - surf.get_width() // 2, y + h // 2 - surf.get_height() // 2))
        return rect

    def run(self):
        self.menu_size_buttons = [None, None, None]
        self.menu_algo_buttons = [None, None, None, None]
        self.menu_player_buttons = [None, None]
        self.start_button = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if not self.in_menu:
                            self.in_menu = True
                            self.screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

                elif event.type == pygame.MOUSEMOTION:
                    if not self.in_menu and self.board:
                        self.hover_cell = self._cell_from_pos(event.pos)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.in_menu:
                        pos = event.pos
                        for i in range(3):
                            if self.menu_size_buttons[i] and self.menu_size_buttons[i].collidepoint(pos):
                                self.menu_board_size = i + 3
                        for i in range(4):
                            if self.menu_algo_buttons[i] and self.menu_algo_buttons[i].collidepoint(pos):
                                self.menu_algorithm = i
                        for i in range(2):
                            if self.menu_player_buttons[i] and self.menu_player_buttons[i].collidepoint(pos):
                                self.menu_player = i
                        if self.start_button and self.start_button.collidepoint(pos):
                            if self.menu_algorithm == 2 and self.menu_board_size != 3:
                                pass  # Don't start with invalid config
                            else:
                                self._start_game()

                    elif self.winner is not None:
                        # Click after game over
                        cell = self._cell_from_pos(event.pos)
                        if cell:
                            self._start_game()
                        # Check if ESC area clicked
                        board_px = self.board_size * CELL_SIZE + MARGIN * 2
                        if event.pos[1] > board_px:
                            self.in_menu = True
                            self.screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

                    elif self.game_active and not self.ai_thinking:
                        cell = self._cell_from_pos(event.pos)
                        if cell:
                            r, c = cell
                            if self.board.current_player() == self.human_symbol:
                                if self.board.make_move(r, c, self.human_symbol):
                                    self._check_game_over()
                                    if self.game_active:
                                        self._trigger_ai_move()

            # Draw
            if self.in_menu:
                self._draw_menu()
            elif self.board:
                self._draw_board()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()
