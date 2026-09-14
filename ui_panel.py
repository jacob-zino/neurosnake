import pygame

WHITE = (240, 240, 245)
GRAY = (130, 130, 140)
PANEL_BG = (24, 24, 28)
CARD_BG = (34, 34, 40)
DIVIDER = (50, 50, 56)
LINE_COLOR = (90, 200, 160)
MEAN_LINE_COLOR = (90, 150, 220)
ACCENT = (230, 110, 70)
ACCENT_BAR = (90, 150, 220)
RECORD_BAR = (230, 110, 70)

try:
    font_title = pygame.font.SysFont('verdana', 20, bold=True)
    font_label = pygame.font.SysFont('verdana', 13)
    font_value = pygame.font.SysFont('verdana', 24, bold=True)
except Exception:
    font_title = pygame.font.SysFont('arial', 20, bold=True)
    font_label = pygame.font.SysFont('arial', 13)
    font_value = pygame.font.SysFont('arial', 24, bold=True)


class StatsPanel:
    def __init__(self, x, y, width, height, max_points=200):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points

        self.scores = []
        self.mean_scores = []
        self.record = 0
        self.n_games = 0
        self.epsilon = 0
        self.total_score = 0

    def update(self, score, n_games, epsilon, record):
        self.scores.append(score)
        self.total_score += score
        self.mean_scores.append(self.total_score / n_games)
        self.n_games = n_games
        self.epsilon = epsilon
        self.record = record

        if len(self.scores) > self.max_points:
            self.scores = self.scores[-self.max_points:]
            self.mean_scores = self.mean_scores[-self.max_points:]

    def draw(self, surface):
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, PANEL_BG, panel_rect)

        pad = 18
        cursor_y = pad

        title = font_title.render("Training stats", True, WHITE)
        surface.blit(title, (self.x + pad, cursor_y))
        cursor_y += 38

        card_w = self.width - 2 * pad
        last_score = self.scores[-1] if self.scores else 0
        mean = round(self.mean_scores[-1], 2) if self.mean_scores else 0

        cursor_y = self._draw_card(surface, self.x + pad, cursor_y, card_w, "GAMES PLAYED", str(self.n_games), ACCENT_BAR)
        cursor_y = self._draw_card(surface, self.x + pad, cursor_y, card_w, "RECORD SCORE", str(self.record), RECORD_BAR)
        cursor_y = self._draw_card(surface, self.x + pad, cursor_y, card_w, "LAST SCORE", str(last_score), ACCENT_BAR)
        cursor_y = self._draw_card(surface, self.x + pad, cursor_y, card_w, "MEAN SCORE", str(mean), ACCENT_BAR)
        cursor_y = self._draw_card(surface, self.x + pad, cursor_y, card_w, "EPSILON", str(max(self.epsilon, 0)), ACCENT_BAR)

        cursor_y += 6
        graph_label = font_label.render("SCORE HISTORY", True, GRAY)
        surface.blit(graph_label, (self.x + pad, cursor_y))
        cursor_y += 22

        graph_h = self.height - cursor_y - pad
        graph_rect = pygame.Rect(self.x + pad, cursor_y, card_w, max(graph_h, 60))
        self._draw_graph(surface, graph_rect)

    def _draw_card(self, surface, x, y, w, label, value, bar_color):
        card_h = 52
        card_rect = pygame.Rect(x, y, w, card_h)
        pygame.draw.rect(surface, CARD_BG, card_rect, border_radius=8)

        bar_rect = pygame.Rect(x, y, 4, card_h)
        pygame.draw.rect(surface, bar_color, bar_rect, border_top_left_radius=8, border_bottom_left_radius=8)

        label_surf = font_label.render(label, True, GRAY)
        value_surf = font_value.render(value, True, WHITE)
        surface.blit(label_surf, (x + 16, y + 8))
        surface.blit(value_surf, (x + 16, y + 24))

        return y + card_h + 10

    def _draw_graph(self, surface, rect):
        pygame.draw.rect(surface, CARD_BG, rect, border_radius=8)

        if len(self.scores) < 2:
            return

        max_val = max(max(self.scores), 1)
        inset = 10

        def to_points(data):
            n = len(data)
            points = []
            for i, v in enumerate(data):
                px = rect.x + inset + (i / (n - 1)) * (rect.width - 2 * inset)
                py = rect.y + rect.height - inset - (v / max_val) * (rect.height - 2 * inset)
                points.append((px, py))
            return points

        pygame.draw.lines(surface, LINE_COLOR, False, to_points(self.scores), 2)
        pygame.draw.lines(surface, MEAN_LINE_COLOR, False, to_points(self.mean_scores), 2)

        max_label = font_label.render(str(max_val), True, GRAY)
        surface.blit(max_label, (rect.x + 8, rect.y + 6))