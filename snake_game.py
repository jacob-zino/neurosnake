import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 22)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
BOARD_BORDER = (90, 140, 70)
GRASS_LIGHT = (168, 209, 100)
GRASS_DARK = (156, 200, 88)

SNAKE_BODY = (70, 120, 220)
SNAKE_BODY_EDGE = (55, 100, 195)
SNAKE_EYE_WHITE = (255, 255, 255)

APPLE_RED = (216, 62, 44)
APPLE_STEM = (90, 150, 70)
APPLE_LEAF = (110, 175, 90)

BLOCK_SIZE = 28
BORDER_THICK = 10

GAME_COLS = 22
GAME_ROWS = 16
GAME_W = GAME_COLS * BLOCK_SIZE + 2 * BORDER_THICK
GAME_H = GAME_ROWS * BLOCK_SIZE + 2 * BORDER_THICK


class SpeedControl:
    """Shared holder for the current render speed (fps).
    Tkinter's slider writes to it, pygame's game loop reads from it every frame."""
    def __init__(self, initial=60):
        self.value = initial


class SnakeGameAI:

    def __init__(self, speed_control=None):
        self.w = GAME_W
        self.h = GAME_H
        self.speed_control = speed_control if speed_control is not None else SpeedControl()

        self.play_left = BORDER_THICK
        self.play_top = BORDER_THICK
        self.play_cols = (self.w - 2 * BORDER_THICK) // BLOCK_SIZE
        self.play_rows = (self.h - 2 * BORDER_THICK) // BLOCK_SIZE
        self.play_right = self.play_left + self.play_cols * BLOCK_SIZE
        self.play_bottom = self.play_top + self.play_rows * BLOCK_SIZE

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake DQN')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        start_col = self.play_cols // 2
        start_row = self.play_rows // 2
        self.head = Point(self.play_left + start_col * BLOCK_SIZE, self.play_top + start_row * BLOCK_SIZE)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        body = set(self.snake)
        free = [Point(self.play_left + c * BLOCK_SIZE, self.play_top + r * BLOCK_SIZE)
                for r in range(self.play_rows) for c in range(self.play_cols)]
        self.food = random.choice([p for p in free if p not in body])

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_head = self.head
        old_dist = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            if len(self.snake) >= self.play_cols * self.play_rows:
                self._update_ui()
                self._show_win_screen()
                game_over = True
                return reward, game_over, self.score
            self._place_food()
        else:
            self.snake.pop()
            # reward shaping: small nudge for moving toward/away from food,
            # so every step carries some learning signal, not just eat/die
            new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            reward = 1 if new_dist < old_dist else -1

        self._update_ui()
        self.clock.tick(self.speed_control.value)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < self.play_left or pt.x >= self.play_right or pt.y < self.play_top or pt.y >= self.play_bottom:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _draw_board(self):
        pygame.draw.rect(self.display, BOARD_BORDER, pygame.Rect(0, 0, self.w, self.h))
        inner = pygame.Rect(self.play_left, self.play_top, self.play_right - self.play_left, self.play_bottom - self.play_top)
        pygame.draw.rect(self.display, GRASS_LIGHT, inner)

        for row in range(self.play_rows):
            for col in range(self.play_cols):
                if (row + col) % 2 == 0:
                    rect = pygame.Rect(self.play_left + col * BLOCK_SIZE, self.play_top + row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(self.display, GRASS_DARK, rect)

    def _draw_snake(self):
        n = len(self.snake)
        for i in range(n - 1, -1, -1):
            pt = self.snake[i]
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, SNAKE_BODY, rect)

            if i > 0:
                prev = self.snake[i - 1]
                if prev.x > pt.x:
                    conn = pygame.Rect(pt.x + BLOCK_SIZE - 2, pt.y, 4, BLOCK_SIZE)
                elif prev.x < pt.x:
                    conn = pygame.Rect(pt.x - 2, pt.y, 4, BLOCK_SIZE)
                elif prev.y > pt.y:
                    conn = pygame.Rect(pt.x, pt.y + BLOCK_SIZE - 2, BLOCK_SIZE, 4)
                else:
                    conn = pygame.Rect(pt.x, pt.y - 2, BLOCK_SIZE, 4)
                pygame.draw.rect(self.display, SNAKE_BODY, conn)

        for pt in self.snake:
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, SNAKE_BODY_EDGE, rect, width=1)

        head = self.snake[0]
        cx, cy = head.x + BLOCK_SIZE / 2, head.y + BLOCK_SIZE / 2
        if self.direction == Direction.RIGHT:
            eye_offsets = [(3, -5), (3, 5)]
        elif self.direction == Direction.LEFT:
            eye_offsets = [(-3, -5), (-3, 5)]
        elif self.direction == Direction.UP:
            eye_offsets = [(-5, -3), (5, -3)]
        else:
            eye_offsets = [(-5, 3), (5, 3)]

        s = BLOCK_SIZE / 20  # offsets were designed for 20px cells
        for ox, oy in eye_offsets:
            ex, ey = cx + ox * s, cy + oy * s
            pygame.draw.rect(self.display, SNAKE_EYE_WHITE, pygame.Rect(ex - 3 * s, ey - 3 * s, 6 * s, 6 * s))

    def _draw_apple(self):
        x, y = self.food.x, self.food.y
        pygame.draw.rect(self.display, APPLE_RED, pygame.Rect(x + 2, y + 4, BLOCK_SIZE - 4, BLOCK_SIZE - 6))
        pygame.draw.rect(self.display, APPLE_STEM, pygame.Rect(x + BLOCK_SIZE // 2 - 1, y, 2, 4))
        pygame.draw.rect(self.display, APPLE_LEAF, pygame.Rect(x + BLOCK_SIZE // 2, y + 1, 5, 3))

    def _update_ui(self):
        self._draw_board()
        self._draw_apple()
        self._draw_snake()

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, (14, 12))

        pygame.display.flip()

    def _show_win_screen(self):
        overlay = pygame.Surface((self.w, self.h))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (0, 0))

        big_font = pygame.font.SysFont('arial', 36, bold=True)
        small_font = pygame.font.SysFont('arial', 20)

        win_text = big_font.render("BOARD COMPLETE!", True, (255, 255, 255))
        score_text = small_font.render(f"Score: {self.score}", True, (255, 255, 255))
        hint_text = small_font.render("Continuing training...", True, (200, 200, 200))

        self.display.blit(win_text, win_text.get_rect(center=(self.w / 2, self.h / 2 - 30)))
        self.display.blit(score_text, score_text.get_rect(center=(self.w / 2, self.h / 2 + 10)))
        self.display.blit(hint_text, hint_text.get_rect(center=(self.w / 2, self.h / 2 + 45)))

        pygame.display.flip()

        for _ in range(60):  # ~2s at 30fps, keep pumping events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            self.clock.tick(30)

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)