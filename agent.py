import torch
import random
import threading
import queue
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE, SpeedControl
from model import Linear_QNet, QTrainer, save_checkpoint, load_checkpoint
from dashboard import Dashboard

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CHECKPOINT_EVERY = 25

NUM_EPISODES = 5000


def free_space(game, start):
    """Fraction of the board reachable from `start` (flood fill). Lets the agent see dead ends."""
    # ponytail: treats the whole body as static walls, ignores that the tail moves away
    if game.is_collision(start):
        return 0.0
    blocked = set(game.snake)
    seen = {start}
    stack = [start]
    while stack:
        p = stack.pop()
        for q in (Point(p.x + BLOCK_SIZE, p.y), Point(p.x - BLOCK_SIZE, p.y),
                  Point(p.x, p.y + BLOCK_SIZE), Point(p.x, p.y - BLOCK_SIZE)):
            if q not in seen and q not in blocked and not game.is_collision(q):
                seen.add(q)
                stack.append(q)
    return len(seen) / (game.play_cols * game.play_rows)


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # neighbour cell for each action: straight, right turn, left turn (same order as the danger bits)
        ahead = {Direction.RIGHT: point_r, Direction.DOWN: point_d, Direction.LEFT: point_l, Direction.UP: point_u}
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        moves = [ahead[clock_wise[idx]], ahead[clock_wise[(idx + 1) % 4]], ahead[clock_wise[(idx - 1) % 4]]]

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ] + [free_space(game, p) for p in moves]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def run_training_loop(update_queue, speed_control):
    agent = Agent()
    game = SnakeGameAI(speed_control=speed_control)

    record = 0
    scores = []
    mean_scores = []
    total_score = 0

    checkpoint = load_checkpoint(agent)
    if checkpoint is not None:
        record = checkpoint['record']
        scores = checkpoint['scores']
        mean_scores = checkpoint['mean_scores']
        total_score = checkpoint['total_score']
        print(f"Resumed from checkpoint: game {agent.n_games}, record {record}")
        for i in range(len(scores)):
            update_queue.put((scores[i], i + 1, max(80 - (i + 1), 0), record, len(agent.memory)))

    while agent.n_games < NUM_EPISODES:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)

            new_record = False
            if score > record:
                record = score
                new_record = True
                agent.model.save()

            if new_record or agent.n_games % CHECKPOINT_EVERY == 0:
                save_checkpoint(agent, record, scores, mean_scores, total_score)

            update_queue.put((score, agent.n_games, agent.epsilon, record, len(agent.memory)))
            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    update_queue = queue.Queue()
    speed_control = SpeedControl(initial=60)

    training_thread = threading.Thread(target=run_training_loop, args=(update_queue, speed_control), daemon=True)
    training_thread.start()

    dashboard = Dashboard(update_queue, speed_control=speed_control)
    dashboard.run_forever()