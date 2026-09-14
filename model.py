import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, target_update_every=100):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # target network: a frozen copy used only to compute training targets,
        # synced from `model` every `target_update_every` training steps.
        # this stops the network from chasing a target that moves every update.
        self.target_model = Linear_QNet(
            model.linear1.in_features, model.linear1.out_features, model.linear2.out_features
        )
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.target_update_every = target_update_every
        self.train_step_count = 0

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        with torch.no_grad():
            next_q_values = self.target_model(next_state)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_q_values[idx])

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def save_checkpoint(agent, record, scores, mean_scores, total_score, path='./model/checkpoint.pth'):
    os.makedirs('./model', exist_ok=True)
    torch.save({
        'model_state': agent.model.state_dict(),
        'target_model_state': agent.trainer.target_model.state_dict(),
        'optimizer_state': agent.trainer.optimizer.state_dict(),
        'n_games': agent.n_games,
        'memory': list(agent.memory),
        'record': record,
        'scores': scores,
        'mean_scores': mean_scores,
        'total_score': total_score,
    }, path)


def load_checkpoint(agent, path='./model/checkpoint.pth'):
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, weights_only=False)
    agent.model.load_state_dict(checkpoint['model_state'])
    agent.trainer.target_model.load_state_dict(checkpoint['target_model_state'])
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
    agent.n_games = checkpoint['n_games']
    agent.memory.extend(checkpoint['memory'])
    return checkpoint