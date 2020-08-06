import torch
import torch.nn as nn
from board import Board
import numpy as np

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        # action policy layers
        self.act_conv1 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1), nn.ReLU(inplace=True))
        self.act_fc1 = nn.Linear(3 * size ** 2, size ** 2)
        # state value layers
        self.val_conv1 = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1), nn.ReLU(inplace=True))
        self.val_fc1 = nn.Sequential(nn.Linear(2 * size ** 2, 64), nn.ReLU(inplace=True))
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state):
        batch_size = state.shape[0]
        temp1 = self.conv1(state)
        temp2 = self.conv2(temp1)
        temp3 = self.conv3(temp2)
        action = self.act_fc1(self.act_conv1(temp3).view(batch_size, -1))
        value = torch.tanh(self.val_fc2(self.val_fc1(self.val_conv1(temp3).view(batch_size, -1))))
        return action, value

class Policy_Value_net():

    def __init__(self, size, init_model=None, use_cuda=False):
        self.size = size
        self.net = Net(size)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.use_cuda = use_cuda
        if use_cuda:
            self.net = self.net.cuda()
        if init_model:
            net_params = torch.load(init_model)
            self.net.load_state_dict(net_params)

    def p_v_function(self, board: Board):
        state = board.get_state()
        state = torch.Tensor(state).unsqueeze(0)
        if self.use_cuda:
            state = state.cuda()
        action, value = self.net(state)
        action = torch.exp(action.squeeze())
        action_probability = np.zeros(self.size ** 2)
        for position in board.get_valid_position():
            move = position[0] * self.size + position[1]
            action_probability[move] = action[move].detach().item()

        assert action_probability.sum() > 1e-5
        action_probability = action_probability / action_probability.sum()
        action_prior = []
        for position in board.get_valid_position():
            move = position[0] * self.size + position[1]
            action_prior.append((move, action_probability[move]))

        return action_prior, value.item()

    def train_step(self, state_batch, mcts_probility_batch, win_batch, lr):
        if self.use_cuda:
            state_batch = state_batch.cuda()
            mcts_probility_batch = mcts_probility_batch.cuda()
            win_batch = win_batch.cuda()
        set_learning_rate(self.optimizer, lr)
        action, value = self.net(state_batch)
        prob = torch.softmax(action, dim=1)

        value_loss = torch.mean((value - win_batch) ** 2)
        cross_entropy_loss = -torch.mean((mcts_probility_batch * torch.log(prob + 1e-10)).sum(1))

        loss = value_loss + cross_entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(prob * torch.log(prob + 1e-10), 1))
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.net.state_dict()  # get model params
        torch.save(net_params, model_file)
