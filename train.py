import numpy as np
from MCTS_alphazero import MCTS_player, GoMoku_player
from MCTS_pure import Pure_MCTS_player
from board import Board, Game
from collections import deque
import random
from policy_value_net import Policy_Value_net
from tqdm import trange
import torch

class train_pipeline:

    def __init__(self, board_size=9, n_playout=2000, init_model=None, use_cuda=False):
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.learning_rate = 2e-3
        self.learning_rate_multiplier = 1.0
        self.n_playout = n_playout
        self.c_puct = 1.0
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.game = Game(board_size)
        self.heat_start = 30
        self.evaluation_time = 10

        if init_model:
            self.p_v_net: Policy_Value_net = Policy_Value_net(self.board_size, init_model=init_model, use_cuda=use_cuda)
        else:
            self.p_v_net = Policy_Value_net(self.board_size, use_cuda=use_cuda)
        self.p_v_function = self.p_v_net.p_v_function
        self.mcts_player = MCTS_player(self.p_v_function, self.n_playout, self.c_puct, is_self_play=True)
        self.mcts_pure = Pure_MCTS_player()
        self.random_player = GoMoku_player()

    def data_augumentation(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_size, self.board_size)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def self_play(self, n_times=1, is_shown=False):
        for _ in range(n_times):
            winner, play_data = self.game.start_self_play(self.mcts_player, is_shown=is_shown, temp=1.0)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.data_augumentation(play_data)
            self.data_buffer.extend(play_data)

    def get_learning_rate(self, epoch):
        if epoch > self.heat_start:
            return self.learning_rate
        else:
            return self.learning_rate * epoch / self.heat_start

    def policy_update(self):

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        state_batch = torch.Tensor(state_batch)
        mcts_probs_batch = torch.Tensor(mcts_probs_batch)
        winner_batch = torch.Tensor(winner_batch)

        for i in range(self.epochs):
            loss, entropy = self.p_v_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.get_learning_rate(i))
            print(loss, entropy)

        return loss, entropy

    def policy_evaluate(self, player2, is_shown=False):
        state = self.mcts_player.get_player_state()
        self.mcts_player.change_to_test_mode()

        win_table = np.zeros([self.evaluation_time, 2])
        for _ in range(self.evaluation_time):
            winner = self.game.start_play(self.mcts_player, player2, is_shown=is_shown)
            win_table[_, 0] = int(winner == "X")
        for _ in range(self.evaluation_time):
            winner = self.game.start_play(self.random_player, player2)
            win_table[_, 1] = int(winner == "O")

        self.mcts_player.reset_player_state(state)
        return win_table.mean()


    def train(self, is_shown=False):
        try:
            for i in trange(self.game_batch_num):
                self.self_play(self.play_batch_size, is_shown=is_shown)
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(self.random_player)
                    self.p_v_net.save_model('./current_policy_{}.model'.format(self.board_size))
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!", win_ratio)
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.p_v_net.save_model('./best_policy_{}.model'.format(self.board_size))
                        # if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                        #     self.pure_mcts_playout_num += 1000
                        #     self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == "__main__":
    tp = train_pipeline(n_playout=200, cuda=True)
    tp.train()
