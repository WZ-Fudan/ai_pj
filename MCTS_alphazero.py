import numpy as np
import torch
from typing import Tuple, Any, Dict, List
# from game import Board
from board import Board
import copy

def softmax(x: np.ndarray):
    prob = np.exp(x - x.max())
    return prob / prob.sum()

class TreeNode:
    """
    Node in the MCT

    member variable:
    parent: parent of the node
    P: prior priority of the node
    Q: Q value of the node
    U: U value of the node
    children: children node of the node
    n_visit: visit time of the node

    function:
    select: select child node with max Q + U value
    expand: expand leaf node
    backup: update Q value
    recursive_backup: recursive update Q vlaue
    get_value: get Q + U value

    note:
    1. board is not stored in the MCT
    """

    def __init__(self, parent, prior_probability: float):
        self.parent: TreeNode = parent
        self.P: float = prior_probability
        self.Q: float = 0.0
        self.children: Dict[int, TreeNode] = {}
        self.n_visit: int = 0

    def select(self, c_puct: float):
        return max(self.children.items(), key=lambda action_node: action_node[1].get_value(c_puct))

    def expand(self, action_prior: List[Tuple[int, float]]):
        for action, prior in action_prior:
            self.children[action] = TreeNode(self, prior)

    def backup(self, _Q: float):
        self.n_visit += 1
        self.Q = self.Q + (_Q - self.Q) / self.n_visit

    def recursive_backup(self, _Q: float):
        self.backup(_Q)
        if self.parent:
            self.parent.backup(-_Q)

    def get_value(self, c_puct: float):
        self.U = c_puct * self.P * np.sqrt(self.parent.n_visit) / (1 + self.n_visit)
        return self.Q + self.U

    def is_leaf(self):
        return not self.children

class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, c_puct, n_playout, p_v_function):
        self.c_puct = c_puct
        self.root = TreeNode(None, 1.0)
        self.n_playout = n_playout
        self.p_v_function = p_v_function

    def recursive_select(self, board: Board = None):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            if board:
                board.do_move(action)
        return node

    def play_probability(self, tau=0.0001):
        assert not self.root.is_leaf()
        temp = [(action, node.n_visit) for action, node in self.root.children.items()]
        action, n_visits = zip(*temp)
        prob = softmax(1.0 / tau * np.log(np.array(n_visits) + 1e-10))
        return action, prob

    def move_to_next_state(self, action: int):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def _one_time_play(self, board: Board):
        leaf_node = self.recursive_select(board)

        # action_prior is a list of tuple indicating posibility of next move
        # leaf_value meaning the reward for the next player (board.get_current_player())

        # end, win_player = board.game_end()
        end, win_player = board.is_end()

        if not end:
            action_prior, leaf_value = self.p_v_function(board)
            leaf_node.expand(action_prior)
        else:
            if win_player is None:
                leaf_value = 0
            else:
                # if the next player wins, reward is 1 else -1.0
                leaf_value = 1.0 if win_player == board.get_next_player() else -1.0

        # - (reward of the next player) is the reward for the previous player
        leaf_node.recursive_backup(-leaf_value)

    def build_tree(self, board: Board):
        for index in range(self.n_playout):
            _board = copy.deepcopy(board)
            self._one_time_play(_board)

class GoMoku_player:
    def __init__(self):
        self.ind = ""
        return

    def set_player_ind(self, ind):
        self.ind = ind

    def reset_player(self):
        return

    def get_player_state(self):
        return None

    def change_to_test_mode(self):
        return

    def reset_player_state(self, is_self_play):
        return

    def get_next_move(self, board):
        if board.get_valid_position():
            # position = np.random.choice(list(board.get_valid_position()))
            index = np.random.choice(range(len(board.get_valid_position())))
            position = list(board.get_valid_position())[index]
            return board.position_to_move(*position), None
        else:
            raise Exception("There is no avaliable move")

    def opponent_move(self, move):
        return

    def __str__(self):
        return "random player " + self.ind


class MCTS_player(GoMoku_player):
    def __init__(self, p_v_function, n_playout=2000, c_puct=5, is_self_play=False):
        super().__init__()
        self.mcts = MCTS(c_puct, n_playout, p_v_function)
        self.is_self_play = is_self_play

    def reset_player(self):
        self.mcts.move_to_next_state(-1)

    def get_next_move(self, board: Board, tau=0.0001):
        if board.get_valid_position():
            self.mcts.build_tree(board)
            action, probs = self.mcts.play_probability(tau)
            move_probs = np.zeros(board.size ** 2)
            move_probs[list(action)] = probs
            if self.is_self_play:
                move = np.random.choice(action, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            else:
                move = np.random.choice(action, p=probs)
            self.mcts.move_to_next_state(move)
            return move, move_probs
        else:
            raise Exception("there is no available move")

    def get_player_state(self):
        return self.is_self_play

    def change_to_test_mode(self):
        self.is_self_play = False

    def reset_player_state(self, is_self_play):
        self.is_self_play = is_self_play

    def opponent_move(self, move):
        self.mcts.move_to_next_state(move)

    def __str__(self):
        return "MCTS alphago zero player " + self.ind
