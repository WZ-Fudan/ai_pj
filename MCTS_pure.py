import math
import random
import copy
import numpy as np
from MCTS_alphazero import GoMoku_player

class Node(object):
    def __init__(self, board, pruning=False):
        self.board = board
        self.lord = 'O' if self.board.player == 'X' else 'X'
        self.value = 0
        self.n = 0
        self.parent = None
        self.child = []
        self.is_leaf = True
        self.is_full = False
        self.winner = None
        self.pruning = pruning

    def update_child(self):
        next_moves = self.board.get_valid_position()
        # next_moves = self.board.get_adjacent_positon()
        child_nodes = []
        for action in next_moves:
            board = copy.deepcopy(self.board)
            board.move(action)
            node = Node(board)
            node.parent = self
            node.is_full, node.winner = node.board.is_end()
            if self.pruning:
                node.pruning = True
                if node.winner is not None:
                    child_nodes = [node]
                    break
            child_nodes.append(node)
        self.child = child_nodes
        self.is_leaf = len(child_nodes) == 0


class MCTS(object):
    def __init__(self, root, max_iterations):
        self.root = root
        self.max_iterations = max_iterations
        self.C = 2  # math.sqrt(2)
        self.selected_leaf = None

    def calculate_UCB1(self, node):
        if node.n == 0:
            return 1e5
        return node.value / node.n + self.C * math.sqrt(math.log(self.root.n) / node.n)

    def select(self):
        current_node = self.root
        while not current_node.is_leaf:
            max_ucb1 = -1e5
            for child in current_node.child:
                tmp_ucb1 = self.calculate_UCB1(child)
                if tmp_ucb1 == 1e5:
                    current_node = child
                    break

                elif tmp_ucb1 > max_ucb1:
                    max_ucb1 = tmp_ucb1
                    current_node = child
                else:
                    continue
        self.selected_leaf = current_node

    def expand(self):
        if self.selected_leaf.n > 0:
            if self.selected_leaf.is_full or self.selected_leaf.winner is not None:
                return self.selected_leaf.is_full, self.selected_leaf.winner
            else:
                self.selected_leaf.update_child()
                self.selected_leaf = self.selected_leaf.child[0]
                return None
        return None

    def simulation(self, record):
        if record is not None:
            self.backpropogation(record[1])
        else:
            current_board = copy.deepcopy(self.selected_leaf.board)
            is_full, winner = self.selected_leaf.is_full, self.selected_leaf.winner
            while True:
                if winner is not None:
                    self.backpropogation(winner)
                    break
                if is_full:
                    self.backpropogation()
                    break
                # valid = current_board.get_adjacent_positon()  # get_valid_position()
                valid = current_board.get_valid_position()
                i, j = random.sample(valid, 1)[0]
                current_board.move((i, j))
                is_full, winner = current_board.is_end()

    def backpropogation(self, winner=None):
        current_node = self.selected_leaf
        current_node.n += 1
        # current_node.value += -(2 * int(winner == current_node.lord) + int(winner is None) - 1)
        if winner == current_node.lord:
            current_node.value += 1
        elif winner is None:
            current_node.value += 0
        else:
            current_node.value += -1
        # current_node.value += winner == current_node.lord
        while current_node.parent is not None:
            current_node = current_node.parent
            current_node.n += 1
            # current_node.value += winner == current_node.lord
            if winner == current_node.lord:
                current_node.value += 1
            elif winner is None:
                current_node.value += 0
            else:
                current_node.value += -1

    def execute(self):
        if self.root.board.last_move is None:
            return self.root.board.size // 2, self.root.board.size // 2, self.root.board.player
        for k in range(self.max_iterations):
            self.select()
            record = self.expand()
            self.simulation(record)
            # if k in (100, 500, 1000, 2000, 3000, 4000, 4999):
            #     print(k)
            #     np.zeros((s))
            #     result = sorted(self.root.child, key=lambda x: x.board.last_move[:2])
            #     # child_n = []
            #     for child in result:
            #         # child_n.append(child.n)
            #         print(child.board.last_move, child.n, child.value,
            #               round(self.calculate_UCB1(child), 3))
        # if child.board.last_move == (4, 3, 'X') and child.child:
        #         print("Here are children: ")
        #         for c in child.child:
        #             if c.board.last_move == (2, 1, 'O'):
        #                 print(print(c.board.last_move, c.lord, c.n, c.value))
        result = sorted(self.root.child, key=lambda x: x.n, reverse=True)
        # print(result[0].board.last_move)
        return result[0].board.last_move


class Pure_MCTS_player(GoMoku_player):
    def __init__(self):
        GoMoku_player.__init__(self)

    def get_next_move(self, board):
        root = Node(board)
        root.update_child()
        mcts = MCTS(root, 2000)
        i, j, _ = mcts.execute()
        return (i * board.size + j), None

    def __str__(self):
        return "pure MCTS player " + self.ind
