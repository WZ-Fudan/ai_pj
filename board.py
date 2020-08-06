import numpy as np
import re
from MCTS import *


class Board(object):
    def __init__(self, size):
        self.size = size
        self.board = [['-' for j in range(self.size)] for i in range(self.size)]
        self.valid_position = set((i, j) for i in range(self.size) for j in range(self.size))
        self.players = ['X', 'O']
        self.player = 'X'
        self.history = []
        self.rest_steps = self.size * self.size
        self.X = np.zeros((self.size, self.size))
        self.O = np.zeros((self.size, self.size))
        self.last_move = None

    def move(self, action):
        row, column = action
        self.board[row][column] = self.player
        self.history.append((row, column, self.player))
        self.last_move = (row, column, self.player)
        self.valid_position.remove((row, column))
        if self.player == 'X':
            self.X[row, column] = 1
        else:
            self.O[row, column] = 1
        self.player = 'O' if self.player == 'X' else 'X'

    def do_move(self, action):
        row, column = action // self.size, action % self.size
        self.board[row][column] = self.player
        self.history.append((row, column, self.player))
        self.last_move = (row, column, self.player)
        self.valid_position.remove((row, column))
        if self.player == 'X':
            self.X[row, column] = 1
        else:
            self.O[row, column] = 1
        self.player = 'O' if self.player == 'X' else 'X'

    def position_to_move(self, row, column):
        return row * self.size + column

    def well_print(self):
        lines = [' '.join(self.board[i]) for i in range(self.size)]
        print("\n".join(lines))

    def get_numpy(self):
        return self.X, self.O

    def get_history(self):
        return self.history

    def get_valid_position(self):
        return self.valid_position

    def get_next_player(self):
        return self.player

    def is_end(self):
        if self.last_move is None:
            return False, None
        row, col, player = self.last_move
        prone_lines = [''.join(self.board[row][max(col - 4, 0):min(col + 5, self.size)]),
                       ''.join([self.board[r][col] for r in range(max(row - 4, 0), min(row + 5, self.size))]),
                       ''.join([self.board[row + k][col + k] for k in range(-min(row, col, 4),
                                                                            min(5, self.size - row, self.size - col))]),
                       ''.join([self.board[row - k][col + k] for k in range(-min(self.size - row - 1, col, 4),
                                                                            min(row + 1, self.size - col, 5))])]
        current_player = self.history[-1][-1]
        pattern = re.compile(current_player + "{5}")
        for line in prone_lines:
            if re.search(pattern, line):
                return True, current_player
        return not self.get_valid_position(), None

    def get_state(self):
        state = np.zeros([3, self.size, self.size])
        if self.get_next_player() == "O":
            state[2, :, :] += 1.0
        state[0, :, :] = self.X
        state[1, :, :] = self.O
        return state

class Game(object):
    """game server"""

    def __init__(self, size, **kwargs):
        self.size = size

    def start_play(self, player1, player2, is_shown=1) -> str:
        """start a game between two players"""
        board = Board(self.size)
        p1, p2 = board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        player1.reset_player()
        player2.reset_player()
        players = {p1: player1, p2: player2}
        if is_shown:
            board.well_print()
        step = 1
        while True:
            current_player = board.get_next_player()
            player_in_turn = players[current_player]
            move, _ = player_in_turn.get_next_move(board)
            board.do_move(move)
            players[board.get_next_player()].opponent_move(move)
            assert board.get_next_player() != player_in_turn
            if is_shown:
                print("step", step, "player", str(player_in_turn))
                board.well_print()
            end, winner = board.is_end()
            if end:
                if is_shown:
                    if winner:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        board = Board(self.size)
        p1, p2 = board.players
        player.reset_player()
        states, mcts_probs, current_players = [], [], []
        step = 1
        while True:
            move, move_probs = player.get_next_move(board, tau=temp)
            # store the data
            states.append(board.get_state())
            mcts_probs.append(move_probs)
            current_players.append(board.get_next_player())
            # perform a move
            board.do_move(move)
            if is_shown:
                print("step", step, "player", current_players[-1])
                board.well_print()
            end, winner = board.is_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner:
                    for index, p in enumerate(current_players):
                        if p == winner:
                            winners_z[index] = 1.0
                        else:
                            winners_z[index] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != '-':
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            step += 1

if __name__ == "__main__":
    import numpy as np
    import re
    b = Board(6)
    # b.board[0] = ["X", "X", "X", "X", "-"]
    # b.valid_position = b.valid_position[4:]
    b.well_print()
    k = 0
    while True:
        k += 1
        root = Node(b)
        root.update_child(root.get_child())
        mcts = MCTS(root, 1000)
        i, j = mcts.execute()
        print("AI move{}, {}".format(i, j))
        b.move((i, j))
        b.well_print()
        if b.is_end()[1] == "X":
            print("AI Win!")
            break

        i, j = input("Please enter:").split(',')
        b.move((int(i), int(j)))
        b.well_print()
        print(b.is_end())
        if b.is_end()[1] == "O":
            print("You Win!")
            break
