import numpy as np
from collections import Counter
from agents.random_agent import RandomAgent
from board import TicTacToeBoard


class TicTacToeEnv:
    def __init__(self):
        self.board = TicTacToeBoard()
        self.feature_vector_size = 28

    def reset(self):
        self.board = TicTacToeBoard()

    def random_position(self):
        self.reset()
        move = np.random.randint(0, 18)
        legal_moves = self.get_legal_moves()
        if move in legal_moves:  # use starting position for moves greater than 8
            self.make_move(move)

    def get_reward(self, board=None):
        if board is None:
            board = self.board
        return board.result()

    def make_move(self, move):
        assert move in self.get_legal_moves()
        self.board.push(move)

    def get_legal_moves(self, board=None):
        if board is None:
            board = self.board
        return board.legal_moves

    def make_feature_vector(self, board):
        fv_size = self.feature_vector_size
        fv = np.zeros((1, fv_size))
        fv[0, :9] = board.xs.reshape(9)
        fv[0, 9:18] = board.os.reshape(9)
        fv[0, 18:27] = ((board.xs + board.os).reshape(9) == 0)
        fv[0, -1] = float(board.turn)
        return fv

    def _print(self, board=None):
        if board is None:
            board = self.board
        s = ''
        for i in range(3):
            s += ' '
            for j in range(3):
                if board.xs[i, j] == 1:
                    s += 'X'
                elif board.os[i, j] == 1:
                    s += 'O'
                else:
                    s += ' '
                if j < 2:
                    s += '|'
            s += '\n'
            if i < 2:
                s += '-------\n'
        print(s)

    def play(self, players, verbose=False):
        while self.get_reward() is None:
            if verbose:
                self._print()
            player = players[int(not self.board.turn)]
            move = player.get_move(self)
            self.make_move(move)

        reward = self.get_reward()
        if verbose:
            self._print()
            if reward == 1:
                print("X won!")
            elif reward == -1:
                print("O won!")
            else:
                print("draw.")
        return reward

    def random_agent_test(self, agent):
        random_agent = RandomAgent()

        x_counter = Counter()
        for _ in range(100):
            self.reset()
            reward = self.play([agent, random_agent])
            x_counter.update([reward])

        o_counter = Counter()
        for _ in range(100):
            self.reset()
            reward = self.play([random_agent, agent])
            o_counter.update([reward])

        results = [x_counter[1], x_counter[0], x_counter[-1],
                   o_counter[-1], o_counter[0], o_counter[1]]

        return results