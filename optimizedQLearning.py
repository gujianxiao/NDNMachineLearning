import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List


class QLearner(object):

    def __init__(self, reward_init: np.ndarray, 
                 f_interp: Callable[[np.ndarray, int, int, float], np.ndarray]):
        """
        Init the q-learning with a square-shaped reward array.
        The interpolation function should be specified by the user.
        """
        assert len(reward_init) > 0 and len(reward_init) == len(reward_init[0])

        self.reward = reward_init
        self.f_interp = f_interp

        self.is_real_value = np.zeros(len(reward_init), len(reward_init[0]))    # 0 for false
        self.q_matrix = np.zeros(len(reward_init), len(reward_init[0]))
    
    def get_optimal_action(self, q_ma, start_idx: int) -> int:
        """
        Return the index of the best action on the current q-matrix.
        """
        return np.argmax()

    def add_measurement(self, start_idx: int, end_idx: int, reward_value: float):
        """
        Update the q-learning parameters with the given reward, after an action is performed.
        """
        self.is_real_value[start_idx, end_idx] = 1
        self.reward = self.f_interp(self.reward, start_idx, end_idx, reward_value)
        self._compute_qmatrix()

    def _compute_qmatrix(self) -> np.ndarray:
        """
        Iteratively compute the q-matrix based on the current reward matrix.
        Return the q matrix computed.
        """
        self.q_matrix = np.zeros((len(self.reward), len(self.reward[0])))
        n_rows, n_cols = reward.shape
        print(rows,cols)

        for step in range(0, 500):
            start_state = np.random.randint(0, n_rows)
            for i in range(n_cols):
                if reward[start_state, i] != -1:
                    maxQ = max(self.q_matrix[i])
                    self.q_matrix[start_state, i] = reward[start_state, i] + gamma * maxQ
        self.q_matrix = np.round(self.q_matrix / 5)


if __name__ == "__main__":
    # TODO: Where is this R_max variable used?

    #reward = pd.read_excel("Qdata.xls")#if use excel then use this as input
    #reward = np.array(reward)

    #parameters
    coefficient=0.95
    offset=0
    # fake rewards matrix generation function: by using a principle that with big batch size comes with higher model accuracy.
    def estimatedFunction(state, action, rewardsSource, rewardsOfLastRound):
        return (state*state)/(rewardsSource*action)*rewardsOfLastRound*coefficient+offset
    #this is a case of rewards matrix, as an exmaple in Nili paper, the row means the batch size and the colume means the next batch size choices
    # reward = np.array([[-1, 0, -1, -1, 0, -1, -1, -1, -1, -1],
    #                    [0, -1, -1, -1, -1, 0, -1, -1, -1, -1],
    #                    [-1, -1, -1, -1, -1, -1, 0, -1, -1, 100],
    #                    [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    #                    [0, -1, -1, -1, -1, 0, -1, -1, -1, -1],
    #                    [0, 0, -1, -1, 0, -1, 0, 0, -1, -1],
    #                    [-1, -1, 0, 0, -1, 0, -1, 0, -1, -1],
    #                    [-1, -1, -1, -1, -1, 0, 0, -1, 0, -1],
    #                    [-1, -1, -1, -1, -1, -1, -1, 0, -1, 100],
    #                    [-1, -1, 0, -1, -1, -1, -1, -1, 0, 100]])
    reward = np.array([
        [0, 100, 200, -1],
        [-100, 0, 100, -1],
        [-200, -100, 0, -1],
        [-300, -200, -100, -1]
    ])

    Q_matrix = np.zeros((len(reward), len(reward)))#generate q table as zero
    rows, cols = reward.shape#get the raw and colum of reward matrix
    print(rows,cols)
    steps = 0
    #discound rate
    gamma = 0.8
    while steps < 500:
        steps += 1
        start_state = np.random.randint(0, rows)# randomly choose a state
        Rmax = max(reward[start_state])#get the max r matrix
        for i in range(cols):
            if reward[start_state, i] != -1: #choose the reward!=1 where reward==1 means the condition cannot be satsified
                maxQ = max(Q_matrix[i])#choose the max reward of next action
                #obatain q table and interatively calculate
                Q_matrix[start_state, i] = reward[start_state, i] + gamma * maxQ

    print(np.round(Q_matrix/5))
