import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reward = pd.read_excel("Qdata.xls")#if use excel then use this as input
#reward = np.array(reward)

#parameters
coefficient=0.95
offset=0
# fake rewards matrix generation function: by using a principle that with big batch size comes with higher model accuracy.
def estimatedFunction(state, action, rewardsSource, rewardsOfLastRound)
    return (state*state)/(rewardsSource*action)*rewardsOfLastRound*coefficient+offset
#this is a case of rewards matrix, as an exmaple in Nili paper, the row means the batch size and the colume means the next batch size choices
reward = np.array([[-1, 0, -1, -1, 0, -1, -1, -1, -1, -1],
                   [0, -1, -1, -1, -1, 0, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, 0, -1, -1, 100],
                   [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
                   [0, -1, -1, -1, -1, 0, -1, -1, -1, -1],
                   [0, 0, -1, -1, 0, -1, 0, 0, -1, -1],
                   [-1, -1, 0, 0, -1, 0, -1, 0, -1, -1],
                   [-1, -1, -1, -1, -1, 0, 0, -1, 0, -1],
                   [-1, -1, -1, -1, -1, -1, -1, 0, -1, 100],
                   [-1, -1, 0, -1, -1, -1, -1, -1, 0, 100]])

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
