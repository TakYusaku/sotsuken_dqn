# array[row(y)][column(x)]
# https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16
# learning agent by Q-learning
# DQN only
import gym
import requests
import numpy as np
import csv
import matplotlib.pyplot as pl
import pprint
#import # 敵のデータ

class QL():
    # update Qtables
    def updateQtable(self,env, q_table, observation, action, reward, next_observation,al):
        gamma = 0.99 # time discount rate
        alpha = al # learning rate

        #行動後の状態で得られる最大行動価値　(つまり最も良い行動を選ぶ)
        next_position = env.getStatus_enemy(next_observation)
        next_max_q_value = np.array([max(q_table[next_position[0]]),max(q_table[next_position[1]])])

        # 行動前の状態の行動価値
        position = env.getStatus_enemy(observation)
        q_value = np.array([q_table[position[0],action[0][0]], q_table[position[1],action[1][0]]])

        #  行動価値関数の更新
        q_table[position[0],action[0][0]] = q_value[0] + alpha * (reward[0] + gamma * next_max_q_value[0] - q_value[0])
        q_table[position[1],action[1][0]] = q_value[1] + alpha * (reward[1] + gamma * next_max_q_value[1] - q_value[1])

        return q_table
    """
    # get action (list)  # フィールド外は再計算,重複あり(ok batting)
    def getAction_ob(env, q_table, observation, episode, i+1):
        epsilon = 0.5 * (1 / (episode + 1))
        a = []
        b = False
        for i in range(2):
            if np.random.uniform(0, 1) > epsilon:  # e-greedy low is off
                x = np.argsort(q_table[observation[i]])[::-1]
                b = False
                c = 0
                while b!=True:
                    b, d, ms, next_pos = env.judAc(i+1[i], x[c])
                    c += 1
                a.append([d, ms, next_pos])

            else: # e-greedy low is on
                b = False
                while b!=True:
                    pa = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                    b, d, ms, next_pos = env.judAc(i+1[i], pa)
                a.append([d, ms, next_pos])

        return a  # [int, str, list]

    # get action (list)  # フィールド外は報酬がマイナス(罰金を与える) (ok out of field)
    def getAction_oof(env, q_table, observation, episode, i+1):
        #epsilon = 0.5 * (1 / (episode + 1))
        epsilon = 0.5
        a = []
        b = False
        for i in range(2):
            if np.random.uniform(0, 1) > epsilon:  # e-greedy low is off
                x = np.argsort(q_table[observation[i]])[::-1]
                b, d, ms, next_pos = env.judAc(i+1[i], x[0])
                a.append([d, ms, next_pos])

            else: # e-greedy low is on
                pa = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                b, d, ms, next_pos = env.judAc(i+1[i], pa)
                a.append([d, ms, next_pos])

        return a  # [int, str, list]
    """
    # 重複無し(no batting)
    def getAction(self,env, q_table, observation, episode,type):
        obs = env.getStatus_enemy(observation)
        epsilon = 0.5 * (1 / (episode + 1))
        a = []
        usr = 3
        while True:
            for i in range(2):
                if np.random.uniform(0, 1) > epsilon:  # e-greedy low is off
                    x = np.argsort(q_table[obs[i]])[::-1]
                    if type == "nb":
                        c = 0
                        while True:
                            b, d, ms, next_pos = env.judAc(i+usr, x[c], observation[i])
                            lv = env.show()
                            try:
                                if b and ms=="move" and (lv[next_pos[0]][next_pos[1]] == 5 or lv[next_pos[0]][next_pos[1]] == 1 or lv[next_pos[0]][next_pos[1]] == 2):
                                        c += 1
                                else:
                                    a.append([d, ms, next_pos])
                                    break
                            except:
                                c += 1
                    elif type == "ob":
                        b, d, ms, next_pos = env.judAc(i+usr, x[0], observation[i])
                        a.append([d, ms, next_pos])

                else: # e-greedy low is on
                    while True:
                        pa = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                        b, d, ms, next_pos = env.judAc(i+usr, pa,observation[i])
                        if type == "nb":
                            lv = env.show()
                            try:
                                if b and ms=="move" and (lv[next_pos[0]][next_pos[1]] == 5 or lv[next_pos[0]][next_pos[1]] == 1 or lv[next_pos[0]][next_pos[1]] == 2):
                                    pass
                                else:
                                    a.append([d, ms, next_pos])
                                    break
                            except:
                                pass
                        elif type == "ob":
                            a.append([d, ms, next_pos])
                            break

            if a[0][2] == a[1][2]: # "nb" , "ob" 関わらず味方同士の行き先がかぶっていたら再計算
                a = []
            else:
                break

        return a  # [int, str, list]
