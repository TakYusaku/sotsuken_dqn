import gym
import requests
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import deque
import time
import threading
import datetime
from statistics import mean
##### library
import learningMethod.Q_Learning as Q
import learningMethod.MonteCarloMethod as M
import tool.tools as ts
import learningMethod.DQN as dqn
#import linenotify
import sys
import traceback
import pprint

class Enemy():
    def __init__(self,types):
        self.type = types
        if types:
            self.enemy = M.MCM()
            self.name = "q_table_MCM.csv"
        else:
            self.enemy = Q.QL()
            self.name = "q_table_QL.csv"
    
    def _second_(self,fm):
        q_table = ts.readQtable(self.name)
        # read q tables from csv file
        ts.writeQtable(fm, self.name, q_table, 0)
        return q_table
    
    def reset(self,terns):
        self.enemy.reset(terns)

    def getActions(self,env, q_table, observation, episode,type_e):
        if self.type:
            action = self.enemy.getAction(env, q_table, observation, episode,type_e)
        else:
            action = self.enemy.getAction(env, q_table, observation, episode,type_e)
        return action
    
    def process(self,env,action,turn,observation,others):
        next_observation, reward = env.step(action,turn,2) # 行動の実行
        r_other = []
        if not len(others) and self.type:
            obs = env.getStatus_enemy(observation)
            self.enemy.memory1.add((obs[0], action[0], reward[0])) # (ob_e[0], action[0], reward[0])
            self.enemy.memory2.add((obs[1], action[1], reward[1]))
        else:
            q_tables = self.enemy.updateQtable(env, others[0], observation, action, reward, next_observation,others[1]) # Qテーブルの更新
            # q_table = Q.updateQtable(env, q_table, observation[0], action, reward, next_observation,al_q)
            r_other.append(q_tables)
        return next_observation, reward, r_other
    
    def update(self, q_table, al_r,):
        q_table = self.enemy.updateQtable(q_table,self.enemy.memory1,al_r)
        q_table = self.enemy.updateQtables(q_table,self.enemy.memory2,al_r)
        return q_table
    
    def writeQtable(self,fm, q_table,episode):
        ts.writeQtable(fm, self.name, q_table, episode)


            
# [] main processing
if __name__ == '__main__':
### -------- 開始処理 --------    
    # ハイパーパラメータの参照
    hypala_name = sys.argv[1] 
    hypala = './hyperpalam/' + hypala_name
    info,palam_cnn,palam_dense = ts.readLParam(hypala)
    # 開始時間の記録
    fm,le_start = ts.getTime("filename")
    ts.init_func(fm)
    ts.Log(fm, "start")
    # 学習パラメータ等の記録
    ts.Log(fm, "info",[info,palam_cnn,palam_dense])

### -------- 学習パラメータ設定 --------
    # 学習回数
    num_episode = info[1]

    # 学習率 _q is q learning, _m is mcm    #############################
    #al_q = info[4]
    #al_m = info[5]
    """
    # 学習タイプの選択
    type_f = info[6]
    """
    type_e = 'nb'
    al_r = 0.01


### -------- 結果保存 --------
    save_episodereward1 = []
    save_episodereward2 = []
    save_avg_totalrewardF = []
    save_sum_totalrewardF = []

    avg_save_episodereward1 = []
    avg_save_episodereward2 = []
    avg_save_avg_totalrewardF = []
    avg_save_sum_totalrewardF = []

    s = [[],[],[],[],[],[]] #[[friend_tile],[friend_field],[friend_total],[enemy_tile],[enemy_field],[enemy_total]] 獲得ポイント
    s_avg = [[],[],[],[],[],[]] # sの平均
    epi_processtime = []
    kari_epi = 0

    e_rr = [] # 敵の各エピソードの報酬
    e_rr_avg = [] # 敵の各エピソードの報酬 avg

    # 勝利数 win1 is ql , win2 is mcm
    Win1 = 0
    Win2 = 0

### -------- 学習環境の作成 --------
    # 学習プラットフォームの選択
    env = gym.make('procon18env_DQN-v0')

### -------- 敵の設定 --------
    enemies = Enemy(info[8])
    q_table = enemies._second_(fm)

### -------- init DQN --------
    DQN_mode = 1
    image_row = 11
    image_column = 8
    channels = info[2]
    batch_size = info[5]
    action_d = info[7]
    init_er_memory = 200000
    fl_memory = 40
    info_dqn = [image_row,image_column,channels,batch_size,action_d,info[3],info[4]]
    main_n = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    target_n = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    memory_state = dqn.ER_Memory(max_size=init_er_memory)
    memory_flame1 = dqn.History_Memory(max_size=4)
    #memory_flame2 = dqn.History_Memory(max_size=4)
    actor = dqn.Actor(120000,init_er_memory)

    #try:
    for episode in range(1):#num_episode):
        # now epoch　の記録
        kari_epi += 1
        # 環境のリセット
        observation, terns = env.reset(info[0])
        # 1試合の報酬のリセット
        total_reward_e = 0
        episode_reward_1 = 0
        episode_reward_2 = 0
        avg_total_reward_f = 0
        sum_total_reward_f = 0

        # 行動決定のNetwork と 価値計算のNetworkを統一
        target_n.model.set_weights(main_n.model.get_weights())

        # 敵がMCMなら実行
        if info[8]:
            enemies.reset(terns)
        
        fs,epi_starttime = ts.getTime("timestamp_s")
        m = "epoch : " + str(episode+1) + " / " + str(num_episode)
        print(m)

        ## 例外発生(try-expectのテスト)
        #if episode == 6:
        #    raise Exception

        # 状態の取得
        ob_f = env.getStatus_enemy(observation[0])
        ob_e = env.getStatus_enemy(observation[1])
        
        POINTFIELD = []
        # 状態の取得 dqn
        p_field,uf_field,ue_field = env.getStatus_dqn(0)
        POINTFIELD = p_field

        user_field = [uf_field,ue_field]
        state_f,memory_flame1 = dqn.getState(env,0,POINTFIELD,user_field,memory_flame1,info[4],observation,ob_f,ob_e)

        ##### main ruetine #####
        for i in range(1): #terns):
            #ts.saveField(env, fm, episode, i)
            env.countStep() # epoch num のカウント

            if not info[4]:
                state_1 = np.array(state_f[0])
                state_2 = np.array(state_f[1])
            else:
                state_1 = [np.array([state_f[0][0]]),np.array([state_f[0][1]])]
                state_2 = [np.array([state_f[1][0]]),np.array([state_f[1][1]])]

            # 行動の取得 dqn
            on_1,coor_1,action_1,ac_1,dir_1 = actor.get_action(env, 1, state_1, main_n, episode)
            # on_ is "OK" or "NO" or "HOLD", coor_ is coordinate, action_ is action_number, ac_ is "rm" or "mv" or "st", dir_ is direction_num
            on_2,coor_2,action_2,ac_2,dir_2 = actor.get_action(env, 2, state_2, main_n, episode)
            # 行動の取得 敵
            action_e = enemies.getActions(env, q_table, observation[1], episode,type_e) # array
            if on_1 == "NO":
                coor_1 = observation[0][0]
                ac_1 = "st"
                dir_1 = 4
            elif on_2 == "NO":
                coor_2 = observation[0][1]
                ac_2 = "st"
                dir_2 = 4

            if coor_1 == action_e[0][2]: # 異動先がかぶる
                action_e[0][1] = 'stay'
                action_e[0][2] = observation[1][0]
                coor_1 = observation[0][0]
                action_1 = 4
                on_1 = "STAY"
                if on_1 == "HOLD":
                    on_1 = "STAY"
            else: # 異動先が被らなかった 
                if on_1 == "HOLD":
                    on_1 = "OK"
            if coor_2 == action_e[1][2]:
                action_e[1][1] = 'stay'
                action_e[1][2] = observation[1][1]
                coor_2 = observation[0][1]
                action_2 = 4
                on_2 = "STAY"
                if on_2 == "HOLD":
                    on_2 = "STAY"
            else: # 異動先が被らなかった
                if on_2 == "HOLD":
                    on_2 = "OK"

            p_pnt = env.calcPoint()

            actions = [[ac_1,dir_1,on_1],[ac_2,dir_2,on_2]]
            next_observation_f = env.step_dqn(actions)

            if not info[8]:
                others = [q_table, al_r]
                next_observation_e, reward_e, r_others = enemies.process(env,action_e,terns,observation[1],others)
                q_table = r_others[0]
            else:
                others = []
                next_observation_e, reward_e, others = enemies.process(env,action_e,terns,observation[1],others)
           
            action_f = [action_1,action_2]

            reward_1 = env.reward_dqn(on_1,p_pnt,POINTFIELD,next_observation_f[0],observation[0][0])
            reward_2 = env.reward_dqn(on_2,p_pnt,POINTFIELD,next_observation_f[1],observation[0][1])
            reward_f = [reward_1,reward_2]

            next_observation = [next_observation_f,next_observation_e]

            # 新状態の取得
            next_ob_f = env.getStatus_enemy(next_observation[0])
            next_ob_e = env.getStatus_enemy(next_observation[1])

            # 新状態の取得 dqn
            p_field,uf_field,ue_field = env.getStatus_dqn(i+1)

            next_user_field = [uf_field,ue_field]
            next_state_f,memory_flame1 = dqn.getState(env,i+1,POINTFIELD,next_user_field,memory_flame1,info[4],next_observation,next_ob_f,next_ob_e)

            memory_state.add((state_f, action_f, reward_f, next_state_f))
            if episode*40 >= init_er_memory:
                main_n.fitting(memory_state, gamma, target_n)
            
            if DQN_mode:
                target_n.model.set_weights(main_n.model.get_weights())
                

            state_f = next_state_f
            ob_f = next_ob_f
            ob_e = next_ob_e

            # 報酬の記録 dqn
            episode_reward_1 += reward_1
            episode_reward_2 += reward_2
            avg_total_reward_f += (reward_1 + reward_2) / 2
            sum_total_reward_f += reward_1 + reward_2

            total_reward_e += (reward_e[0] + reward_e[1]) / 2

            if info[8] and env.terns == env.now_terns:
                # update q_table_Enemy
                q_table = enemies.update(q_table,al_r)
                break


        epi_time_delta,fs,now = ts.getTime("timestamp_on",epi_starttime) # 1epoch 実行時間
        epi_processtime.append(epi_time_delta) # 実行時間の記録
        # 1 epoch の報酬の記録
        save_episodereward1.append(episode_reward_1)
        save_episodereward2.append(episode_reward_2)
        save_avg_totalrewardF.append(avg_total_reward_f)
        save_sum_totalrewardF.append(sum_total_reward_f)

        avg_save_episodereward1.append(mean(save_episodereward1))
        avg_save_episodereward2.append(mean(save_episodereward2))
        avg_save_avg_totalrewardF.append(mean(save_avg_totalrewardF))
        avg_save_sum_totalrewardF.append(mean(save_avg_totalrewardF))

        e_rr.append(total_reward_e)
        e_rr_avg.append(mean(e_rr))

        # 1ゲームのポイントの記録
        s_p = env.calcPoint()
        s[0].append(s_p[0])
        s_avg[0].append(mean(s[0]))
        s[1].append(s_p[1])
        s_avg[1].append(mean(s[1]))
        s[2].append(s_p[2])
        s_avg[2].append(mean(s[2]))
        s[3].append(s_p[3])
        s_avg[3].append(mean(s[3]))
        s[4].append(s_p[4])
        s_avg[4].append(mean(s[4]))
        s[5].append(s_p[5])
        s_avg[5].append(mean(s[5]))

        if env.judVoL() == "Win_1":
            Win1 += 1
            print('agent1 won')
        else:
            Win2 += 1
            print('agent2 won')


        if episode != 0 and episode%250 == 0 and episode!=num_episode-1 : # episode%250 == 0
            enemies.writeQtable(fm,q_table, episode+1)
            info_epoch = [epi_processtime[episode],float(Win1/episode),float(Win2/episode),mean(avg_save_avg_totalrewardF),mean(avg_save_sum_totalrewardF)]
            ts.Log(fm,"now learning",info_epoch,episode+1)
            result = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF,avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF,e_rr,e_rr_avg,s,s_avg]
            ts.saveImage(fm,s,f_rr,e_rr,s_avg,f_rr_avg,e_rr_avg,episode)
        """
        # 学習終了後の後処理
        le_delta,fs,now = ts.getTime("timestamp_on",le_start) # 総実行時間の記録
        ts.writeQtable(fm,info[8], q_table, num_episode)
        ts.writeQtable(fm,info[9], q_table_Enemy, num_episode)
        print("How many times did QL win, and What is WPCT of QL ?")
        w1 = str(Win1) + " , " + str(float(Win1/num_episode))
        print(w1)
        print("How many times did MCM win, and What is WPCT of MCM ?")
        w2 = str(Win2) + " , " + str(float(Win2/num_episode))
        print(w2)
        m = "finished time is " + str(now)
        print(m)

        info_finished = [Win1,Win2,float(Win1/num_episode),float(Win2/num_episode),mean(f_rr),mean(e_rr),fs,le_delta]
        ts.Log(fm,"finished",info_finished)
        ts.saveImage(fm,s,f_rr,e_rr,s_avg,f_rr_avg,e_rr_avg,num_episode)
        """

        #except:
        """
        m = str(sys.exc_info())
        le_delta,fs,now = ts.getTime("timestamp_on",le_start) # 総実行時間の記録
        info_error = [Win1,Win2,float(Win1/kari_epi),float(Win2/kari_epi),mean(f_rr),mean(e_rr),fs,le_delta,m]
        ts.Log(fm,"error",info_error)
        print(m)
        ###
        fn = './log/' + fm
        with open(fn, 'a') as f:
            traceback.print_exc(file=f)
        ###
        ts.writeQtable(fm, info[8],q_table,kari_epi)
        ts.writeQtable(fm, info[9],q_table_Enemy,kari_epi)
        print("How many times did QL win, and What is WPCT of QL ?")
        w1 = str(Win1) + " , " + str(float(Win1/kari_epi))
        print(w1)
        print("How many times did MCM win, and What is WPCT of MCM ?")
        w2 = str(Win2) + " , " + str(float(Win2/kari_epi))
        print(w2)
        m = "finished time is " + str(now)
        print(m)
        ts.saveImage(fm,s,f_rr,e_rr,s_avg,f_rr_avg,e_rr_avg,kari_epi)
        """