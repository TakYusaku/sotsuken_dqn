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

    save_1 = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF,avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF]
    save_2 = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF,avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF]

    s = [[],[],[],[],[],[]] #[[friend_tile],[friend_field],[friend_total],[enemy_tile],[enemy_field],[enemy_total]] 獲得ポイント
    s_avg = [[],[],[],[],[],[]] # sの平均
    epi_processtime = []
    kari_epi = 0

    # 勝利数 win1 は DQN1
    Win1 = 0
    Win2 = 0

### -------- 学習環境の作成 --------
    # 学習プラットフォームの選択
    env = gym.make('procon18env_DQN-v0')

### -------- init DQN palam--------
    selfplay = False
    slp = 0
    cnt = 1
    DQN_mode = 0
    image_row = 11
    image_column = 8
    channels = info[2]
    batch_size = info[5]
    action_d = info[7]
    gamma = info[8]
    init_er_memory = 5000 * 40
    fl_memory = 40
    info_dqn = [image_row,image_column,channels,batch_size,action_d,info[3],info[4]]
    total_no_counts = []
    no_counts_one = []
    no_counts_two = []
    no_counts_three = []
    no_counts_four = []
    win_one = 0
    win_two = 0
    tsuyokunatta = 0

### -------- init DQN player 1 --------
    main_n_1 = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    target_n_1 = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    memory_state_1 = dqn.ER_Memory(max_size=init_er_memory)
    memory_flame1_1 = dqn.History_Memory(max_size=4)
    #memory_flame2 = dqn.History_Memory(max_size=4)
    actor_1 = dqn.Actor(120000,500)

### -------- init DQN player2 --------
    main_n_2 = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    target_n_2 = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    memory_state_2 = dqn.ER_Memory(max_size=init_er_memory)
    memory_flame1_2 = dqn.History_Memory(max_size=4)
    #memory_flame2 = dqn.History_Memory(max_size=4)
    actor_2 = dqn.Actor(120000,500)

### -------- init DQN old -------- os is old and strong
    main_os = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
    target_os = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)

    try:
        for episode in range(num_episode):
            # now epoch　の記録
            kari_epi += 1
            # 環境のリセット
            if episode == 0:
                observation, terns = env.reset(info[0])
            else:
                observation, terns = env.reset()
            # 1試合の報酬のリセット
            episode_reward_1 = 0
            episode_reward_2 = 0
            avg_total_reward = 0
            sum_total_reward = 0

            rew_1 = [episode_reward_1,episode_reward_2,avg_total_reward,sum_total_reward]
            rew_2 = [episode_reward_1,episode_reward_2,avg_total_reward,sum_total_reward]

            # 行動決定のNetwork と 価値計算のNetworkを統一
            target_n_1.model.set_weights(main_n_1.model.get_weights())
            target_n_2.model.set_weights(main_n_2.model.get_weights())
            
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
            state,memory_flame1_1,memory_flame1_2 = dqn.getState(env,0,POINTFIELD,user_field,memory_flame1_1,memory_flame1_2,info[4],observation,ob_f,ob_e)

            state_f = [state[0],state[1]]
            state_e = [state[2],state[3]]

            # selfplay 1000回毎に100試合をネットワークだけで対戦させてみる.
            if (episode+1) == (400+1100*cnt):
                m = ''
                if cnt == 1:
                    m += str(cnt) + ' : '
                    memory_flame1_1,memory_flame1_2,memory_state_1,memory_state_2,no_counts_one,no_counts_two,no_counts_three,no_counts_four,win = dqn.selfPlay(fm,env,cnt,main_n_1,main_n_2,target_n_1,target_n_2,memory_flame1_1,memory_flame1_2,memory_state_1,memory_state_2,info[4],actor_1,actor_2,no_counts_one,no_counts_two,no_counts_three,no_counts_four)
                    if win:
                        main_os.model.set_weights(main_n_1.model.get_weights())
                        target_os.model.set_weights(target_n_1.model.get_weights())
                        main_n_2.model.set_weights(main_n_1.model.get_weights())
                        target_n_2.model.set_weights(target_n_1.model.get_weights())
                        m += 'n1_won, os_n1, n1_n1, n2_n1, '
                    else:
                        main_os.model.set_weights(main_n_2.model.get_weights())
                        target_os.model.set_weights(target_n_2.model.get_weights())
                        main_n_1.model.set_weights(main_n_2.model.get_weights())
                        target_n_1.model.set_weights(target_n_2.model.get_weights())
                        m2 += 'n2_won, os_n2, n1_n2, n2_n2'

                else :
                    m += str(cnt) + ' : '
                    if win_one >= win_two:
                        main_new = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
                        target_new = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
                        main_new.model.set_weights(main_n_1.model.get_weights())
                        target_new.model.set_weights(target_n_1.model.get_weights())
                        m += 'n1_new, '

                    else:
                        main_new = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
                        target_new = dqn.DQN("CNN",info_dqn,palam_cnn,palam_dense)
                        main_new.model.set_weights(main_n_2.model.get_weights())
                        target_new.model.set_weights(target_n_2.model.get_weights())
                        m += 'n2_new, '

                    memory_flame1_1,memory_flame1_2,memory_state_1,memory_state_2,no_counts_one,no_counts_two,no_counts_three,no_counts_four,win = dqn.selfPlay(fm,env,cnt,main_new,main_os,target_new,target_os,memory_flame1_1,memory_flame1_2,memory_state_1,memory_state_2,info[4],actor_1,actor_2,no_counts_one,no_counts_two,no_counts_three,no_counts_four)
                    if win:
                        tsuyokunatta += 1
                        main_os.model.set_weights(main_new.model.get_weights())
                        target_os.model.set_weights(target_new.model.get_weights())
                        m += 'new_won, os_new, '                      
                    else:
                        m += 'new_lose, os_stay, '

                    if win_one >= win_two:
                        main_n_2.model.set_weights(main_new.model.get_weights())
                        target_n_2.model.set_weights(target_new.model.get_weights())
                        m += 'n1_n1, n2_n1, '
                    else:
                        main_n_1.model.set_weights(main_new.model.get_weights())
                        target_n_1.model.set_weights(target_new.model.get_weights())
                        m += 'n1_n2, n2_n2, ' 
                m += '\n'
                ts.Log(fm,'spl_f',m)
                win_one = 0
                win_two = 0
                cnt += 1

            
            ##### main ruetine #####
            for i in range(terns):
                env.countStep() # epoch num のカウント

                if not info[4]:
                    state_1 = np.array(state_f[0])
                    state_2 = np.array(state_f[1])
                    state_3 = np.array(state_e[0])
                    state_4 = np.array(state_e[1])
                else:
                    state_1 = [np.array([state_f[0][0]]),np.array([state_f[0][1]])]
                    state_2 = [np.array([state_f[1][0]]),np.array([state_f[1][1]])]
                    state_3 = [np.array([state_e[0][0]]),np.array([state_e[0][1]])]
                    state_4 = [np.array([state_e[1][0]]),np.array([state_e[1][1]])]

                # 行動の取得 dqn
                on_1,coor_1,action_1,ac_1,dir_1 = actor_1.get_action(env, 1, state_1, main_n_1, episode, selfplay)
                # on_ is "OK" or "NO" or "HOLD", coor_ is coordinate, action_ is action_number, ac_ is "rm" or "mv" or "st", dir_ is direction_num
                on_2,coor_2,action_2,ac_2,dir_2 = actor_1.get_action(env, 2, state_2, main_n_1, episode, selfplay)
                on_3,coor_3,action_3,ac_3,dir_3 = actor_2.get_action(env, 3, state_3, main_n_2, episode, selfplay)
                on_4,coor_4,action_4,ac_4,dir_4 = actor_2.get_action(env, 4, state_4, main_n_2, episode, selfplay)

                # 移動先の希望がNoならば,止まる
                if on_1 == "NO" or on_1 == "Error":
                    on_1 = "NO"
                    coor_1 = observation[0][0]
                    ac_1 = "st"
                    dir_1 = 4
                if on_2 == "NO" or on_2 == "Error":
                    on_2 = "NO"
                    coor_2 = observation[0][1]
                    ac_2 = "st"
                    dir_2 = 4
                if on_3 == "NO" or on_3 == "Error":
                    on_3 = "NO"
                    coor_3 = observation[1][0]
                    ac_3 = "st"
                    dir_3 = 4
                if on_4 == "NO" or on_4 == "Error":
                    on_4 = "NO"
                    coor_4 = observation[1][1]
                    ac_4 = "st"
                    dir_4 = 4
             
                # 移動先の希望がかぶった時の処理
                if coor_1 == coor_3: # 異動先がかぶる
                    coor_1 = observation[0][0]
                    action_1 = 4
                    on_1 = "STAY"
                    coor_3 = observation[1][0]
                    action_3 = 4
                    on_3 = "STAY"
                    if on_1 == "HOLD":
                        on_1 = "STAY"
                    if on_3 == "HOLD":
                        on_3 = "STAY"
                elif coor_1 != coor_3: # 異動先が被らなかった 
                    if on_1 == "HOLD":
                        on_1 = "OK"
                    if on_3 == "HOLD":
                        on_3 = "OK"

                if coor_1 == coor_4: # 異動先がかぶる
                    coor_1 = observation[0][0]
                    action_1 = 4
                    on_1 = "STAY"
                    coor_4 = observation[1][1]
                    action_4 = 4
                    on_4 = "STAY"
                    if on_1 == "HOLD":
                        on_1 = "STAY"
                    if on_4 == "HOLD":
                        on_4 = "STAY"
                elif coor_1 != coor_4: # 異動先が被らなかった 
                    if on_1 == "HOLD":
                        on_1 = "OK"
                    if on_4 == "HOLD":
                        on_4 = "OK"

                if coor_2 == coor_3: # 異動先がかぶる
                    coor_2 = observation[0][1]
                    action_2 = 4
                    on_2 = "STAY"
                    coor_3 = observation[1][0]
                    action_3 = 4
                    on_3 = "STAY"
                    if on_2 == "HOLD":
                        on_2 = "STAY"
                    if on_3 == "HOLD":
                        on_3 = "STAY"
                elif coor_2 != coor_3: # 異動先が被らなかった 
                    if on_2 == "HOLD":
                        on_2 = "OK"
                    if on_3 == "HOLD":
                        on_3 = "OK"

                if coor_2 == coor_4: # 異動先がかぶる
                    coor_2 = observation[0][1]
                    action_2 = 4
                    on_2 = "STAY"
                    coor_4 = observation[1][1]
                    action_4 = 4
                    on_4 = "STAY"
                    if on_2 == "HOLD":
                        on_2 = "STAY"
                    if on_4 == "HOLD":
                        on_4 = "STAY"
                elif coor_2 != coor_4: # 異動先が被らなかった 
                    if on_2 == "HOLD":
                        on_2 = "OK"
                    if on_4 == "HOLD":
                        on_4 = "OK"

                if coor_1 == coor_2: # 異動先がかぶる
                    coor_1 = observation[0][0]
                    action_1 = 4
                    on_1 = "STAY"
                    coor_2 = observation[0][1]
                    action_2 = 4
                    on_2 = "STAY"
                    if on_1 == "HOLD":
                        on_1 = "STAY"
                    if on_2 == "HOLD":
                        on_2 = "STAY"
                elif coor_1 != coor_2: # 異動先が被らなかった 
                    if on_1 == "HOLD":
                        on_1 = "OK"
                    if on_2 == "HOLD":
                        on_2 = "OK"              
                
                if coor_3 == coor_4: # 異動先がかぶる
                    coor_3 = observation[1][0]
                    action_3 = 4
                    on_3 = "STAY"
                    coor_4 = observation[1][1]
                    action_4 = 4
                    on_4 = "STAY"
                    if on_3 == "HOLD":
                        on_3 = "STAY"
                    if on_4 == "HOLD":
                        on_4 = "STAY"
                elif coor_3 != coor_4: # 異動先が被らなかった 
                    if on_3 == "HOLD":
                        on_3 = "OK"
                    if on_4 == "HOLD":
                        on_4 = "OK"    
                
                # 移動前の得点を取得
                p_pnt = env.calcPoint()

                # 移動して次の移動先を取得
                actions_1 = [[ac_1,dir_1,on_1],[ac_2,dir_2,on_2]]
                next_observation_f = env.step_dqn(actions_1,0)
                actions_2 = [[ac_3,dir_3,on_3],[ac_4,dir_4,on_4]]
                next_observation_e = env.step_dqn(actions_2,1)
                next_observation = [next_observation_f,next_observation_e]
                action_f = [action_1,action_2]
                action_e = [action_3,action_4]

                if selfplay:
                    if on_1 == 'NO':
                        no_one += 1
                    if on_2 == 'NO':
                        no_two += 1
                    if on_3 == 'NO':
                        no_three += 1
                    if on_4 == 'NO':
                        no_four += 1


                # 報酬を取得
                reward_1 = env.reward_dqn(on_1,p_pnt,POINTFIELD,next_observation_f[0],observation[0][0])
                reward_2 = env.reward_dqn(on_2,p_pnt,POINTFIELD,next_observation_f[1],observation[0][1])
                reward_f = [reward_1,reward_2]
                reward_3 = env.reward_dqn(on_3,p_pnt,POINTFIELD,next_observation_e[0],observation[1][0])
                reward_4 = env.reward_dqn(on_4,p_pnt,POINTFIELD,next_observation_e[1],observation[1][1])
                reward_e = [reward_3,reward_4]

                # 新状態の取得
                next_ob_f = env.getStatus_enemy(next_observation[0])
                next_ob_e = env.getStatus_enemy(next_observation[1])

                # 新状態の取得 dqn
                p_field,uf_field,ue_field = env.getStatus_dqn(i+1)
                next_user_field = [uf_field,ue_field]
                next_state,memory_flame1_1,memory_flame1_2 = dqn.getState(env,i+1,POINTFIELD,next_user_field,memory_flame1_1,memory_flame1_2,info[4],next_observation,next_ob_f,next_ob_e)

                # 状態の保存
                memory_state_1.add((state_f, action_f, reward_f, [next_state[0],next_state[1]]))
                memory_state_2.add((state_e, action_e, reward_e, [next_state[2],next_state[3]]))

                if episode*40 >= 500*40 and not selfplay:
                    main_n_1.fitting(memory_state_1, gamma, target_n_1)
                    main_n_2.fitting(memory_state_2, gamma, target_n_2)
                
                if DQN_mode:
                    target_n_1.model.set_weights(main_n_1.model.get_weights())
                    target_n_2.model.set_weights(main_n_2.model.get_weights())
                    
                observation = next_observation
                state_f = [next_state[0],next_state[1]]
                state_e = [next_state[2],next_state[3]]

                ob_f = next_ob_f
                ob_e = next_ob_e

                # 報酬の記録 dqn
                #rew_1 = [episode_reward_1,episode_reward_2,avg_total_reward_f,sum_total_reward_f]
                #rew_2 = [episode_reward_3,episode_reward_4,avg_total_reward_e,sum_total_reward_e]
                rew_1[0] += reward_1
                rew_1[1] += reward_2
                rew_2[0] += reward_3
                rew_2[1] += reward_4
                rew_1[2] += (reward_1 + reward_2) / 2
                rew_2[2] += (reward_3 + reward_4) / 2
                rew_1[3] += reward_1 + reward_2
                rew_2[3] += reward_3 + reward_4

            epi_time_delta,fs,now = ts.getTime("timestamp_on",epi_starttime) # 1epoch 実行時間
            epi_processtime.append(epi_time_delta) # 実行時間の記録
            # 1 epoch の報酬の記録
            #save_1 = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF, avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF]
            #save_2 = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF, avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF]
            save_1[0].append(rew_1[0])
            save_1[1].append(rew_1[1])
            save_2[0].append(rew_2[0])
            save_2[1].append(rew_2[1])
            save_1[2].append(rew_1[2])
            save_2[2].append(rew_2[2])
            save_1[3].append(rew_1[3])
            save_2[3].append(rew_2[3])

            save_1[4].append(mean(save_1[0]))
            save_2[4].append(mean(save_2[0]))
            save_1[5].append(mean(save_1[1]))
            save_2[5].append(mean(save_2[1]))
            save_1[6].append(mean(save_1[2]))
            save_2[6].append(mean(save_2[2]))
            save_1[7].append(mean(save_1[3]))
            save_2[7].append(mean(save_2[3]))

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
                win_one += 1
                print('agent1 won')
            else:
                Win2 += 1
                win_two += 1
                print('agent2 won')


            if episode != 0 and episode%250 == 0 and episode!=num_episode-1 : # episode%250 == 0
                info_epoch = [epi_processtime[episode],float(Win1/(episode+1)),float(Win2/(episode+1)),np.argmax(np.array(save_1[2])),save_1[2][np.argmax(np.array(save_1[2]))],np.argmax(np.array(save_2[2])),save_2[2][np.argmax(np.array(save_2[2]))]]
                ts.Log(fm,"now learning",info_epoch,episode+1)
                result = [s,s_avg,save_1,save_2]
                ts.saveImage(fm,result,episode+1)
                print("ok")

        # 学習終了後の後処理
        le_delta,fs,now = ts.getTime("timestamp_on",le_start) # 総実行時間の記録
        print("How many times did QL win, and What is WPCT of QL ?")
        w1 = str(Win1) + " , " + str(float(Win1/num_episode))
        print(w1)
        print("How many times did MCM win, and What is WPCT of MCM ?")
        w2 = str(Win2) + " , " + str(float(Win2/num_episode))
        print(w2)
        m = "finished time is " + str(now)
        print(m)

        info_finished = [Win1,Win2,float(Win1/num_episode),float(Win2/num_episode),np.argmax(np.array(save_1[2])),save_1[2][np.argmax(np.array(save_1[2]))],np.argmax(np.array(save_2[2])),save_2[2][np.argmax(np.array(save_2[2]))],fs,le_delta,tsuyokunatta]
        ts.Log(fm,"finished",info_finished)
        result = [s,s_avg,save_1,save_2]
        ts.saveImage(fm,result,num_episode)
        main_n_1.plot_history(fm,num_episode,'main_n1')
        main_n_2.plot_history(fm,num_episode,'main_n2')
        target_n_1.plot_history(fm,num_episode,'target_n1')
        target_n_2.plot_history(fm,num_episode,'target_n2')
        main_n_1.save_weight(fm,num_episode,'main_n1')
        main_n_2.save_weight(fm,num_episode,'main_n2')
        target_n_1.save_weight(fm,num_episode,'target_n1')
        target_n_2.save_weight(fm,num_episode,'target_n2')
        main_n_1.save_history(fm,num_episode,'main_n1')
        main_n_2.save_history(fm,num_episode,'main_n2')
        target_n_1.save_history(fm,num_episode,'target_n1')
        target_n_2.save_history(fm,num_episode,'target_n2')
        total_no_counts.append(no_counts_one)
        total_no_counts.append(no_counts_two)
        total_no_counts.append(no_counts_three)
        total_no_counts.append(no_counts_four)
        ts.saveImage_nocounts(fm,total_no_counts,num_episode)


    except:

        m = str(sys.exc_info())
        le_delta,fs,now = ts.getTime("timestamp_on",le_start) # 総実行時間の記録
        info_error = [Win1,Win2,float(Win1/kari_epi),float(Win2/kari_epi),np.argmax(np.array(save_1[2])),save_1[2][np.argmax(np.array(save_1[2]))],np.argmax(np.array(save_2[2])),save_2[2][np.argmax(np.array(save_2[2]))],fs,le_delta,m]
        ts.Log(fm,"error",info_error)
        print(m)
        ###
        fn = './log/' + fm + '/'
        with open(fn, 'a') as f:
            traceback.print_exc(file=f)
        ###
        print("How many times did QL win, and What is WPCT of QL ?")
        w1 = str(Win1) + " , " + str(float(Win1/kari_epi))
        print(w1)
        print("How many times did MCM win, and What is WPCT of MCM ?")
        w2 = str(Win2) + " , " + str(float(Win2/kari_epi))
        print(w2)
        m = "error:finished time is " + str(now)
        print(m)
        result = [s,s_avg,save_1,save_2]
        ts.saveImage(fm,result,kari_epi)
        main_n_1.plot_history(fm,kari_epi,'main_n1')
        main_n_2.plot_history(fm,kari_epi,'main_n2')
        target_n_1.plot_history(fm,kari_epi,'target_n1')
        target_n_2.plot_history(fm,kari_epi,'target_n2')
        main_n_1.save_weight(fm,kari_epi,'main_n1')
        main_n_2.save_weight(fm,kari_epi,'main_n2')
        target_n_1.save_weight(fm,kari_epi,'target_n1')
        target_n_2.save_weight(fm,kari_epi,'target_n2')
        main_n_1.save_history(fm,kari_epi,'main_n1')
        main_n_2.save_history(fm,kari_epi,'main_n2')
        target_n_1.save_history(fm,kari_epi,'target_n1')
        target_n_2.save_history(fm,kari_epi,'target_n2')
        total_no_counts.append(no_counts_one)
        total_no_counts.append(no_counts_two)
        total_no_counts.append(no_counts_three)
        total_no_counts.append(no_counts_four)
        ts.saveImage_nocounts(fm,total_no_counts,kari_epi)
