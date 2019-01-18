# DQN only

import csv
import gym
import numpy as np
import sys
import datetime
import os
import matplotlib.pyplot as plt

def init_func(fm):
    mkdi = './log/' + fm
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/text_log'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_history'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/main_os'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/target_os'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/main_n1'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/main_n2'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/target_n1'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/nn_weight/target_n2'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_totalpoint'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_tilepoint'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_fieldpoint'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_reward'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_reward/agent1'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_reward/agent2'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_reward/avg_avg_totalreward/'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/result_reward/avg_sum_totalreward/'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/accuracy'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/agent1'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/agent2'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/avg'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/loss'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/loss/agent1'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/loss/agent2'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/model_loss_and_accuracy/loss/avg'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/images/no_counts'
    os.mkdir(mkdi)
    """
    mkdi = './log/' + fm + '/im_field'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/im_field/point'
    os.mkdir(mkdi)
    mkdi = './log/' + fm + '/im_field/tile'
    os.mkdir(mkdi)
    """


def readQtable(type):
    fn = './hyperpalam/' + type
    with open(fn, 'r') as file:
        lst = list(csv.reader(file))
    a = []
    for i in range(88):
        a.append(list(map(float,lst[i])))
    q_table = np.array(a)

    return q_table

def writeQtable(fm, type, q_table, episode):
    fn = './log/' + fm + '/q_table/' + str(episode) + '_' + type
    with open(fn, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(q_table)


def readLParam(fn):
    r = []
    s = []
    t = []
    with open(fn, 'r') as file:
        lst = list(csv.reader(file))
    for i in range(len(lst[0])):
        if i==len(lst[0])-1:
            r.append(float(lst[0][i]))
        else:
            r.append(int(lst[0][i]))
    for i in range(r[3]):
        fm = './hyperpalam/cnn_input_' + str(i+1) + '.csv'
        with open(fm, 'r') as file:
            lst = list(csv.reader(file))
        ss = []
        for j in range(len(lst[0])):
            if j < 6:
                ss.append(int(lst[0][j]))
            else:
                ss.append(lst[0][j])
        s.append(ss)
    if r[4]!=0:
        fm = './hyperpalam/nn_input.csv'
        with open(fm, 'r') as file:
            lst = list(csv.reader(file))
        for j in range(len(lst[0])):
            if j < 3:
                t.append(int(lst[0][j]))
            else:
                t.append(lst[0][j])
    return r,s,t

def Log(fm, when,info=None,epoch=None):
    fn = './log/' + fm + '/text_log/' + fm + '.txt'
    f = open(fn,'a')
    if epoch is None and info is None and when is "start":
        m1 = "==================== start ( start time : " + fm + " ) ==================== \n"
        f.write(m1)
        f.close()
    elif epoch is None and when is "info":
        m1 = "li.csv:portnum,epoch,input_image_channels,input_image_num,vector_dim,batch_size_of_input_images,dense_num,output_num,dqn_learning_late\n"
        a = ''
        for i in range(len(info[0])):
            if i == len(info[0])-1:
                a += str(info[0][i]) + "\n"
            else:
                a += str(info[0][i]) + ","
        m1 += a
        b = ''
        for i in range(info[0][3]):
            b = "cnn_input_" + str(i+1) + ".csv:conv2D_1_num,conv2D_2_num,pooling_1_filter,pooling_2_filter,dense_num,output_num,activation_func1,activation_func2,activation_func3,activation_func4,activation_func5,lossfunc_type,optimizer_type\n" 
            for j in range(len(info[1][i])):
                if j == len(info[1][i])-1:
                    b += str(info[1][i][j]) + "\n"
                else:
                    b += str(info[1][i][j]) + ","
            m1 += b
        d = ''
        for i in range(info[0][4]):
            d = "nn_input.csv:dense1,dense2,dense3,activation_func1,activation_func2,activation_func3\n"
            for j in range(len(info[2])):
                d += str(info[2][j])
                if j == len(info[2])-1:
                    d += "\n"
                else:
                    d+= ","
            m1 += d
        m2 = "-------------------------------------------------- \n"
        m = m1 + m2
        f.write(m)
        f.close()

    elif when == "now learning":
        m1 = str(epoch) + " epoch finished : epi_processtime[episode], WPCT of agent1, WPCT of agent2, when is the highest reward of agent1, max reward of agent1, when is the highest reward of agent2, max reward of agent2\n"
        m2 = ''
        for i in info:
            m2 += str(i) + ','
        m1 += (m2 + '\n')
        f.write(m1)
        f.close()

    elif when == 'slp':
        m1 = str(epoch) + " epoch/ selfplay: agent1 won, agent2 won, WPCT of agent1, WPCT of agent2\n"
        m2 = ''
        for i in info:
            m2 += str(i) + ', '
        m1 += (m2 + '\n')
        f.write(m1)
        f.close

    elif when == 'slp_f':
        f.write(info)
        f.close

    elif epoch is None and when is "finished":
        m1 = "successfuly! : runtime is " + str(info[9]) + " \n"
        m2 = "agent1 won : " + str(info[0]) + " , agent2 won : " + str(info[1]) + "/ WPCT of agent1 is " + str(info[2]) + " , agent2 is " + str(info[3]) + " .\n"
        m3 = "when is the highest reward of agent1, max reward of agent1, when is the highest reward of agent2, max reward of agent2, tsuyokunatta: "
        for i in range(4,8):
            m3 += str(info[i]) + ', '
        m3 += str(info[10])
        m3 += '\n'
        m4 = "==================== finished *successfuly* ( finished time : " + info[8] + " ) ==================== \n"
        m = m1 + m2 + m3 + m4
        f.write(m)
        f.close()

    elif epoch is None and when is "error":
        m1 = "error! : runtime is " + str(info[9]) + "\n"
        m2 = "agent1 won : " + str(info[0]) + " , agent2 won : " + str(info[1]) + "/ WPCT of agent1 is " + str(info[2]) + " , agent2 is " + str(info[3])  + "\n"
        m3 = "when is the highest reward of agent1, max reward of agent1, when is the highest reward of agent2, max reward of agent2: "
        for i in range(4,8):
            m3 += str(info[i]) + ', '
        m3 += '\n'
        m4 = info[10]
        m5 = "==================== finished *error* ( finished time : " + info[8] + " ) ==================== \n"
        m = m1 + m2 + m3 + m4 + m5
        f.write(m)
        f.close()

def getTime(type, start=None):
    now = datetime.datetime.now()
    if start is None and type == "filename":
        fs = now.strftime("%Y%m%d_%H%M%S")
        return fs,now
    elif start is None and type == "timestamp_s":
        fs = now.strftime("%Y%m%d_%H%M%S")
        return fs,now
    elif type == "timestamp_on":
        fs = now.strftime("%Y%m%d_%H%M%S")
        delta = now - start
        return delta,fs,now
"""
def mkCSV_reward_init(epoch):
    log_reward = np.zeros((epoch, 2))
    fn = 'q_table_' + sys.argv[1] + '.csv'
    with open(fn, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(q_table)
"""
def saveField(env, fm, epoch, turn):

    if turn == 0:
        mkdi = './log/' + fm + '/im_field/tile/' + str(epoch)
        os.mkdir(mkdi)
        pf = env.savePField()
        pf_a = np.array(pf)
        fn = './log/' + fm + '/im_field/point/' + str(epoch) + '_point.csv'
        with open(fn, 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerows(pf_a)
    uf = env.saveUField()
    uf_a = np.array(uf)
    fn = './log/' + fm + '/im_field/tile/' + str(epoch) + '/' + str(epoch) + '_' + str(turn) + '_tile.csv'
    with open(fn, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(uf_a)

def saveImage(fm,result,episode):
    s = result[0]
    s_avg = result[1]
    save_1 = result[2]
    save_2 = result[3]
    #save_1 = [save_episodereward1,save_episodereward2,save_avg_totalrewardF,save_sum_totalrewardF, avg_save_episodereward1,avg_save_episodereward2,avg_save_avg_totalrewardF,avg_save_sum_totalrewardF]
    #save_2 = [save_episodereward3,save_episodereward4,save_avg_totalrewardE,save_sum_totalrewardE, avg_save_episodereward3,avg_save_episodereward4,avg_save_avg_totalrewardE,avg_save_sum_totalrewardE]
    
    # total point
    plt.figure()
    plt.plot(s[2], 'r', label="agent1",alpha=0.2)
    plt.plot(s[5], 'b', label="agent2",alpha=0.2)
    plt.plot(s_avg[2], 'r', label="agent1 avg")
    plt.plot(s_avg[5], 'b', label="agent2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(s[2]),min(s[5]))-50, max(max(s[2]),max(s[5]))+50)
    plt.xlabel("epoch")
    plt.ylabel("total point")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn1 = './log/' + fm + '/images/result_totalpoint/result_totalpoint_' + str(episode) + '.png'
    plt.savefig(fn1)
    plt.close()

    # tile point
    plt.figure()
    plt.plot(s[0], 'r', label="agent1",alpha=0.2)
    plt.plot(s[3], 'b', label="agent2",alpha=0.2)
    plt.plot(s_avg[0], 'r', label="agent1 avg")
    plt.plot(s_avg[3], 'b', label="agent2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(s[0]),min(s[3]))-50, max(max(s[0]),max(s[3]))+50)
    plt.xlabel("epoch")
    plt.ylabel("tilepoint")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn2 = './log/' + fm + '/images/result_tilepoint/result_tilepoint_' + str(episode) + '.png'
    plt.savefig(fn2)
    plt.close()

    # field point
    plt.figure()
    plt.plot(s[4], 'b', label="agent2",alpha=0.2)
    plt.plot(s[1], 'r', label="agent1", alpha=0.2)
    plt.plot(s_avg[4], 'b', label="agent2 avg")
    plt.plot(s_avg[1], 'r', label="agent1 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(s[1]),min(s[4]))-150, max(max(s[1]),max(s[4]))+50)
    plt.xlabel("epoch")
    plt.ylabel("fieldpoint")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn3 = './log/' + fm + '/images/result_fieldpoint/result_fieldpoint_' + str(episode) + '.png'
    plt.savefig(fn3)
    plt.close()

    # agent1_1 and agent1_2
    plt.figure()
    plt.plot(save_1[0], 'r', label="agent1_1",alpha=0.2)
    plt.plot(save_1[1], 'b', label="agent1_2",alpha=0.2)
    plt.plot(save_1[4], 'r', label="agent1_1 avg")
    plt.plot(save_1[5], 'b', label="agent1_2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_1[0]),min(save_1[1]))-50, max(max(save_1[0]),max(save_1[1]))+50)
    plt.xlabel("epoch")
    plt.ylabel("agent1_1 and agent1_2 : reward")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn4 = './log/' + fm + '/images/result_reward/agent1/result_reward11_and_reward12_' + str(episode) + '.png'
    plt.savefig(fn4)
    plt.close()

    # agent2_1 and agent2_2
    plt.figure()
    plt.plot(save_2[0], 'r', label="agent2_1",alpha=0.2)
    plt.plot(save_2[1], 'b', label="agent2_2",alpha=0.2)
    plt.plot(save_2[4], 'r', label="agent2_1 avg")
    plt.plot(save_2[5], 'b', label="agent2_2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_2[0]),min(save_2[1]))-50, max(max(save_2[0]),max(save_2[1]))+50)
    plt.xlabel("epoch")
    plt.ylabel("agent2_1 and agent2_2 : reward")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn5 = './log/' + fm + '/images/result_reward/agent2/result_reward21_and_reward22_' + str(episode) + '.png'
    plt.savefig(fn5)
    plt.close()

    # avg of avg total reward
    plt.figure()
    plt.plot(save_1[2], 'r', label="agent1",alpha=0.2)
    plt.plot(save_2[2], 'b', label="agent2",alpha=0.2)
    plt.plot(save_1[6], 'r', label="agent1 avg")
    plt.plot(save_2[6], 'b', label="agent2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_1[2]),min(save_2[2]))-50, max(max(save_1[2]),max(save_2[2]))+50)
    plt.xlabel("epoch")
    plt.ylabel("avg of avg total reward")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn6 = './log/' + fm + '/images/result_reward/avg_avg_totalreward/avg_avg_totalreward_' + str(episode) + '.png'
    plt.savefig(fn6)
    plt.close()

    # avg of sum total reward
    plt.figure()
    plt.plot(save_1[3], 'r', label="agent1",alpha=0.2)
    plt.plot(save_2[3], 'b', label="agent2",alpha=0.2)
    plt.plot(save_1[7], 'r', label="agent1 avg")
    plt.plot(save_2[7], 'b', label="agent2 avg")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_1[3]),min(save_2[3]))-50, max(max(save_1[3]),max(save_2[3]))+50)
    plt.xlabel("epoch")
    plt.ylabel("avg of sum total reward")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn7 = './log/' + fm + '/images/result_reward/avg_sum_totalreward/avg_sum_totalreward_' + str(episode) + '.png'
    plt.savefig(fn7)
    plt.close()

def save_history(fm,save_history,episode):
    # history of agent1 acc
    plt.figure()
    plt.plot(save_history[0][0], 'r', label="acc of agent1")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[0][0]))-1, max(max(save_history[0][0])+1))
    plt.xlabel("epoch")
    plt.ylabel("accuracy of agent1")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn8 = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/agent1/acc_agent1_' + str(episode) + '.png'
    plt.savefig(fn8)
    plt.close()

    # history of agent2 acc
    plt.figure()
    plt.plot(save_history[1][0], 'b', label="acc of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[1][0]))-1, max(max(save_history[1][0]))+1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy of agent2")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn9 = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/agent2/acc_agent2_' + str(episode) + '.png'
    plt.savefig(fn9)
    plt.close()

    # history of agent1 and agent2 acc
    plt.figure()
    plt.plot(save_history[0][0], 'r', label="acc of agent1")
    plt.plot(save_history[1][0], 'b', label="acc of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[0][0]),min(save_history[1][0]))-1, max(max(save_history[0][0]),max(save_history[1][0]))+1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn10 = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/acc_' + str(episode) + '.png'
    plt.savefig(fn10)
    plt.close()

    # history of agent1 loss
    plt.figure()
    plt.plot(save_history[0][1], 'r', label="loss of agent1")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[0][1]))-1, max(max(save_history[0][1])+1))
    plt.xlabel("epoch")
    plt.ylabel("accuracy of agent1")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn11 = './log/' + fm + '/images/model_loss_and_accuracy/loss/agent1/loss_agent1_' + str(episode) + '.png'
    plt.savefig(fn11)
    plt.close()

    # history of agent2 loss
    plt.figure()
    plt.plot(save_history[1][1], 'b', label="loss of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[1][1]))-1, max(max(save_history[1][1]))+1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy of agent2")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn12 = './log/' + fm + '/images/model_loss_and_accuracy/loss/agent2/loss_agent2_' + str(episode) + '.png'
    plt.savefig(fn12)
    plt.close()

    # history of agent1 and agent2 loss
    plt.figure()
    plt.plot(save_history[0][1], 'r', label="loss of agent1")
    plt.plot(save_history[1][1], 'b', label="loss of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[0][1]),min(save_history[1][1]))-1, max(max(save_history[0][1]),max(save_history[1][1]))+1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn13 = './log/' + fm + '/images/model_loss_and_accuracy/loss/loss_' + str(episode) + '.png'
    plt.savefig(fn13)
    plt.close()

    # history of agent1 and agent2 avg acc
    plt.figure()
    plt.plot(save_history[2][0], 'r', label="acc of agent1")
    plt.plot(save_history[3][0], 'b', label="acc of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[2][0]),min(save_history[3][0]))-1, max(max(save_history[2][0]),max(save_history[3][0]))+1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn14 = './log/' + fm + '/images/model_loss_and_accuracy/accuracy/avg/acc_avg_' + str(episode) + '.png'
    plt.savefig(fn14)
    plt.close()

    # history of agent1 and agent2 avg loss
    plt.figure()
    plt.plot(save_history[2][1], 'r', label="loss of agent1")
    plt.plot(save_history[3][1], 'b', label="loss of agent2")
    plt.xlim(0, episode)
    plt.ylim(min(min(save_history[2][1]),min(save_history[3][1]))-1, max(max(save_history[2][1]),max(save_history[3][1]))+1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    fn15 = './log/' + fm + '/images/model_loss_and_accuracy/loss/avg/loss_avg_' + str(episode) + '.png'
    plt.savefig(fn15)
    plt.close()


def saveImage_nocounts(fm,nocounts,episode):
    fn = './log/' + fm + '/images/no_counts/' + str(episode) + '_nocounts.png'
    plt.figure()
    plt.plot(nocounts[0], 'r', label="agent1")
    plt.plot(nocounts[1], 'b', label="agent2")
    plt.plot(nocounts[2], 'g', label="agent3")
    plt.plot(nocounts[3], 'm', label="agent4")
    plt.xlim(0, episode)
    plt.ylim(min(min(nocounts[1]),min(nocounts[0]),min(min(nocounts[2]),min(nocounts[3]))-50, max(max(nocounts[3]),max(nocounts[2]),max(nocounts[1]),max(nocounts[0]))+50))
    plt.xlabel("epoch")
    plt.ylabel("number of 'NO' counts")
    plt.legend(loc='lower right')
    #plt.xscale('log')
    plt.savefig(fn)
    plt.close()
"""
def notify(num_episode,Win1,Win2,s3,s6):#,s3,s4,s5,s6):
    #table = Texttable()
    ended_mess = "Learning was successful!\n"
    epoch_mess = "epoch is " + str(num_episode) + "\n"
    result_mess = "How many times did QL win?\n" + str(Win1) + "\n" + "How many times did MCM win?\n" + str(Win2) + "\n"
    finaltotalPoint_mess = "{total point}\n" + "[final point]\n" + "QL is " + str(s3[num_episode-1]) + "\n" + "MCM is " + str(s6[num_episode-1]) + "\n"
    maxtotalPoint_mess = "[max point]\n" + "QL is " + str(max(s3)) + "\n" + "MCM is " + str(max(s6)) + "\n"
    mintotalPoint_mess = "[min point]\n" + "QL is " + str(min(s3)) + "\n" + "MCM is " + str(min(s6)) + "\n"
    finaltilePoint_mess = "{tile point}\n" + "[final point]\n" + "QL is " + str(s1[num_episode-1]) + "\n" + "MCM is " + str(s4[num_episode-1]) + "\n"
    maxtilePoint_mess = "[max point]\n" + "QL is " + str(max(s1)) + "\n" + "MCM is " + str(max(s4)) + "\n"
    mintilePoint_mess = "[min point]\n" + "QL is " + str(min(s1)) + "\n" + "MCM is " + str(min(s4)) + "\n"
    finalpanelPoint_mess = "{panel point}\n" + "[final point]\n" + "QL is " + str(s2[num_episode-1]) + "\n" + "MCM is " + str(s2[num_episode-1]) + "\n"
    maxpanelPoint_mess = "[max point]\n" + "QL is " + str(max(s2)) + "\n" + "MCM is " + str(max(s5)) + "\n"
    minpanelPoint_mess = "[min point]\n" + "QL is " + str(min(s2)) + "\n" + "MCM is " + str(min(s5)) + "\n"
    mess = ended_mess + epoch_mess + result_mess + finaltotalPoint_mess + maxtotalPoint_mess + mintotalPoint_mess #+ finaltilePoint_mess + maxtilePoint_mess + mintilePoint_mess + finalpanelPoint_mess + maxpanelPoint_mess + minpanelPoint_mess
    fig_name = ['./result/result_point.png', './result/result_reward.png']
    #table.add_rows(['total','final','max','min'],['QL',str(s3[num_episode-1]),str(max(s3)),str(min(s3))],['MCM',str(s6[num_episode-1]),str(max(s6)),str(min(s6))])
    Log(m,fm)
    linenotify.main_m(mess)
    for i in range(2):
        linenotify.main_f(fig_name[i],fig_name[i])
"""
