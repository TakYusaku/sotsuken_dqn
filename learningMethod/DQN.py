#! /usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, add
from keras import backend as K
from keras.utils import plot_model
# pip3 install pydotplus and pip3 install graphviz and brew install graphviz and pip install pydot
import pydotplus
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

class DQN:
    def __init__(self,type,info,palam,palam_dense):
        self.image_row = info[0]
        self.image_column = info[1]
        self.channels = info[2]
        self.batch_size = info[3]
        self.action_d = info[4]
        input_num = info[5]
        input_nn_num = info[6]
        self.input_shape = (self.channels, self.image_row, self.image_column)

        if type == "CNN":
            if input_nn_num == 0:
                self.init_CNN_only(palam,input_num)
            else:
                self.init_CNN_and_NN(palam,palam_dense,input_num,input_nn_num)

    def init_CNN_only(self,palam,input_num): # input is only images
        if input_num > 1:
            self.firstlayer = []
            self.model_input = []
            self.model_output = []
            for i in range(input_num):
                palams = palam[i]
                self.firstlayer.append(Sequential())
                # palam -> [filter[len:3], conv_size(:tuple),activation_func[len:5],pool_size(:tuple),input_shape,num_dence, num_action, loss function, optimizer]
                # convolution _1
                self.firstlayer[i].add(Conv2D(palams[0], (palams[2],palams[2]), activation=palams[6], input_shape=self.input_shape)) # batch_size = 10,
                # max pooling _1
                self.firstlayer[i].add(MaxPooling2D(pool_size=(palams[3],palams[3])))
                # convolution _2
                self.firstlayer[i].add(Conv2D(palams[1], (palams[2],palams[2]), activation=palams[7]))
                # max pooling _2 必要ない-> プーリングが必要ないほどConv_2で小さくなっている
                # Flatten layer
                self.firstlayer[i].add(Flatten())
                # make output list and input list
                self.model_output.append(self.firstlayer[i].output)
                self.model_input.append(self.firstlayer[i].input)
            # add all input
            self.added = add(self.model_output)
            self.addblock = Model(self.model_input, self.added)

            # dense layer
            self.dense_layer = Dense(palam[0][4])(self.addblock.output)
            self.dense_layer = Activation(palam[0][8])(self.dense_layer)
            self.dense_layer = Dense(palam[0][5])(self.dense_layer)
            self.dense_layer = Activation(palam[0][9])(self.dense_layer)

            # output and model compiles
            self.model = Model(self.model_input, self.dense_layer)

        elif input_num == 1:
            # palam -> [filter[len:3], conv_size(:tuple),activation_func[len:5],pool_size(:tuple),input_shape,num_dence, num_action, loss function, optimizer]
            # def model
            self.model = Sequential()
            # convolution _1
            self.model.add(Conv2D(palam[0][0],(palam[0][2],palam[0][2]), activation=palam[0][6], input_shape=self.input_shape)) # batch_size = 10,
            # max pooling _1
            self.model.add(MaxPooling2D(pool_size=(palam[0][3],palam[0][3])))#tuple(palam[3])))
            # convolution _2
            self.model.add(Conv2D(palam[0][1], (palam[0][2],palam[0][2]), activation=palam[0][7]))
            # max pooling _2 必要ない-> プーリングが必要ないほどConv_2で小さくなっている
            self.model.add(Flatten())
            # 全結合層1
            self.model.add(Dense(palam[0][4]))
            self.model.add(Activation(palam[0][8]))
            # 全結合層2
            self.model.add(Dense(palam[0][5])) # num_category
            self.model.add(Activation(palam[0][9]))

        self.model.compile(loss=palam[0][10], optimizer=palam[0][11], metrics=['accuracy'])

        # show model summary
        self.model.summary()
        # save model image
        fn = './log/model_init_CNN_only_inputnum_' + str(input_num) + '.png'
        plot_model(self.model, to_file=fn, show_shapes=True)
    
    def init_CNN_and_NN(self,palam,palam_dense,input_num,input_nn_num): # input is images and vector
        if input_num > 1:
            self.firstlayer = []
            self.model_input = []
            self.model_output = []
            for i in range(input_num):
                palams = palam[i]
                self.firstlayer.append(Sequential())
                # palam -> [filter[len:3], conv_size(:tuple),activation_func[len:5],pool_size(:tuple),input_shape,num_dence, num_action, loss function, optimizer]
                # convolution _1
                self.firstlayer[i].add(Conv2D(palams[0], (palams[2],palams[2]), activation=palams[6], input_shape=self.input_shape)) # batch_size = 10,
                # max pooling _1
                self.firstlayer[i].add(MaxPooling2D(pool_size=(palams[3],palams[3])))
                # convolution _2
                self.firstlayer[i].add(Conv2D(palams[1], (palams[2],palams[2]), activation=palams[7]))
                # max pooling _2 必要ない-> プーリングが必要ないほどConv_2で小さくなっている
                # Flatten layer
                self.firstlayer[i].add(Flatten())
                # make output list and input list
                self.model_output.append(self.firstlayer[i].output)
                self.model_input.append(self.firstlayer[i].input)

            self.firstlayer.append(Sequential())
            self.firstlayer[len(self.firstlayer)-1].add(Dense(palam_dense[0], activation=palam_dense[3], input_dim=input_nn_num))
            self.firstlayer[len(self.firstlayer)-1].add(Dense(palam_dense[1], activation=palam_dense[4]))
            self.firstlayer[len(self.firstlayer)-1].add(Dense(palam_dense[2], activation=palam_dense[5]))
            self.model_output.append(self.firstlayer[len(self.firstlayer)-1].output)
            self.model_input.append(self.firstlayer[len(self.firstlayer)-1].input)

            # add all input
            self.added = add(self.model_output)
            self.addblock = Model(self.model_input, self.added)

            # dense layer
            self.dense_layer = Dense(palam[0][4])(self.addblock.output)
            self.dense_layer = Activation(palam[0][8])(self.dense_layer)
            self.dense_layer = Dense(palam[0][5])(self.dense_layer)
            self.dense_layer = Activation(palam[0][9])(self.dense_layer)

            # output and model compiles
            self.model = Model(self.model_input, self.dense_layer)

        elif input_num == 1:
            # palam -> [filter[len:3], conv_size(:tuple),activation_func[len:5],pool_size(:tuple),input_shape,num_dence, num_action, loss function, optimizer]
            # def model
            self.firstlayer = []
            self.firstlayer.append(Sequential())
            self.model_input = []
            self.model_output = []
            # convolution _1
            self.firstlayer[0].add(Conv2D(palam[0][0],(palam[0][2],palam[0][2]), activation=palam[0][6], input_shape=self.input_shape)) # batch_size = 10,
            # max pooling _1
            self.firstlayer[0].add(MaxPooling2D(pool_size=(palam[0][3],palam[0][3])))#tuple(palam[3])))
            # convolution _2
            self.firstlayer[0].add(Conv2D(palam[0][1], (palam[0][2],palam[0][2]), activation=palam[0][7]))
            # max pooling _2 必要ない-> プーリングが必要ないほどConv_2で小さくなっている
            #self.model.add(MaxPooling2D(pool_size=tuple(palam[3])))
            self.firstlayer[0].add(Flatten())
            self.model_output.append(self.firstlayer[0].output)
            self.model_input.append(self.firstlayer[0].input)

            self.firstlayer.append(Sequential())
            self.firstlayer[1].add(Dense(palam_dense[0], activation=palam_dense[3], input_dim=input_nn_num))
            self.firstlayer[1].add(Dense(palam_dense[1], activation=palam_dense[4]))
            self.firstlayer[1].add(Dense(palam_dense[2], activation=palam_dense[5]))
            self.model_output.append(self.firstlayer[1].output)
            self.model_input.append(self.firstlayer[1].input)
                    
            # add all input
            self.added = add(self.model_output)
            self.addblock = Model(self.model_input, self.added)

            # dense layer
            self.dense_layer = Dense(palam[0][4])(self.addblock.output)
            self.dense_layer = Activation(palam[0][8])(self.dense_layer)
            self.dense_layer = Dense(palam[0][5])(self.dense_layer)
            self.dense_layer = Activation(palam[0][9])(self.dense_layer)

            # output and model compiles
            self.model = Model(self.model_input, self.dense_layer)

        self.model.compile(loss=palam[0][10], optimizer=palam[0][11], metrics=['accuracy'])

        # show model summary
        self.model.summary()
        # save model image
        fn = './log/model_init_CNN_and_NN_inputnum_' + str(input_num) + '_inputnnnum_' +str(input_nn_num) + '.png'
        plot_model(self.model, to_file=fn, show_shapes=True)

    def fitting(self, D, gamma, targetQN):
        inputs = np.zeros((self.batch_size, self.channels, self.image_row, self.image_column))
        inputs2 = np.zeros((self.batch_size, self.channels, self.image_row, self.image_column))
        targets = np.zeros((self.batch_size, self.action_d))
        targets2 = np.zeros((self.batch_size, self.action_d))
        mini_batch = D.sample(self.batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i+1] = np.array(state_b[0])
            inputs2[i:i+1] = np.array(state_b[1])
            target = reward_b[0]
            target2 = reward_b[1]

            # 価値計算
            if not (np.array(next_state_b[0]) == np.zeros(np.array(state_b[0]).shape)).all(): # next_state が 0 でない
                # 行動決定のQネットワークと価値観数のQネットワークは分離
                print("on1")
                retmainQs1 = self.model.predict(np.array(next_state_b[0]))[0] # next state に対するpredict
                next_action1 = np.argmax(retmainQs1)  # 最大の報酬を返す行動を選択する
                target = reward_b[0] + gamma * targetQN.model.predict(np.array(next_state_b[0]))[0][next_action1]

            if not (np.array(next_state_b[1]) == np.zeros(np.array(state_b[1]).shape)).all(): # next_state が 0 でない
                print("on2")
                # 行動決定のQネットワークと価値観数のQネットワークは分離
                retmainQs2 = self.model.predict(np.array(next_state_b[1]))[0] # next state に対するpredict
                next_action2 = np.argmax(retmainQs2)  # 最大の報酬を返す行動を選択する
                target2 = reward_b[1] + gamma * targetQN.model.predict(np.array(next_state_b[1]))[0][next_action2]

            targets[i] = self.model.predict(np.array(state_b[0]))    # Qネットワークの出力
            targets[i][action_b[0]] = target               # 教師信号
            targets2[i] = self.model.predict(np.array(state_b[1]))   # Qネットワークの出力
            targets2[i][action_b[1]] = target2             # 教師信号

        self.result = self.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size) # verbose=0 は訓練の様子を表示しない
        self.result = self.model.fit(inputs2, targets2, epochs=1, verbose=1, batch_size=batch_size) # verbose=0 は訓練の様子を表示しない
        self.history = self.result.history
        return self.result
    
    def loadWeight(self, fn):
        self.model.load_weights(fn)

    def save_weight(self, fn, episode, types):
        fm = './log/' + fn + '/nn_weight/' + types + '/' + types + '_weight_' + str(episode) + '.hd5'
        self.model.save_weights(fm, True)

    def loadHistory(self, fn):
        with open(fn, mode='rb') as f:
            self.history = pickle.load(f)

    def save_history(self, fn, episode, types):
        fm = './log/' + fn + '/nn_history/' + types + '_history_' + str(episode) + '.pickle'
        with open(fm, mode='wb') as f:
            pickle.dump(self.history, f)

    def plot_history(self,fm,epoch,name):
        ## 学習時の誤差
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        axL.plot(self.history['loss'],label="loss for training")
        #axL.plot(self.history['val_loss'],label="loss for validation")
        nm1 = 'model loss : ' + name
        axL.set_title(mn1)
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')

        ## 学習時の精度
        axR.plot(self.history['acc'],label="accuracy for training")
        #axR.plot(self.history['val_acc'],label="accuracy for validation")
        mn2 = 'model accuracy : ' + name
        axR.set_title(mn2)
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='lower right')

        fn = './log/' + fm + '/images/model_loss_and_accuracy/model_loss_and_accuracy_' + str(epoch) + '.png'
        plt.savefig(fn)
        plt.close()

    """
    def selfPlay(self,env,num_games,mainQN1,mainQN2):
        for i in range(num_games):
            observation, turn = env.reset()
            for j in range(turn):
                predict1 = mainQN1.model.predict(state1)[0]
                action1 = np.argmax(predict1)
                predict2 = mainQN2.model.predict(state2)[0]
                action2 = np.argmax(predict2)
    """



class ER_Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        samp = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in samp]

    def len(self):
        return len(self.buffer)

class History_Memory:
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        return [self.buffer[i] for i in range(3, -1, -1)]

    def len(self):
        return len(self.buffer)

class Actor:
    def __init__(self,exploration_step,init_er_size,epsilon=None):
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.steps = exploration_step
        self.init_er_size = init_er_size
        self.epsilon_step = (self.init_epsilon - self.final_epsilon) / self.steps
        if epsilon is None:
            self.epsilon = self.init_epsilon
        else:
            self.epsilon = epsilon

    def get_action(self, env, usr, state, mainQN, episode, selfPlay):
        # ε-greedy法
        epsilon = self.epsilon
        action = 0

        if epsilon <= np.random.uniform(0, 1) or selfPlay:
            predict = mainQN.model.predict(state)[0]
            action = np.argmax(predict)
        elif epsilon > np.random.uniform(0, 1) or episode < self.init_er_size:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        on,coor,ac,di=env.deciAction(usr,action) # on = OK,NO,HOLD / coor = coordinate[y(row),x(column)]

        if episode >= self.init_er_size:
            self.epsilon -= self.epsilon_step
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon

        return on,coor,action,ac,di

def getOthers(env,obs,turn):
    obs_f = obs[0]
    obs_e = obs[1]
    our_coordinate = []
    my1 = [] 
    my2 = []
    enemy_coordinate = []
    for i in range(11):
        a = []
        b = []
        c = []
        d = []
        for j in range(8):
            c.append(0)
            d.append(1)
            if obs_f[0] == [i,j] or obs_f[1] == [i,j]:
                a.append(1)
            else:
                a.append(0)
            if obs_e[0] == [i,j] or obs_e[1] == [i,j]:
                b.append(1)
            else:
                b.append(0)
        my1.append(c) # all 0
        my2.append(d) # all 1
        our_coordinate.append(a)
        enemy_coordinate.append(b)

    my_type = [my1,my2] # all 0 is "usr1", all 1 is "usr2"
    turn_image = []
    s = np.digitize(turn, bins=bins(1, 40, 8))
    for i in range(11):
        if i < s+1:
            ka = [1,1,1,1,1,1,1,1]
        else:
            ka = [0,0,0,0,0,0,0,0]
        turn_image.append(ka)
    
    ret = [my_type,our_coordinate,enemy_coordinate,turn_image]

    return ret
           
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def getState(env,i,POINTFIELD,user_field,memory_flame_1,memory_flame_2,info,observation,ob_f,ob_e):
    ret_state = []
    others = getOthers(env,observation,i+1)
    for usr in range(1,3):
        if usr == 1:
            if i!=0:
                memory_flame_1.add(user_field)
            elif i==0:
                for j in range(4):
                    memory_flame_1.add(user_field)
        state = []
        state.append(POINTFIELD)
        user_field_history = memory_flame_1.sample()
        for j in range(4):
            state.append(user_field_history[j][0]) # 4フレーム分uf_fieldをappend
        for j in range(4):
            state.append(user_field_history[j][1]) # 4フレーム分のue_fieldをappend
        if not info:
            for j in range(4):
                if j == 0:
                    if usr == 1:
                        state.append(others[j][0])
                    elif usr == 2:
                        state.append(others[j][1])
                else:
                    state.append(others[j])
            state = [state]
        else:
            state = [state]
            if usr == 1:
                others = [1,ob_f[0],ob_f[1],ob_e[0],ob_e[1],i+1]
            elif usr == 2:
                others = [2,ob_f[0],ob_f[1],ob_e[0],ob_e[1],i+1]
            state.append(others)
        ret_state.append(state)

    for usr in range(3,5):
        u_field_f = [user_field[1],user_field[0]]
        if usr == 3:
            if i!=0:
                memory_flame_2.add(u_field_f)
            elif i==0:
                for j in range(4):
                    memory_flame_2.add(u_field_f)
        state = []
        state.append(POINTFIELD)
        user_field_history = memory_flame_2.sample()
        for j in range(4):
            state.append(user_field_history[j][0]) # 4フレーム分uf_fieldをappend
        for j in range(4):
            state.append(user_field_history[j][1]) # 4フレーム分のue_fieldをappend
        if not info:
            o = [others[0],others[2],others[1],others[3]]
            for j in range(4):
                if j == 0:
                    if usr == 3:
                        state.append(o[j][0])
                    elif usr == 4:
                        state.append(o[j][1])
                else:
                    state.append(o[j])
            state = [state]
        else:
            state = [state]
            if usr == 3:
                others = [3,ob_e[0],ob_e[1],ob_f[0],ob_f[1],i+1]
            elif usr == 4:
                others = [4,ob_e[0],ob_e[1],ob_f[0],ob_f[1],i+1]
            state.append(others)
        ret_state.append(state)
     
    return ret_state,memory_flame_1,memory_flame_2
"""
if __name__ == "__main__":
    palam = [[32,64],[3,3],['relu','relu','relu','relu','softmax'],[2,2],[1,11,8],128,17,'mean_squared_error','adam']
    palameter = [palam, palam, palam, palam, palam, palam, palam, palam, palam, palam]
    dnn = DQN("CNN",palam,1)
"""

