import requests
import sys
import gym
import numpy as np
import json
import random
import gym.spaces
from retry import retry


class procon18Env_re(gym.Env): #define environment
    # initial con
    metadata = {'render.modes': ['human', 'ansi']} #omajinai

    def __init__(self): #initialization
        super().__init__()
        self.local_url = ''
        #initialization of agent1
        self._1action_space = gym.spaces.Discrete(9) #行動(Action)の張る空間
        self._1reward_range = [-120.,100.] #報酬の最小値と最大値のリスト

        #initialization of agent2
        self._2action_space = gym.spaces.Discrete(9) #行動(Action)の張る空間
        self._2reward_range = [-120.,100.] #報酬の最小値と最大値のリスト

        #initialization of agent3
        self._3action_space = gym.spaces.Discrete(9) #行動(Action)の張る空間
        self._3reward_range = [-120.,100.] #報酬の最小値と最大値のリスト

        #initialization of agent4
        self._4action_space = gym.spaces.Discrete(9) #行動(Action)の張る空間
        self._4reward_range = [-120.,100.] #報酬の最小値と最大値のリスト

    @retry(delay=1, backoff=1)
    def makeField(self,pattern,init_order):  # make point field . return is tuple of (Row, Column)   // verified
        url = self.local_url + '/start'
        info = {"init_order":init_order,"pattern":pattern}
        response = requests.post(url, data=info)
        f = response.text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        iv_list = [int(i) for i in f.split()] #listing initial value
        self.terns = iv_list[0] #number of terns
        self.Row = iv_list[1] #row of field
        self.Column = iv_list[2] #column of field
        fs = (self.Row, self.Column) # tuple
        self.pf = [] #field of point
        for i in range(self.Row * self.Column):
            self.pf.append(iv_list[i + 3])
        return fs

    @retry(delay=1, backoff=1)
    def init_setPosition(self):
        self._1pos = self.getPosition(1)
        self._2pos = self.getPosition(2)
        self._3pos = self.getPosition(3)
        self._4pos = self.getPosition(4)
        return [[self._1pos,self._2pos],[self._3pos,self._4pos]]

    @retry(delay=1, backoff=1)
    def reset(self,port=None): # initialization of position,points and steps  (rv is array of position)
        if port is None:
            pass
        else:
            self.local_url = 'http://localhost:' + str(port)
        p = random.choice([[0,1,2,3,4],[0,3,4,1,2],[1,1,3,2,4],[1,3,1,4,2],[2,1,3,4,2],[2,3,1,2,4]])

        self.pattern = p[0]
        p.pop(0)
        self.init_order = p

        fs = self.makeField(self.pattern,self.init_order)

        self._1observation_space = gym.spaces.Box( #観測値(Observation)の張る空間,環境から得られる値
            low = -16, #x軸の最値,y軸の最小値,pointsの最小値
            high = 16, #x,y,pointsの最大値
            shape = fs #yousosu
        )
        self._2observation_space = gym.spaces.Box( #観測値(Observation)の張る空間,環境から得られる値
            low = -16, #x軸の最値,y軸の最小値,pointsの最小値
            high = 16, #x,y,pointsの最大値
            shape = fs #yousosu
        )
        self._3observation_space = gym.spaces.Box( #観測値(Observation)の張る空間,環境から得られる値
            low = -16, #x軸の最値,y軸の最小値,pointsの最小値
            high = 16, #x,y,pointsの最大値
            shape = fs #yousosu
        )
        self._4observation_space = gym.spaces.Box( #観測値(Observation)の張る空間,環境から得られる値
            low = -16, #x軸の最値,y軸の最小値,pointsの最小値
            high = 16, #x,y,pointsの最大値
            shape = fs #yousosu
        )
        observation = self.init_setPosition()
        self.points = [0,0]
        self.now_terns = 0
        return observation

    def countStep(self):
        self.now_terns += 1

    @retry(delay=1, backoff=1)
    def step(self,action,terns,team): # processing of 1step (rv is observation,reward,done,info)
        observation = []
        if team == 0:
            for i in range(2):
                if action[i][1] == "move":
                    self.Move(i+1,action[i][0])
                elif action[i][1] == "remove":
                    self.Remove(i+1,action[i][0])
                elif action[i][1] == "stay":
                    self.Move(i+1,4)
            rewards = self._get_reward_QL(action)
            observation = [self.getPosition(1),self.getPosition(2)]
        elif team == 2:
            for i in range(2):
                if action[i][1] == "move":
                    self.Move(i+team+1,action[i][0])
                elif action[i][1] == "remove":
                    self.Remove(i+team+1,action[i][0])
                elif action[i][1] == "stay":
                    self.Move(i+team+1,4)
            rewards = self._get_reward_MCM(terns,action)
            observation = [self.getPosition(3),self.getPosition(4)]

        #self.done = self._is_done()
        return observation, rewards#, self.done

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    @retry(delay=1, backoff=1)
    def _get_reward_QL(self,action): # return reward (str)  by q-learning
        if self.now_terns == self.terns: # if final
            if self.judVoL() == "Win_1": #if won
                return [10,10]
            else:
                return [-10,-10]
        else:
            p = self.calcPoint()
            if action[0][1] == "oof" or action[1][1] == "oof":
                if action[0][1] == "oof" and action[1][1] != "oof":
                    return [-50,p[2]]
                elif action[0][1] != "oof" and action[1][1] == "oof":
                    return [p[2],-50]
                elif action[0][1] == "oof" and action[1][1] == "oof":
                    return [-50,-50]
            else:
                return [p[2],p[2]]

    @retry(delay=1, backoff=1)
    def _get_reward_MCM(self,terns,action):
        p = self.calcPoint()
        if self.now_terns == self.terns:
            if self.judVoL() == "Win_2":
                return [10,10]
            else:
                #if p[5] > 0: # 負けて合計ポイントが正なら
                if p[5] > 0:
                    r = terns * 0.85 * (-1)
                    return [int(r),int(r)]
                else:
                    return [int(terns * 0.95 * (-1)),int(terns * 0.95 * (-1))]

        else:
            if action[0][1] == "oof" or action[1][1] == "oof":
                if action[0][1] == "oof" and action[1][1] != "oof":
                    return [-50,1]
                elif action[0][1] != "oof" and action[1][1] == "oof":
                    return [1,-50]
                elif action[0][1] == "oof" and action[1][1] == "oof":
                    return [-50,-50]
            else:
                return [p[5],p[5]]

    def _is_done(self): #done or not (bool)
        if self.terns == self.now_terns:
            return True
        else:
            return False

    @retry(delay=1, backoff=1)
    # show で作る list of log  (rv is list)  // verified
    def show(self):
        url = self.local_url + '/show'
        f = requests.post(url).text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        iv_list = [int(i) for i in f.split()]
        lf = []
        for i in range(self.Row):
            l = []
            for j in range(self.Column):
                l.append(iv_list[self.Row * self.Column + self.Column * i + j])
            lf.append(l)
        return lf

    @retry(delay=1, backoff=1)
    def calcPoint(self):
        url = self.local_url + '/pointcalc'
        response = requests.post(url).text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        iv_list = [int(i) for i in response.split()]
        return iv_list

    @retry(delay=1, backoff=1)
    def judVoL(self): #judge won or lose  str  // verified
        p = self.calcPoint()  # @
        #if p[2] > p[5]: # won friends
        if p[2] > p[5]:
            return "Win_1"
        elif p[2] == p[5]: # draw
            if p[0] > p[3]: # won friends (tile point)
                return "Win_1"
            elif p[0] == p[3]:
                re = random.choice(["Win_1", "Win_2"])
                return re
            else:
                return "Win_2"
        else:
            return "Win_2"

    @retry(delay=1, backoff=1)
    def getPosition(self, usr): #get position (array)  // verified
        data = {
          'usr': str(usr)
        }
        url = self.local_url + '/usrpoint'
        response = requests.post(url, data=data)
        f = response.text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        pos_array =[int(i) for i in f.split()]
        return pos_array  # [x(column), y(row)]

    @retry(delay=1, backoff=1)
    def judAc(self, usr, dir,observation):   # judge Actionb   // verified
        data = {
          'usr': str(usr),
          'd': self.gaStr(dir)
        }
        url = self.local_url + '/judgedirection'
        f = requests.post(url, data = data).text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        iv_list = [i for i in f.split()]
        il = [int(iv_list[1]),int(iv_list[0])]

        if iv_list[2] == "Error":
            return False, dir, "oof", il
        elif iv_list[2] == "is_panel":
            if il == observation:
                return True, 4, "stay", il
            else:
                return True, dir, "remove", il
        else:
            if dir == 4:
                return True, 4, "stay", il
            else:
                return True, dir, "move", il

    @retry(delay=1, backoff=1)
    def Move(self, usr, dir): #move agent  // verified
        data = {
          'usr': str(usr),
          'd': self.gaStr(dir)
        }
        url = self.local_url + '/move'
        response = requests.post(url, data=data)

    @retry(delay=1, backoff=1)
    def Remove(self, usr, dir): #remove panels  // verified
        data = {
          'usr': str(usr),
          'd': self.gaStr(dir)
        }
        url = self.local_url + '/remove'
        response = requests.post(url, data=data)

    # dim2 -> dim1  フィールドのマス目に番号を振る #array  // verified
    def getStatus(self, observation): #
        obs1 = observation[0]
        obs2 = observation[1]

        a =  np.array([obs1[1]*12 + obs1[0], obs2[1]*12 + obs2[0]])
        return a  # [int, int]

    @retry(delay=1, backoff=1)
    def savePField(self):
        url = self.local_url + '/show/im_field'
        resp = requests.get(url).text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        li_pfield = [int(i) for i in resp.split()]
        r_list = []
        for i in range(14):
            a = []
            for j in range(14):
                a.append(li_pfield[i*14+j])
            r_list.append(a)
        return r_list

    @retry(delay=1, backoff=1)
    def saveUField(self):
        url = self.local_url + '/show/im_user'
        resp = requests.get(url).text.encode('utf-8').decode().replace("\n", " ").replace("  "," ")
        li_ufield = [int(i) for i in resp.split()]
        r_list = []
        for i in range(14):
            a = []
            for j in range(14):
                a.append(li_ufield[i*14+j])
            r_list.append(a)
        return r_list

    def gaStr(self, action): # get action str // verified
        if action == 0:
            return "lu"
        elif action == 1:
            return "u"
        elif action == 2:
            return "ru"
        elif action == 3:
            return "l"
        elif action == 4:
            return "z"
        elif action == 5:
            return "r"
        elif action == 6:
            return "ld"
        elif action == 7:
            return "d"
        elif action == 8:
            return "rd"

    def grDir(self, action):  # // get reverse action (str)
        if action == 0:
            return "rd"
        elif action == 1:
            return "d"
        elif action == 2:
            return "ld"
        elif action == 3:
            return "r"
        elif action == 4:
            return "s"
        elif action == 5:
            return "l"
        elif action == 6:
            return "ru"
        elif action == 7:
            return "u"
        elif action == 8:
            return "lu"
