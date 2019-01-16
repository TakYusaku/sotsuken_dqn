import requests
import random
import numpy as np
import os
import datetime
import pprint

def main():
    p = random.choice([[0,1,2,3,4],[0,3,4,1,2],[1,1,3,2,4],[1,3,1,4,2],[2,1,3,4,2],[2,3,1,2,4]])
    pattern = p[0]
    p.pop(0)
    init_order = p

    info = {"init_order":init_order,"pattern":pattern}
    response = requests.post('http://localhost:8000/start', data=info)

    print(init_order)
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

if __name__ == "__main__":
    #main()
    #print([i for i in range(3, -1, -1)])
    turn = 10
    for i in range(40):
        a = np.digitize(i+1, bins=bins(1, 40, 8))
        print("%d is %s" % (i+1, a))