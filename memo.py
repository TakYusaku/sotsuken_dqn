import numpy as np
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

for turn in range(40):
    s = np.digitize(turn, bins=bins(1, 40, 8))
    print("%d,%e"%(turn,s))