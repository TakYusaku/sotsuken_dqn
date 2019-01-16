# read csv files and write csv file
# dqn only

import csv
import numpy as np
import sys

"""
with open('data.csv', 'r') as file:
    lst = list(csv.reader(file))

a = np.array(lst)
print(a)
"""
options = sys.argv[1]
if options == "zeros":
    q_table = np.zeros((88, 9))
elif options == "random":
    q_table = np.random.rand(88,9)

fn = '../hyperpalam/q_table_' + sys.argv[2] + '.csv'
with open(fn, 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(q_table)
