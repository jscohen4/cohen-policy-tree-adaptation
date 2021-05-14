import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt.tree import PTree
from ptreeopt import PTreeOpt
import random
from mpi4py import MPI
from subprocess import call

features = json.load(open('orca/data/json_files/indicators_rel_bounds.json'))
feature_names = []
feature_bounds = []
indicator_codes = []
for k,v in features.items():
	indicator_codes.append(k)
	feature_names.append(v['name'])
	feature_bounds.append(v['bounds'])
action_dict = json.load(open('orca/data/json_files/action_list.json'))
actions = action_dict['actions']
seed = 4
optrun = 'training_scenarios_seed_%s'%seed
snapshots = pickle.load(open('../snapshots/%s.pkl'%optrun, 'rb'))
P = snapshots['best_P'][-1]
f = snapshots['best_f'][-1]
L = P[98]
P = PTree(L, feature_names = feature_names)
plt.show()
P.graphviz_export('figures/Figure-3.svg')
