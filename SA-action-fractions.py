import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
def init_plotting():
  sns.set_style("darkgrid", {"axes.facecolor": "0.9"}) 
  # plt.rcParams['figure.figsize'] = (15, 8)
  plt.rcParams['figure.figsize'] = (4, 6)

  plt.rcParams['font.size'] = 13
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 1.2*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
init_plotting()

seed_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust.json'))
robust_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust_robust.json'))

count = 0
for i in range(0,10):
    for j in range(len(robust_policy_adjust['%s'%i])):
      if count == 0:
        seed_pol_arr = [[i,j]]
      else:
        seed_pol_arr = np.concatenate((seed_pol_arr,[[i,j]]),axis = 0)
      count+= 1

action_dict = json.load(open('orca/data/json_files/action_list.json'))
action_names = action_dict['actions']


# first: feature scores 
for sens in range(1000):
  counts = {}
  total = 0
  seed = 'test'
  for f in action_names:
    counts[f] = 0
  sensitivity_policies = np.load('SA_files/SA-nondom-policies/params_%s.npy'%sens)
  for i in sensitivity_policies:
    seed = seed_pol_arr[i][0]
    pol = seed_pol_arr[i][1] 
    snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))

    P = snapshots['best_P'][-1][pol]
    f = snapshots['best_f'][-1][pol]
    for node in P.L:
        if not node.is_feature:
            a = '%s' % node.value
            counts[a] += 1
            total += 1
  for f in action_names:
    counts[f] /= total
    print(counts)
  with open('SA_files/action-fractions-sensitivity-combined/SA-action-fractions-combined-%s.json'%sens,'w') as outfile:
  #   json.dump(counts, outfile,indent = 4)