import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
import matplotlib.gridspec as gridspec

def init_plotting():
  sns.set_style("darkgrid", {"axes.facecolor": "0.9"}) 
  # plt.rcParams['figure.figsize'] = (15, 8)
  plt.rcParams['figure.figsize'] = (8.25, 8)
  plt.rcParams['font.family'] = 'DejaVu Sans'
  plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['font.size'] = 13
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
init_plotting()
fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
# fig = plt.figure(constrained_layout=True)
widths = [1,1]
heights = [1,0.25]
spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios = widths, height_ratios =heights)
ax0 = fig.add_subplot(spec[0, 0])
ax1 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[1, 1])
action_dict = json.load(open('../orca/data/json_files/action_list.json'))
action_names = action_dict['actions']
robust_policy_adjust = json.load(open('../misc-files/nondom-tracker/seed_policy_adjust_robust.json'))
policy_adjust = json.load(open('../misc-files/nondom-tracker/seed_policy_adjust.json'))

#########################################robust policies#################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

counts = {}
total = 0
for f in action_names:
  counts[f] = 0
for seed in range(10):
  snapshots = pickle.load(open('../snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
  robust_policies = policy_adjust['%s'%seed]
  for i in robust_policies:
    # print(str(f))
    P = snapshots['best_P'][-1][i]
    f = snapshots['best_f'][-1][i]
    # if f[3] <1.48e9:
    for node in P.L:
        if not node.is_feature:
            a = '%s' % node.value
            counts[a] += 1
            total += 1
for f in action_names:
  counts[f] /= total

sortedkeys = sorted(counts, key=counts.get, reverse=False)

values = [counts[sortedkeys[i]] for i in range(len(counts))]
ax0.barh(range(len(counts)), values, align='center',color='xkcd:indian red',alpha = 0.75)
label_keys = []
###here I'm adjusting the names to be better displayed
new_keys = ['Levee 4', 'Offstream 1', 'Offstream 3', 'Demand 80', 'Levee 5', 'Standard rule', 'Hedging A', \
            'Levee 2', 'Levee 3', 'Hedging B', 'Offstream 2', 'Demand 90', 'Levee 1','Demand 70', 'Groundwater 5', \
            'Groundwater 1','Groundwater 2','Groundwater 4','Groundwater 3', 'Dam 1', 'Dam 2', 'Dam 3']
ax0.set_yticks(range(len(counts)))
ax0.set_yticklabels(new_keys[::-1])

ax0.set_xlim([0,0.08])
ax2.set_xlabel('Fraction of action nodes', weight = 'bold')
ax0.set_title('(a) Robust policies', weight = 'bold')

#######################################robust combined###############################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

values_combined = np.zeros(6)
values = np.flip(values)
values_combined[0] = values[0]+values[4]+values[7]+values[8]+values[12] #Levees
values_combined[1] = values[1]+values[2]+values[10] #Offstream
values_combined[2] = values[5]+values[6]+values[9] #operating policies
values_combined[3] = values[3]+values[11]+values[13] #Demand
values_combined[4] = values[14]+values[15]+values[16]+values[17]+values[18] #Groundwater
values_combined[5] = values[19]+values[20]+values[21]
values_combined = np.flip(values_combined)
ax2.barh(range(len(values_combined)), values_combined, align='center',color='xkcd:indian red',alpha = 0.75)
ax2.set_yticks(range(len(values_combined)))
keys = ['Levees', 'Offstream', 'Operations', 'Demand', 'Groundwater', 'Dam expansion']
ax2.set_yticklabels(keys[::-1])
ax2.set_xlim([0,0.305])
ax2.set_title('(c)', loc = 'left', weight = 'bold')

#######################################non-robust policies###############################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

counts = {}
total = 0
for f in action_names:
  counts[f] = 0
for seed in range(10):
  snapshots = pickle.load(open('../snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
  robust_policies = policy_adjust['%s'%seed]
  for i in robust_policies:
    if i not in robust_policy_adjust['%s'%seed]:
      # print(str(f))
      P = snapshots['best_P'][-1][i]
      f = snapshots['best_f'][-1][i]
      # if f[3] <1.48e9:
      for node in P.L:
          if not node.is_feature:
              a = '%s' % node.value
              counts[a] += 1
              total += 1
for f in action_names:
  counts[f] /= total

sortedkeys = sorted(counts, key=counts.get, reverse=False)

values = [counts[sortedkeys[i]] for i in range(len(counts))]
ax1.barh(range(len(counts)), values, align='center',color='xkcd:indian red',alpha = 0.75)
ax1.set_yticks(range(len(counts)))
##here I'm adjusting the names to be better displayed
new_keys = ['Demand 70', 'Levee 5', 'Offstream 3', 'Offstream 1', ' Standard rule', 'Offstream 2', 'Hedging B', 'Hedging A', 'Levee 4', 'Demand 80', 'Levee 2', 'Levee 3',\
            'Demand 90', 'Levee 1', 'Groundwater 4', 'Groundwater 2', 'Groundwater 0.5', 'Groundwater 0.1', 'Groundwater 1', 'Dam 1', 'Dam 3', 'Dam 2']
ax1.set_yticklabels(new_keys[::-1])

ax1.set_title('(b) Non-robust policies', weight = 'bold')
ax3.set_xlabel('Fraction of action nodes', weight = 'bold')
ax1.set_xlim([0,0.08])


#######################################non-robust combined###############################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

values_combined = np.zeros(6)
values = np.flip(values)
values_combined[0] = values[1] + values[8] + values[10] + values[11] + values[13]# Levees
values_combined[1] = values[2] + values[3] + values[5] # Offstream
values_combined[2] = values[4] + values[6] + values[7] #Operations
values_combined[3] = values[0] + values[9] + values[12] #Demand
values_combined[4] = values[14]+ values[15]+ values[16]+ values[17] + values[18] #Groundwater
values_combined[5] = values[19]+ values[20]+ values[21]
values_combined = np.flip(values_combined)
ax3.barh(range(len(values_combined)), values_combined, align='center',color='xkcd:indian red',alpha = 0.75)
ax3.set_yticks(range(len(values_combined)))
keys = ['Levees', 'Offstream', 'Operations', 'Demand', 'Groundwater', 'Dam expansion']
ax3.set_yticklabels(keys[::-1])
ax3.set_xlim([0,0.305])
ax3.set_title('(d)', loc = 'left', weight = 'bold')
plt.savefig('figures/Figure-7.pdf')

plt.show()