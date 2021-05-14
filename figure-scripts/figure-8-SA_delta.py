import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
from SALib.analyze import delta
from SALib.util import read_param_file

def init_plotting():
  # sns.set_style("darkgrid", {"axes.facecolor": "0.9"}) 
  # plt.rcParams['figure.figsize'] = (15, 8)
  plt.rcParams['figure.figsize'] = (6, 3.75)
  plt.rcParams['font.family'] = 'DejaVu Sans'
  plt.rcParams['font.weight'] = 'bold'

  plt.rcParams['font.size'] = 12
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 1*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 1*plt.rcParams['font.size']
init_plotting()

np.random.seed(0)
problem = {
  'num_vars': 5,
  'names': ['Levee', 'Dam', 'Offstream', 'Demand', 'GW'],
  'bounds': [[0.25, 1.75]]*5
}

action_dict = json.load(open('../orca/data/json_files/action_list.json'))
action_names = action_dict['actions']
action_names_combined = ['Levee', 'Dam', 'Offstream', 'Demand', 'GW', 'OpPol', 'Standard']
SA_input = np.load('../SA_files/latin_samples.npy')
SA_output = np.load('../SA_files/SA_output.npy')
Si_arr = np.zeros([7,5])
for i in range(0,7):
  Si = delta.analyze(problem, SA_input, SA_output[:,i], num_resamples=10, conf_level=0.95, print_to_console=False)
  Si_arr[i] = Si['delta']
ax = sns.heatmap(Si_arr.T, cmap = 'mako_r', cbar=True,cbar_kws={"shrink": .875})
ax.collections[0].colorbar.set_label("Delta SI", fontweight = 'bold')

ax.set_xticklabels(action_names_combined, rotation = 45, ha = 'right')
ax.set_yticklabels(['Levee', 'Dam', 'Offstream', 'Demand', 'GW'], rotation = 45, ha = 'right')
ax.set_ylabel('Cost multiplier', fontweight = 'bold')
ax.set_xlabel('Action group', fontweight = 'bold')
plt.tight_layout()
plt.savefig('figures/Figure-8.pdf')
plt.show()
