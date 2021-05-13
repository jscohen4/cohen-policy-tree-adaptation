import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
import random
import json
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import pickle
from matplotlib import cm
def init_plotting():
  sns.set_style("whitegrid", {"axes.facecolor": "1",'axes.edgecolor': '1','grid.color': '0.1'}) 
  sns.set_context({'grid.linewidth':'2'})
  plt.rcParams['font.family'] = 'DejaVu Sans'
  plt.rcParams['font.weight'] = 'bold'

  plt.rcParams['figure.figsize'] = (10, 4)

  plt.rcParams['font.size'] = 14
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 1.2*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 0.8*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 1*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 1*plt.rcParams['font.size']
init_plotting()

colors = pickle.load(open("parallel-axis-color-order.pkl", "rb" )) 
f_robust = pickle.load(open("f_robust.pkl", "rb" )) 
f_robust_ordered = pickle.load(open("f_robust_ordered.pkl", "rb" )) 
print(len(f_robust_ordered))
robust_scores_ordered = pickle.load(open("robust_scores_ordered.pkl", "rb" )) 
fig,ax0 = plt.subplots(1,1)
print(min(f_robust[:,1]))
print(max(f_robust[:,1]))
baseline_ind = np.load('training_baseline_obj.npy')
baseline_ind[1] =-baseline_ind[1]*0.9
baseline_ind[2] =baseline_ind[2]*1.08

for i,objs in enumerate(f_robust_ordered):
	obj = (objs - f_robust.min(axis=0)) / (f_robust.max(axis=0) - f_robust.min(axis=0))
	plt.plot(range(4), obj, color = colors[i], alpha =0.8, lw = 1,zorder=0)

baseline = (baseline_ind - f_robust.min(axis=0)) / (f_robust.max(axis=0) - f_robust.min(axis=0))
plt.plot(range(4), baseline, color = 'k', lw = 3,zorder=0, label = 'No action')
pol = (f_robust_ordered[1030] - f_robust.min(axis=0)) / (f_robust.max(axis=0) - f_robust.min(axis=0))
plt.plot(range(4), pol, color = 'xkcd:red orange', lw = 3,zorder=0, label = 'Robustness score = 0.94')
print(robust_scores_ordered[1030])
plt.legend(ncol = 2)

ax0.set_xticks(range(4))
plt.gca().set_xticklabels(['\n Cost\n($billion,\ndiscounted)','\n Reliability\n(volumetric)','\n  Carryover storage\n(# below 5000 TAF)','\n Flooding\n(cumulative)'])
ax0.yaxis.grid()
ax0.spines['left'].set_visible(False)
ax0.set_yticklabels([''])

cost_max = max(f_robust[:,0]/47/1000)
cost_min = min(f_robust[:,0]/47/1000)

rel_max = -min(f_robust[:,1])
rel_min = -max(f_robust[:,1])

car_max = max(f_robust[:,2]/47)
car_min = min(f_robust[:,2]/47)

flood_max = max(f_robust[:,3]/47)
flood_min = min(f_robust[:,3]/47)

ax0.text(0,1.12,'%0.2f'%cost_max, color='black', ha='center', va='center',fontsize=14)
ax0.text(0,-0.12,'%0.2f'%cost_min, color='black', ha='center', va='center',fontsize=14)

ax0.text(1,1.12,'%0.2f'%rel_min, color='black', ha='center', va='center',fontsize=14)
ax0.text(1,-0.12,'%0.2f'%rel_max, color='black', ha='center', va='center',fontsize=14)

ax0.text(2,1.12,'%0.2f'%car_max, color='black', ha='center', va='center',fontsize=14)
ax0.text(2,-0.12,'%0.2f'%car_min, color='black', ha='center', va='center',fontsize=14)

ax0.text(3,1.12,'{:.2E}'.format(flood_max), color='black', ha='center', va='center',fontsize=14)
ax0.text(3,-0.12,'{:.2E}'.format(flood_min), color='black', ha='center', va='center',fontsize=14)

plt.tight_layout()
im = ax0.scatter([1,1], [1,1], c=[0,1], s=[0,1], cmap=plt.cm.viridis)
cb = fig.colorbar(im, ax=ax0, aspect = 5, pad=0.1)
cb.ax.tick_params(labelsize=12)
cb.ax.set_title('Robustness\nscore',size = 15, weight = 'bold')
plt.tight_layout()
# plt.grid()
# plt.savefig('parallel-axis-robust.svg')
plt.show()
