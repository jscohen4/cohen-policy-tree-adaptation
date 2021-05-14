import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt.tree import PTree
pd.set_option('display.max_rows', 1000)
with open('orca/data/scenarios_split_testing.txt') as f:
	testing_scenarios = f.read().splitlines()
import pickle
def init_plotting():
  sns.set_style("darkgrid", {"axes.facecolor": "0.8"}) 
  plt.rcParams['figure.figsize'] = (8, 5)
  plt.rcParams['font.size'] = 13
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['font.family'] = 'Tahoma'
  plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = 0.75*plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 0.9*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
init_plotting()   
tafd_cfs = 1000 / 86400 * 43560
AF_MCF = 86400 / 1000**2 

#need climate data folders for this, which are too large for github (a few are presevnt in repository for example)
dfh =pd.read_csv('../orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
SHA_baseline = pd.read_csv('../orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('../orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('../orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]

baseline = pd.read_csv('../orca/data/climate_results/maintenance_cost.csv',parse_dates = True, index_col = 0)
baseline = baseline[(baseline.index >= '2019-09-30') & (baseline.index <= '2099-10-01')]

features = json.load(open('../orca/data/json_files/indicators_rel_bounds.json'))
feature_names = []
feature_bounds = []
indicator_codes = []
seed = 4
min_depth = 4

for k,v in features.items():
	indicator_codes.append(k)
	feature_names.append(v['name'])
	feature_bounds.append(v['bounds'])
action_dict = json.load(open('../orca/data/json_files/action_list.json'))
actions = action_dict['actions']
seed = 4
optrun = 'training_scenarios_seed_%s'%seed
snapshots = pickle.load(open('../snapshots/%s.pkl'%optrun, 'rb'))
P = snapshots['best_P'][-1]
f = snapshots['best_f'][-1]
L = P[98]

P = PTree(L, feature_names = feature_names)
count = 0
sc = testing_scenarios[11]	
df =pd.read_csv('../orca/data/scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')
count +=1
dfind = pd.read_csv('../orca/data/scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
dfind = dfind[indicator_codes]
dfind['I57'] = dfind['I57']*0.393701*1.2 #cm to inches
Model_orca = Model(P, df, dfind, 81, min_depth, dfh, SHA_baseline[sc], ORO_baseline[sc], FOL_baseline[sc], baseline_run = False)
results, penalty, policy_track = Model_orca.simulate(P)
fig,((ax0,ax3),(ax1,ax4),(ax2,ax5)) = plt.subplots(3,2,sharex = True)
colors = sns.color_palette("muted")
x_total = np.arange(2020,2100)

ax0.plot(x_total,dfind.I61.values, color = 'k')
ax0.plot([2020,2100],[103,103],color = 'k', ls = '--')
ax0.set_xlim([2020,2099])
ax0.set_ylabel('% historical')
ax0.set_title(r'(a) $\mathbf{D\mu_{20}}$', loc = 'left', weight = 'bold')
ax0.set_ylim([95,145])

ax1.plot(x_total,dfind.I33.values*AF_MCF, color = 'k')
ax1.plot([2020,2100],[519*AF_MCF,519*AF_MCF],color = 'k', ls = '--')
ax1.set_ylabel('MCF')
ax1.set_title(r'(b) $\mathbf{Q_{3d} P_{90\%}Y_{20}}$', loc = 'left', weight = 'bold')
ax1.set_ylim([28, 55])

ax2.plot(x_total,dfind.I26.values, color = 'k')
ax2.plot([2020,2100],[693,693],color = 'k', ls = '--')
ax2.set_ylabel('TAF')
# ax2.set_ylabel(r'$\mathbf{M_3 P_{30} Y_{30}}$'))
ax2.set_title(r'(c) $\mathbf{Q_{3M} P_{30\%} Y_{30}}$', loc = 'left', weight = 'bold')

ax3.plot(x_total,dfind.I59.values, color = 'k')
ax3.plot([2020,2100],[121,121],color = 'k', ls = '--')
ax3.set_ylabel('% historical')
# ax3.set_ylabel(r'$\mathbf{D\mu_{5}}$')
ax3.set_title(r'(d) $\mathbf{D\mu_{5}}$', loc = 'left', weight = 'bold')

ax4.plot(x_total,dfind.I57.values, color = 'k')
ax4.plot([2020,2100],[22,22],color = 'k', ls = '--')
ax4.set_ylabel('Inches')
# ax4.set_ylabel(r'$\mathbf{SWE_{\bf{max}} \mu_{30}}$')
ax4.set_title(r'(e) $\mathbf{SWE \mu_{30}}$', loc = 'left', weight = 'bold')

ax5.plot(x_total,dfind.I44.values, color = 'k')
ax5.plot([2020,2100],[167,167],color = 'k', ls = '--')
ax5.set_ylabel('DOWY')
# ax5.set_ylabel(r'$\mathbf{T_{70} \mu_{30}}$'))
ax5.set_title(r'(f) $\mathbf{T_{70\%} \mu_{30}}$', loc = 'left', weight = 'bold')


# dfind.I61.plot(ax = ax4,label = 'demand_mu_roll20')
# dfind.I33.plot(ax = ax1,label = 'fnf_3D_pct90_yrs20')
# # dfind.I26.plot(ax = ax0, label = 'fnf_3M_pct30_yrs30')
# dfind.I59.plot(ax = ax4,label = 'demand_mu_roll5')
# dfind.I44.plot(ax = ax2,label='fnf_tim70_mu_roll30')
# dfind.I57.plot(ax = ax3,label = 'swe_AN_max_mu_roll30')
# policy_section =None
# 
# if policy_section == 0:
numyears = 5
signal = dfind.I61.values[0:numyears]
time = np.arange(2020,2020 +numyears)
thresh = 103
start = 2020

ax0.fill_between(time,signal,thresh)
ax0.plot([start,start],[thresh,95],color = colors[5],lw = 3, ls = '--')


# elif policy_section == 1:
numyearspass = 5
numyears = 22
signal = dfind.I61.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I61.values[numyearspass]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
start = 2020+numyearspass
thresh = 103
ax0.plot([start,start],[thresh,145], color = colors[0], ls = '--', lw = 3)#,color = colors[0])

signal = dfind.I33.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I33.values[numyearspass]
thresh = 519*AF_MCF
ax1.plot([start,start],[thresh,signal_init*AF_MCF],color = colors[0], ls = '--', lw = 3)

signal = dfind.I26.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I26.values[numyearspass]
thresh = 693
ax2.plot([start,start],[thresh,signal_init],color = colors[0], ls = '--', lw = 3)

signal = dfind.I59.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I59.values[numyearspass]
thresh = 121
ax3.plot([start,start],[thresh,signal_init],color = colors[0], ls = '--', lw = 3)



# elif policy_section == 2:
numyearspass = 27
start = 2020+numyearspass
numyears = 12
signal = dfind.I61.values[numyearspass:(numyearspass+numyears)]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
thresh = 103
ax0.fill_between(time,signal,thresh,color = colors[1])

signal = dfind.I33.values[numyearspass:(numyearspass+numyears)]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
thresh = 519*AF_MCF
ax1.fill_between(time,signal*AF_MCF,thresh,color = colors[1])

signal = dfind.I26.values[numyearspass:(numyearspass+numyears)]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
thresh = 693
ax2.fill_between(time,signal,thresh,color = colors[1])

signal = dfind.I59.values[numyearspass:(numyearspass+numyears)]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
thresh = 121
ax3.fill_between(time,signal,thresh,color = colors[1])
# ax3.plot([start,start],[thresh,ax3.get_ylim()[1]],color = colors[3])



# elif policy_section == 3:
numyearspass = 38
numyears =10
signal = dfind.I61.values[numyearspass:(numyearspass+numyears)]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
start = 2020+numyearspass

thresh = 103
ax0.fill_between(time,signal,thresh,color = colors[2])

signal = dfind.I33.values[numyearspass:(numyearspass+numyears)]
thresh = 519*AF_MCF
ax1.fill_between(time,signal*AF_MCF,thresh,color = colors[2])

signal = dfind.I26.values[numyearspass:(numyearspass+numyears)]
thresh = 693
ax2.fill_between(time,signal,thresh,color = colors[2])
# ax2.plot([start,start],[thresh,ax2.get_ylim()[0]],color = colors[2])

signal = dfind.I57.values[numyearspass:(numyearspass+numyears)]
thresh = 22
ax4.fill_between(time,signal,thresh,color = colors[2])



# elif policy_section == 4:
numyearspass = 47
numyears =26
signal = dfind.I61.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I61.values[numyearspass]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
start = 2020+numyearspass
thresh = 103
ax0.plot([start,start],[thresh,signal_init],color = colors[4], lw = 3, ls = '--')

signal = dfind.I33.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I33.values[numyearspass]
thresh = 519
ax1.plot([start,start],[thresh,signal_init],color = colors[4], lw = 3, ls = '--')

signal = dfind.I26.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I26.values[numyearspass]
thresh = 693
ax2.plot([start,start],[thresh,signal_init],color = colors[4], lw = 3, ls = '--')

signal = dfind.I57.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I57.values[numyearspass]
thresh = 22

ax4.plot([start,start],[thresh,10],color = colors[4], lw = 3, ls = '--')

# elif policy_section == 5:
numyearspass = 72
numyears =9
signal = dfind.I61.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I61.values[numyearspass]
time = np.arange(2020+numyearspass,2020+numyearspass+numyears)
start = 2020+numyearspass
thresh = 103
ax0.plot([start,start],[thresh,signal_init],color = colors[3], lw = 3, ls = '--')

signal = dfind.I33.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I33.values[numyearspass]
thresh = 519*AF_MCF
print(thresh)
ax1.plot([start,start],[thresh,55],color = colors[3], lw = 3, ls = '--')

signal = dfind.I44.values[numyearspass:(numyearspass+numyears)]
signal_init = dfind.I44.values[numyearspass]
thresh = 167
ax5.plot([start,start],[thresh,signal_init],color = colors[8], lw = 3, ls = '--')


df_tracker = pd.DataFrame(index = np.arange(2019,2100))
df_tracker['policy'] = policy_track

ax0.set_xlabel('')
ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')
ax4.set_xlabel('')

plt.tight_layout()
plt.savefig('figures/Figure-6.pdf')
plt.show()
