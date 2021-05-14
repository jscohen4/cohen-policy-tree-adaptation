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
  plt.rcParams['figure.figsize'] = (8, 6)
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
fig, (ax0,ax1) = plt.subplots(1,2)
features = json.load(open('../orca/data/json_files/indicators_rel_bounds.json'))
feature_names = []
feature_bounds = []
for k,v in features.items():
	feature_names.append(v['name'])
	feature_bounds.append(v['bounds'])
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
for f in feature_names:
	counts[f] = 0
for seed in range(10):

	snapshots = pickle.load(open('../snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	robust_policies = policy_adjust['%s'%seed]
	for i in robust_policies:
		P = snapshots['best_P'][-1][i]
		f = snapshots['best_f'][-1][i]
		for node in P.L:
			if node.is_feature:
				counts[node.name] += 1
				total += 1

for f in feature_names:
	counts[f] /= total


sortedkeys = sorted(counts, key=counts.get, reverse=False)
values = [counts[sortedkeys[i]] for i in np.arange(62-16,62)]
ax0.barh(range(16), values, align='center', color='xkcd:metallic blue',alpha = 0.75)
new_keys = [r'$Q_A \mu_{20} \Delta_5$', r'$Q_{3d}P_{90\%}Y_{20}$', r'$Q_{3M} P_{30\%}Y_{20}$',r'$Q_{3d} P_{90\%}Y_{30}$',r'$Q_{3M} P_{30\%}Y_{30}$', \
						r'$T_{70\%} \mu_{30}$',r'$Q_{1d} P_{50\%}Y_{20}$', r'$Q_A \mu_{30}$',r'$SWE \mu_{30}$', r'$T_{70\%} \mu_{20}$', \
						r'$D\mu_{20}$', r'$SWE \mu_{10}$', r'$T_{90\%} \sigma_{20}$',r'$Q_{1M} P_{50\%}Y_{30}$', r'$T_{70\%} \mu_{10}$',r'$Q_{3M} P_{30\%}Y_{50}$']#,r'$d_3 P_{50}Y_{50}$', r'$T_{70} \mu_{10}$', r'$d_3 P_{50}Y_{30}$', r'$T_{50} \mu_{50}$',\
						#r'$D\mu_{10}$', r'$T_{50} \mu_{30}$']
ax0.set_yticks(range(16))
ax0.set_yticklabels(new_keys[::-1])
ax0.set_title('(a) Robust policies', weight = 'bold',loc = 'left')
ax0.set_xlabel('Fraction of indicator nodes', weight = 'bold')

ax0.set_xlim([0,0.105])
#######################################non-robust policies###############################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
counts = {}
total = 0
for f in feature_names:
	counts[f] = 0
for seed in range(10):

	snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	robust_policies = policy_adjust['%s'%seed]
	for i in range(118):
		if i not in robust_policies:
			P = snapshots['best_P'][-1][i]
			f = snapshots['best_f'][-1][i]
			for node in P.L:
				if node.is_feature:
					counts[node.name] += 1
					total += 1

for f in feature_names:
	counts[f] /= total
sortedkeys = sorted(counts, key=counts.get, reverse=False)
values = [counts[sortedkeys[i]] for i in np.arange(62-16,62)]
ax1.barh(range(16), values, align='center', color='xkcd:metallic blue',alpha = 0.75)

new_keys = [r'$Q_{1M} P_{30\%} Y_{10}$', r'$Q_{1d} P_{50\%} Y_{30}$', r'$SWE\mu_{10}$', r'$Q_{1d} P_{50\%} Y_{50}$', \
						r'$Q_{3d} P_{50\%} Y_{50}$', r'$Q_{3M} P_{50\%} Y_{10}$', r'$Q_{1M} P_{50\%} Y_{10}$',            r'$Q_A \mu_{20}$', \
						r'$T_{90\%} \sigma_{10}$',  r'$Q_{1d} P_{90\%} Y_{20}$', r'$T_{90\%} \mu_{10}$',							 r'$Q_{1M} P_{50\%} Y_{50}$', \
						r'$SWE \mu_{50}$', r'$Q_A \mu_{20} \Delta_{30}$', r'$T_{90\%} \sigma_{50}$', r'$Q_A \mu_{20} \Delta_{20}$']

ax1.set_yticks(range(16))
ax1.set_yticklabels(new_keys[::-1])

ax1.set_xlim([0,0.105])
ax1.set_title('(b) Non-robust policies', weight = 'bold', loc = 'left')
ax1.set_xlabel('Fraction of indicator nodes', weight = 'bold')

plt.tight_layout()
plt.savefig('figures/Figure-10.pdf')

plt.show()