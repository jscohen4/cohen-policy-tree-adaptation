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
  plt.rcParams['figure.figsize'] = (7.25, 4)
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
fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
action_dict = json.load(open('../orca/data/json_files/action_list.json'))
action_names = action_dict['actions']
robust_policy_adjust = json.load(open('../misc-files/nondom-tracker/seed_policy_adjust_robust.json'))

features = json.load(open('../orca/data/json_files/indicators_rel_bounds.json'))
feature_names = []
for k,v in features.items():
	feature_names.append(v['name'])

full_action_feat= {}
for a in action_names:
	full_action_feat[a] = {}
	for f in feature_names: 
		full_action_feat[a][f] = 0

for seed in range(10):
	robust_policies = robust_policy_adjust['%s'%seed]
	snapshots = pickle.load(open('../snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	policies = snapshots['best_P'][-1]

	num_sol = len(policies)
	for p in range(num_sol):
		action_feat_dict = {}
		for a in action_names:
			action_feat_dict[a] = []
		feature_dict = {}
		print(p)
		P  = policies[p]
		head_node = P.L[0]
		head_node_name = '%s_%0.2f'%(head_node.name,head_node.threshold)
		feature_dict = {}
		feature_name_converter = {}
		feature_dict_chain = {}

		for node in P.L:
			if node.is_feature:
				feature_dict['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				feature_name_converter['%s_%0.2f'%(node.name,node.threshold)] = node.name
		
		for node in P.L:
			if node.is_feature:
				if node.l.is_feature:
					feature_dict['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				if node.r.is_feature:
					feature_dict['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))

		for node_name in feature_dict:
			if feature_dict[node_name] != []:
				head = False
				while head == False:
					if feature_dict_chain[node_name][-1] == head_node_name:
						head = True
					if feature_dict_chain[node_name][-1] != head_node_name:
						next_up_chain = feature_dict[feature_dict_chain[node_name][-1]][0]
						feature_dict_chain[node_name].append(next_up_chain)

		for node in P.L:
			if node.is_feature:
				if not node.l.is_feature:
					action_feat_dict[node.l.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])
				if not node.r.is_feature:
					action_feat_dict[node.r.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])

		for a in action_feat_dict:
			action_feat_dict[a] = list(dict.fromkeys(action_feat_dict[a]))
			for f in action_feat_dict[a]:
				feat = feature_name_converter[f]
				full_action_feat[a][feat] += 1

total = 0

counts = full_action_feat['Levee_4']
for f in counts:
	total += counts[f]
for f in counts:
	counts[f] /= total
sortedkeys = sorted(counts, key=counts.get, reverse=False)
values = [counts[sortedkeys[i]] for i in np.arange(62-4,62)]
ax0.barh(range(4), values, align='center', color='xkcd:metallic blue',alpha = 0.75)

new_keys_levee = [r'$Q_{3d} P_{90\%}Y_{20}$',r'$Q_{3d} P_{90\%}Y_{30}$',r'$T_{70\%} \mu_{30}$', r'$SWE \mu_{10}$']#,r'$\Sigma FNF\mu_{20} \Delta_5$',r'$T_{90} \sigma_{30}$']

ax0.set_yticks(range(4))
ax0.set_yticklabels(new_keys_levee[::-1])
ax0.set_title('(a) Levee 4',weight = 'bold')

###############Offstream 3
for seed in range(10):
	robust_policies = robust_policy_adjust['%s'%seed]
	snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	policies = snapshots['best_P'][-1]

	num_sol = len(policies)
	for p in range(num_sol):
		action_feat_dict = {}
		for a in action_names:
			action_feat_dict[a] = []
		feature_dict = {}
		P  = policies[p]
		head_node = P.L[0]
		head_node_name = '%s_%0.2f'%(head_node.name,head_node.threshold)
		feature_dict = {}
		feature_name_converter = {}
		feature_dict_chain = {}

		for node in P.L:
			if node.is_feature:
				feature_dict['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				feature_name_converter['%s_%0.2f'%(node.name,node.threshold)] = node.name
		
		for node in P.L:
			if node.is_feature:
				if node.l.is_feature:
					feature_dict['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				if node.r.is_feature:
					feature_dict['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))

		for node_name in feature_dict:
			if feature_dict[node_name] != []:
				head = False
				while head == False:
					if feature_dict_chain[node_name][-1] == head_node_name:
						head = True
					if feature_dict_chain[node_name][-1] != head_node_name:
						next_up_chain = feature_dict[feature_dict_chain[node_name][-1]][0]
						feature_dict_chain[node_name].append(next_up_chain)

		for node in P.L:
			if node.is_feature:
				if not node.l.is_feature:
					action_feat_dict[node.l.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])
				if not node.r.is_feature:
					action_feat_dict[node.r.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])

		for a in action_feat_dict:
			action_feat_dict[a] = list(dict.fromkeys(action_feat_dict[a]))
			for f in action_feat_dict[a]:
				feat = feature_name_converter[f]
				full_action_feat[a][feat] += 1

total = 0

counts = full_action_feat['Sites_3']
for f in counts:
	total += counts[f]
for f in counts:
	counts[f] /= total

sortedkeys = sorted(counts, key=counts.get, reverse=False)
print(sortedkeys[0:4])
values = [counts[sortedkeys[i]] for i in np.arange(62-4,62)]
ax1.barh(range(4), values, align='center', color='xkcd:metallic blue',alpha = 0.75)
new_keys_offstream = [r'$Q_{3M} P_{30\%}Y_{20}$',r'$Q_{1d} P_{50\%}Y_{30}$',r'$SWE\mu_{30}$', r'$T_{90\%} \mu_{50}$']#,r'$\Sigma FNF\mu_{20} \Delta_5$',r'$T_{70} \mu_{50}$']

ax1.set_yticks(range(4))
ax1.set_yticklabels(new_keys_offstream[::-1])
ax1.set_title('(b) Offstream 3',weight = 'bold')

###############Demand
for seed in range(10):
	robust_policies = robust_policy_adjust['%s'%seed]
	snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	policies = snapshots['best_P'][-1]

	num_sol = len(policies)
	for p in range(num_sol):
		action_feat_dict = {}
		for a in action_names:
			action_feat_dict[a] = []
		feature_dict = {}
		print(p)
		P  = policies[p]
		head_node = P.L[0]
		head_node_name = '%s_%0.2f'%(head_node.name,head_node.threshold)
		feature_dict = {}
		feature_name_converter = {}
		feature_dict_chain = {}

		for node in P.L:
			if node.is_feature:
				feature_dict['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				feature_name_converter['%s_%0.2f'%(node.name,node.threshold)] = node.name
		
		for node in P.L:
			if node.is_feature:
				if node.l.is_feature:
					feature_dict['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				if node.r.is_feature:
					feature_dict['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))

		for node_name in feature_dict:
			if feature_dict[node_name] != []:
				head = False
				while head == False:
					if feature_dict_chain[node_name][-1] == head_node_name:
						head = True
					if feature_dict_chain[node_name][-1] != head_node_name:
						next_up_chain = feature_dict[feature_dict_chain[node_name][-1]][0]
						feature_dict_chain[node_name].append(next_up_chain)

		for node in P.L:
			if node.is_feature:
				if not node.l.is_feature:
					action_feat_dict[node.l.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])
				if not node.r.is_feature:
					action_feat_dict[node.r.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])

		for a in action_feat_dict:
			action_feat_dict[a] = list(dict.fromkeys(action_feat_dict[a]))
			for f in action_feat_dict[a]:
				feat = feature_name_converter[f]
				full_action_feat[a][feat] += 1

total = 0

counts = full_action_feat['Demand_80']
for f in counts:
	total += counts[f]
for f in counts:
	counts[f] /= total

sortedkeys = sorted(counts, key=counts.get, reverse=False)
values = [counts[sortedkeys[i]] for i in np.arange(62-4,62)]
ax2.barh(range(4), values, align='center', color='xkcd:metallic blue',alpha = 0.75)

new_keys_demand= [r'$D\mu_{20}$',r'$Q_{1M} P_{50\%}Y_{10}$',r'$Q_{1M} P_{30\%}Y_{50}$',r'$D\mu_{10}$']#,r'$SWE_{\mathbf{\max}} \mu_{5}$',r'$D\mu_{20}$']

ax2.set_yticks(range(4))
ax2.set_yticklabels(new_keys_demand[::-1])
ax2.set_xlabel('Fraction of indicator nodes', weight = 'bold')
ax2.set_title('(c) Demand 80',weight = 'bold')

###############Hedging
for seed in range(10):
	robust_policies = robust_policy_adjust['%s'%seed]
	snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	policies = snapshots['best_P'][-1]

	num_sol = len(policies)
	for p in range(num_sol):
		action_feat_dict = {}
		for a in action_names:
			action_feat_dict[a] = []
		feature_dict = {}
		print(p)
		P  = policies[p]
		head_node = P.L[0]
		head_node_name = '%s_%0.2f'%(head_node.name,head_node.threshold)
		feature_dict = {}
		feature_name_converter = {}
		feature_dict_chain = {}

		for node in P.L:
			if node.is_feature:
				feature_dict['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)] = []
				feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				feature_name_converter['%s_%0.2f'%(node.name,node.threshold)] = node.name
		
		for node in P.L:
			if node.is_feature:
				if node.l.is_feature:
					feature_dict['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.l.name,node.l.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
				if node.r.is_feature:
					feature_dict['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))
					feature_dict_chain['%s_%0.2f'%(node.r.name,node.r.threshold)].append('%s_%0.2f'%(node.name,node.threshold))

		for node_name in feature_dict:
			if feature_dict[node_name] != []:
				head = False
				while head == False:
					if feature_dict_chain[node_name][-1] == head_node_name:
						head = True
					if feature_dict_chain[node_name][-1] != head_node_name:
						next_up_chain = feature_dict[feature_dict_chain[node_name][-1]][0]
						feature_dict_chain[node_name].append(next_up_chain)

		for node in P.L:
			if node.is_feature:
				if not node.l.is_feature:
					action_feat_dict[node.l.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])
				if not node.r.is_feature:
					action_feat_dict[node.r.value].extend(feature_dict_chain['%s_%0.2f'%(node.name,node.threshold)])

		for a in action_feat_dict:
			action_feat_dict[a] = list(dict.fromkeys(action_feat_dict[a]))
			for f in action_feat_dict[a]:
				feat = feature_name_converter[f]
				full_action_feat[a][feat] += 1

total = 0

counts = full_action_feat['OpPolA']
for f in counts:
	total += counts[f]
for f in counts:
	counts[f] /= total


sortedkeys = sorted(counts, key=counts.get, reverse=False)
values = [counts[sortedkeys[i]] for i in np.arange(62-4,62)]
ax3.barh(range(4), values, align='center', color='xkcd:metallic blue',alpha = 0.75)
new_keys_hedging = [r'$T_{70\%} \mu_{30}$',r'$SWE \mu_{10}$',r'$Q_A \mu_{20}$',r'$D\mu_{20}$']#,r'$T_{90} \mu{50}$',r'$T_{70} \mu{50}$']

ax3.set_yticks(range(4))
ax3.set_yticklabels(new_keys_hedging[::-1])
ax3.set_xlabel('Fraction of indicator nodes', weight = 'bold')
ax3.set_title('(d) Hedging A',weight = 'bold')

plt.tight_layout()
plt.savefig('figures/Figure-11.pdf')

plt.show()