import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt.tree import PTree

with open('orca/data/scenarios_split_testing.txt') as f:
	testing_scenarios = f.read().splitlines()

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
with open('orca/data/scenarios_split_training.txt') as f:
	scenarios = f.read().splitlines()
dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]
for seed in range(10):
	snapshots = pickle.load(open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'rb'))
	policies = snapshots['best_P'][-1]
	num_pol = len(policies)
	first = 0
	last = 235
	min_depth = 4
	i = 98
	P = policies[i]
	policy_actions = []
	for node in P.L:
		if not node.is_feature:
			if node.value not in policy_actions:
				policy_actions.append(node.value)
	print('')
	print(P)
	action_hits = np.zeros(len(policy_actions))
	for sc in testing_scenarios[first:last]: #climate-demand scenarios
			df =pd.read_csv('orca/data/scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')
			dfind = pd.read_csv('orca/data/scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
			dfind = dfind[indicator_codes]
			Model_orca = Model(P, df, dfind, 81, min_depth, dfh, SHA_baseline[sc], ORO_baseline[sc], FOL_baseline[sc], baseline_run = False)
			results, penalty, policy_track = Model_orca.simulate(P)
			for p in policy_track[1:81]:
				index = policy_actions.index(p)
				action_hits[index] += 1
			dfP = pd.DataFrame(index = np.arange(2019,2100))
			dfP['policy'] = policy_track
			dfP.to_csv('action-hits/sc-%s-seed-%s.csv'%(sc,seed))
