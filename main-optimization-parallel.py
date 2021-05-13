import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt.tree import PTree
from ptreeopt import PTreeOpt
import random
with open('orca/data/scenario_names_all.txt') as f:
	scenarios = f.read().splitlines()
with open('orca/data/demand_scenario_names_all.txt') as f:
	demand_scenarios = f.read().splitlines()
with open('orca/data/scenarios_split_training.txt') as f:
	training_scenarios = f.read().splitlines()

seed = 9
random.seed(seed)
np.random.seed(seed)
features = json.load(open('orca/data/json_files/indicators_rel_bounds.json'))
min_depth = 4

with open('indicator_codes.pkl', 'rb') as filehandle:
	indicator_codes = pickle.load(filehandle)

with open('feature_names.pkl', 'rb') as filehandle:
	feature_names = pickle.load(filehandle)

with open('feature_bounds.pkl', 'rb') as filehandle:
	feature_bounds = pickle.load(filehandle)
action_dict = json.load(open('orca/data/json_files/action_list.json'))
actions = action_dict['actions']

#setting up coupled climate-demand scenarios

#######historical and baseline storage data files
dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]

def wrapper_opt(P, first, last):
	shortage_arr =[]
	carry_arr = np.array([])
	flood_sum = 0
	cost_sum = 0
	# print(P)
	i = 0
	SWP_shortage = 0
	SWP_demand = 0

	CVP_shortage = 0
	CVP_demand = 0

	SHA_NODD_shortage = 0
	SHA_NODD_target = 0

	FOL_NODD_shortage = 0
	FOL_NODD_target = 0

	for sc in training_scenarios: #climate-demand scenarios
			# print(sc)
			i+=1
			df =pd.read_csv('orca/data/scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')
			dfind = pd.read_csv('orca/data/scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
			dfind = dfind[indicator_codes]
			Model_orca = Model(P, df, dfind, 81, min_depth, dfh, SHA_baseline[sc], ORO_baseline[sc], FOL_baseline[sc], baseline_run = False)
			results, penalty, policies = Model_orca.simulate(P)
			SWP_shortage += np.sum(results.DEL_SWP_shortage.values)
			SWP_demand += np.sum(results.DEL_SODD_SWP.values)
 
			CVP_shortage += np.sum(results.DEL_CVP_shortage.values)
			CVP_demand += np.sum(results.DEL_SODD_CVP.values)

			SHA_NODD_shortage += np.sum(results.SHA_NODD_shortage.values)
			SHA_NODD_target += np.sum(results.SHA_NODD_target.values)

			FOL_NODD_shortage += np.sum(results.FOL_NODD_shortage.values)
			FOL_NODD_target += np.sum(results.FOL_NODD_target.values)

			flood_result = (results.SHA_spill.values + results.ORO_spill.values + results.FOL_spill.values)*tafd_cfs
			carryover = results.SHA_storage.resample('AS-OCT').last().values + results.ORO_storage.resample('AS-OCT').last().values + results.FOL_storage.resample('AS-OCT').last().values
			carry_arr = np.append(carry_arr, carryover)


			flood_result = flood_result[flood_result != 0]
			flood_sum += np.sum(flood_result)
			cost_sum += results.build_cost.sum()+ results.maintenance_cost.sum()+ results.conservation_cost.sum() + results.SHA_gw_cost.sum() + results.ORO_gw_cost.sum() + results.FOL_gw_cost.sum()
	reliability = (SWP_demand*(1 - SWP_shortage/SWP_demand) + CVP_demand*(1 - CVP_shortage/CVP_demand) + SHA_NODD_target*(1 - SHA_NODD_shortage/SHA_NODD_target) +FOL_NODD_target*(1 - FOL_NODD_shortage/FOL_NODD_target))/(SWP_demand+CVP_demand+SHA_NODD_target+FOL_NODD_target)
	carry_count = len(np.where(carry_arr < 5000)[0])
	# shortage_arr = shortage_arr[~np.isnan(shortage_arr)]
	if cost_sum < 0: 
		cost_sum = 0
	if flood_sum >= 2e9:
		penalty += 10**18
	return [cost_sum + penalty, -reliability + penalty, carry_count + penalty, flood_sum + penalty]
# scsplit = np.arange(0,99,2)
# scsplit[49] = 97
scsplit = [0,50]
algorithm = PTreeOpt(wrapper_opt, 	
					scsplit = scsplit,
					feature_bounds = feature_bounds,
					feature_names = feature_names,
					discrete_actions = True,
					action_names = actions,
					mu = 20,
					cx_prob = 0.7,
					population_size = 96, # hpc hack 7/18/17
					max_depth = 8,multiobj = True,epsilons = [2000,0.0005,2,50000])

#check espsilons after
snapshots = algorithm.run(max_nfe = 48096, log_frequency = 96, parallel=True, filename= 'training_scenarios_seed_%s'%seed)
pickle.dump(snapshots, open('snapshots/training_scenarios_seed_%s.pkl'%seed, 'wb'))
