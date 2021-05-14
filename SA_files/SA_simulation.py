import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.sample import latin
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt import PTree
from ptreeopt import PTreeOpt
import random
from mpi4py import MPI
import warnings
# from SALib.sample import latin
from SALib.analyze import sobol

warnings.filterwarnings("ignore", category=RuntimeWarning) 

min_depth = 4

# this whole script will run on all processors requested by the job script
with open('orca/data/scenarios_split_testing.txt') as f:
	testing_scenarios = f.read().splitlines()
with open('indicator_codes.pkl', 'rb') as filehandle:
	indicator_codes = pickle.load(filehandle)

with open('feature_names.pkl', 'rb') as filehandle:
	feature_names = pickle.load(filehandle)

with open('feature_bounds.pkl', 'rb') as filehandle:
	feature_bounds = pickle.load(filehandle)

seed_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust.json'))
count = 0

for i in range(10):
		for j in range(len(seed_policy_adjust['%s'%i])):
			if count == 0:
				seed_pol_arr = [[i,j]]
			else:
				seed_pol_arr = np.concatenate((seed_pol_arr,[[i,j]]),axis = 0)
			count+= 1

dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]

param_vals = np.load('SA_files/latin_samples.npy')
comm = MPI.COMM_WORLD # communication object
rank = comm.rank # what number processor am I?

for i in [0]:
	dfP = pd.DataFrame(index = ['cost','reliability','carrover','flood'])
	seed_pol_num = seed_pol_arr[rank+i*132]
	seed = seed_pol_num[0]
	pol_num = seed_pol_num[1]
	optrun = 'training_scenarios_seed_%s'%seed
	snapshots = pickle.load(open('snapshots/%s.pkl'%optrun, 'rb'))
	P = snapshots['best_P'][-1]
	f = snapshots['best_f'][-1]
	policy = P[pol_num]

	for j,SA_cost_params in enumerate(param_vals):
		flood_sum = 0
		cost_sum = 0
		SWP_shortage = 0
		SWP_demand = 0

		CVP_shortage = 0
		CVP_demand = 0

		SHA_NODD_shortage = 0
		SHA_NODD_target = 0

		FOL_NODD_shortage = 0
		FOL_NODD_target = 0
		carry_arr = np.array([])

		for s in testing_scenarios:
			scset = 0
			sc = '%s-%s'%(s,scset)
			df = pd.read_csv('orca/data/testing_scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')
			dfind = pd.read_csv('orca/data/testing_scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
			dfind = dfind[indicator_codes]
			Model_orca = Model_SA(policy, df, dfind, 81, min_depth, dfh, SHA_baseline[s], ORO_baseline[s], FOL_baseline[s], SA_cost_params, baseline_run = False)
			results, penalty, policies = Model_orca.simulate(policy)
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
		dfP['%s'%j] = [cost_sum, -reliability, carry_count, flood_sum]
	dfP.to_csv('SA_files/SA_testing_outputs/SA_testing_seed_%s_pol_%s.csv'%(seed,pol_num))
