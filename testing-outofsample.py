import numpy as np
np.warnings.filterwarnings('ignore') #to not display numpy warnings... be careful
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from datetime import datetime
import warnings
from ptreeopt.tree import PTree
warnings.filterwarnings('ignore')

min_depth = 4
seeds = np.arange(10)
seedsr = np.repeat(seeds,5)

scsets = np.arange(5)
scsetsr = np.tile(scsets,10)

setseeds = np.stack([seedsr,scsetsr],axis = 1)
# this whole script will run on all processors requested by the job script
with open('orca/data/scenarios_split_testing.txt') as f:
	testing_scenarios = f.read().splitlines()

with open('misc-files/indicator_codes.pkl', 'rb') as filehandle:
	indicator_codes = pickle.load(filehandle)

with open('misc-files/feature_names.pkl', 'rb') as filehandle:
	feature_names = pickle.load(filehandle)

with open('misc-files/feature_bounds.pkl', 'rb') as filehandle:
	feature_bounds = pickle.load(filehandle)

seed_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust.json'))


dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]




comm = MPI.COMM_WORLD # communication object
rank = comm.rank # what number processor am I?
setseedsc = setseeds[rank]
seed = setseedsc[0]
scset = setseedsc[1]
optrun = 'training_scenarios_seed_%s'%seed
snapshots = pickle.load(open('snapshots/%s.pkl'%optrun, 'rb'))
P = snapshots['best_P'][-1]
f = snapshots['best_f'][-1]

for pol_num in seed_policy_adjust['%s'%seed]:
	policy = P[pol_num]
	dfP = pd.DataFrame(index = ['cost','reliability','carrover','flood'])
	for s in testing_scenarios:
		carry_arr = []
		sc = '%s-%s'%(s,scset)
		df = pd.read_csv('orca/data/testing_scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')
		dfind = pd.read_csv('orca/data/testing_scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
		dfind = dfind[indicator_codes]
		Model_orca = Model(policy, df, dfind, 81, min_depth, dfh, SHA_baseline[s], ORO_baseline[s], FOL_baseline[s], baseline_run = False)
		results, penalty, policies = Model_orca.simulate(policy)
		SWP_shortage = np.sum(results.DEL_SWP_shortage.values)
		SWP_demand = np.sum(results.DEL_SODD_SWP.values)

		CVP_shortage = np.sum(results.DEL_CVP_shortage.values)
		CVP_demand = np.sum(results.DEL_SODD_CVP.values)

		SHA_NODD_shortage = np.sum(results.SHA_NODD_shortage.values)
		SHA_NODD_target = np.sum(results.SHA_NODD_target.values)

		FOL_NODD_shortage = np.sum(results.FOL_NODD_shortage.values)
		FOL_NODD_target = np.sum(results.FOL_NODD_target.values)

		flood_result = (results.SHA_spill.values + results.ORO_spill.values + results.FOL_spill.values)*tafd_cfs
		carryover = results.SHA_storage.resample('AS-OCT').last().values + results.ORO_storage.resample('AS-OCT').last().values + results.FOL_storage.resample('AS-OCT').last().values
		carry_arr = np.append(carry_arr, carryover)


		flood_result = flood_result[flood_result != 0]
		flood_sum = np.sum(flood_result)
		cost_sum = results.build_cost.sum()+ results.maintenance_cost.sum()+ results.conservation_cost.sum() + results.SHA_gw_cost.sum() + results.ORO_gw_cost.sum() + results.FOL_gw_cost.sum()
		reliability = (SWP_demand*(1 - SWP_shortage/SWP_demand) + CVP_demand*(1 - CVP_shortage/CVP_demand) + SHA_NODD_target*(1 - SHA_NODD_shortage/SHA_NODD_target) +FOL_NODD_target*(1 - FOL_NODD_shortage/FOL_NODD_target))/(SWP_demand+CVP_demand+SHA_NODD_target+FOL_NODD_target)
		carry_count = len(np.where(carry_arr < 5000)[0])
		dfP['%s'%(sc)] = [cost_sum, -reliability, carry_count, flood_sum]
		dfP.to_csv('testing_outputs/seed_%s_pol_%s_scset_%s.csv'%(seed,pol_num,scset))
