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

# this whole script will run on all processors requested by the job script
with open('orca/data/scenario_names_all.txt') as f:
	scenarios = f.read().splitlines()
with open('orca/data/demand_scenario_names_all.txt') as f:
	demand_scenarios = f.read().splitlines()

calc_indices = False
climate_forecasts = False
simulation = True
tree_input_files = False
indicator_data_file = False

window_type = 'rolling'
window_length = 40
index_exceedence_sac = 8
shift = 0
SHA_shift = shift
ORO_shift = shift
FOL_shift = shift

SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2006-09-30') & (SHA_baseline.index <= '2099-10-01')]
ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2006-09-30') & (ORO_baseline.index <= '2099-10-01')]
FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2006-09-30') & (FOL_baseline.index <= '2099-10-01')]

features = json.load(open('orca/data/json_files/indicators_whole_bounds.json'))
feature_names = []
feature_bounds = []
indicator_codes = []
min_depth = 4

for k,v in features.items():
	indicator_codes.append(k)
	feature_names.append(v['name'])
	feature_bounds.append(v['bounds'])
	action_dict = json.load(open('orca/data/json_files/action_list.json'))
	actions = action_dict['actions']
snapshots = pickle.load(open('snapshots/training_scenarios_seed_2.pkl', 'rb'))
P = snapshots['best_P'][-1][0]

demand_indicators = {}
for D in demand_scenarios:
	dfdemand = pd.read_csv('orca/data/demand_files/%s.csv'%D, index_col = 0, parse_dates = True)
	dfdemand['demand_multiplier'] = dfdemand['combined_demand']
	dfd_ind = pd.DataFrame(index = dfdemand.index)
	for i in features: #indicators
		ind = features[i]
		if ind['type'] == 'demand':
			if ind['delta'] == 'no':
				if ind['stat'] == 'mu':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).mean()*100
				elif ind['stat'] == 'sig':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).std()*100
				elif ind['stat'] == 'max':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).max()*100
			else:
				if ind['stat'] == 'mu':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).mean().pct_change(periods=ind['delta'])*100
				elif ind['stat'] == 'sig':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).std().pct_change(periods=ind['delta'])*100
				elif ind['stat'] == 'max':
					dfd_ind[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).max().pct_change(periods=ind['delta'])*100
		elif ind['type'] == "discount":
			discount_indicator = i
	demand_indicators[D] = dfd_ind
indicator_columns = []
comm = MPI.COMM_WORLD # communication object
rank = comm.rank # what number processor am I?
sc = scenarios[rank] 
call(['mkdir', 'orca/data/scenario_runs/%s'%sc])
if calc_indices: 
	gains_loop_df = pd.read_csv('orca/data/historical_runs_data/gains_loops.csv', index_col = 0, parse_dates = True)
	OMR_loop_df = pd.read_csv('orca/data/historical_runs_data/OMR_loops.csv', index_col = 0, parse_dates = True)
	input_df = pd.read_csv('orca/data/input_climate_files/%s_input_data.csv'%sc, index_col = 0, parse_dates = True)
	proj_ind_df, ind_df = process_projection(input_df,gains_loop_df,OMR_loop_df,'orca/data/json_files/gains_regression.json','orca/data/json_files/inf_regression.json',window = window_type) 
	proj_ind_df.to_csv('orca/data/scenario_runs/%s/orca-data-processed-%s.csv'%(sc,sc))
	ind_df.to_csv('orca/data/scenario_runs/%s/hydrologic-indicators-%s.csv'%(sc,sc))
	# proj_ind_df = pd.read_csv('orca/data/scenario_runs/%s/orca-data-processed-%s.csv'%(sc,sc),index_col = 0, parse_dates = True)
	WYI_stats_file = pd.read_csv('orca/data/forecast_regressions/WYI_forcasting_regression_stats.csv', index_col = 0, parse_dates = True)
	carryover_stats_file = pd.read_csv('orca/data/forecast_regressions/carryover_regression_statistics.csv', index_col = 0, parse_dates = True)
	print('indices done')
if climate_forecasts:
	proj_ind_df = pd.read_csv('orca/data/scenario_runs/%s/orca-data-processed-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
	forc_df= projection_forecast(proj_ind_df,WYI_stats_file,carryover_stats_file,window_type,window_length, index_exceedence_sac)
	forc_df.to_csv('orca/data/scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc))
	print('forecast done')

if tree_input_files: 
	discount_vals = np.load('orca/data/random-samples/discount_rates.npy')
	random_demands = np.load('orca/data/random-samples/random_demands.npy')
	forc_df = pd.read_csv('orca/data/scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
	df = forc_df
	D = random_demands[rank]
	dfdemand = pd.read_csv('orca/data/demand_files/%s.csv'%D, index_col = 0, parse_dates = True)
	df['demand_multiplier'] = dfdemand['combined_demand']
	df['rdiscount'] = np.ones(len(df))*discount_vals[rank]
	df = df[(df.index >= '2019-10-01') & (df.index <= '2099-09-30')]
	df.to_csv('orca/data/scenario_runs/%s/tree-input-%s.csv'%(sc,sc))
	print('tree file done')

if indicator_data_file: 
	discount_vals = np.load('orca/data/random-samples/discount_rates.npy')
	random_demands = np.load('orca/data/random-samples/random_demands.npy')
	ind_df = pd.read_csv('orca/data/scenario_runs/%s/hydrologic-indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True) #hydrologic indicators
	D = random_demands[rank]
	dfd = demand_indicators[D]
	for c in dfd.columns:
		ind_df[c] = dfd[c]
	ind_df[discount_indicator] = np.ones(len(ind_df))*discount_vals[rank]*100
	ind_df = ind_df.resample('AS-OCT').first()
	ind_df = ind_df[(ind_df.index >= '2019-10-01') & (ind_df.index <= '2099-09-30')]
	ind_df.to_csv('orca/data/scenario_runs/%s/indicators-%s.csv'%(sc,sc))

if simulation:
	df =pd.read_csv('orca/data/scenario_runs/%s/tree-input-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)#, engine = 'python')	df = forc_df
	dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
	dfind = pd.read_csv('orca/data/scenario_runs/%s/indicators-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
	dfind = dfind[indicator_codes]
	Model_orca = Model(P, df, dfind, 81, min_depth, dfh, SHA_baseline[sc], ORO_baseline[sc], FOL_baseline[sc], baseline_run = True)
	projection_results = Model_orca.simulate(P)[0] # takes a while... save results
	projection_results.to_csv('orca/data/scenario_runs/%s/%s-results.csv'%(sc,sc))



