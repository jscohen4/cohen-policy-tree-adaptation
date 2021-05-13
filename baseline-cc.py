import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
from orca import *
from orca.data import *
from ptreeopt.tree import PTree
now = datetime.now().strftime('Last modified %Y-%m-%d %H:%M:%S')
new_discount_rates = [False,1] #seconnd is random seed
climate_indices = False
climate_forecasts = False
run_simulation = False
consolidate_outputs = False
consolidate_inputs = False
#need climate data folders for this, which are too large for github (a few are presevnt in repository for example)
with open('orca/data/scenario_names_all.txt') as f:
	scenarios = f.read().splitlines()

#make discounting file
if new_discount_rates[0] == True:	
	np.random.seed(new_discount_rates[1])
	lower_discount = 0.1
	upper_discount = 0.8
	for sc in scenarios:
		samples = 10
		discount_vals = np.random.uniform(lower_discount,upper_discount,samples)
		np.save('orca/data/discount_files/discount_rates_%s.npy'%sc, discount_vals)
	text_file = open("orca/data/discount_files/datetime.txt", "w")
	text_file.write("%s" %now)
	text_file.close()

scenarios = scenarios[0]
window_type = 'rolling'
window_length = 40
index_exceedence_sac = 8
demand_scenarios =['LUCAS-BAU-mean']
feature_names = ['fnf_AN_mu_roll5']
feature_bounds = [[7,25]]
L = [[0,20],['OpPolA'],['OpPolA']]
indicators = json.load(open('orca/data/json_files/indicators_whole.json'))

P = PTree(L, feature_names = feature_names)
demand_data = {}
for D in demand_scenarios:
	dfdemand = pd.read_csv('orca/data/demand_files/%s.csv'%D, index_col = 0, parse_dates = True)
	dfdemand['demand_multiplier'] = dfdemand['combined_demand']
	for i in indicators:
		ind = indicators[i]
		if ind['type'] == 'demand':
			if ind['delta'] == 'no':
				if ind['stat'] == 'mu':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).mean()*100
				elif ind['stat'] == 'sig':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).std()*100
				elif ind['stat'] == 'max':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).max()*100
			else:
				if ind['stat'] == 'mu':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).mean().pct_change(periods=ind['delta'])*100
				elif ind['stat'] == 'sig':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).std().pct_change(periods=ind['delta'])*100
				elif ind['stat'] == 'max':
					dfdemand[i] = dfdemand.demand_multiplier.resample('AS-OCT').first().rolling(ind['window']).max().pct_change(periods=ind['delta'])*100
		elif ind['type'] == "discount":
			discount_indicator = i
	demand_data[D] = dfdemand
count = 0
for sc in scenarios:
	rcount = 0 #up to 10. 1 for each demand scenario with each climate scenario
	discount_vals = np.load('orca/data/discount_files/discount_rates_%s.npy'%sc)
	count+=1
	print('projection # %s' %count)
	call(['mkdir', 'orca/data/scenario_runs/%s'%sc])
	if climate_indices:
		gains_loop_df = pd.read_csv('orca/data/historical_runs_data/gains_loops.csv', index_col = 0, parse_dates = True)
		OMR_loop_df = pd.read_csv('orca/data/historical_runs_data/OMR_loops.csv', index_col = 0, parse_dates = True)
		input_df = pd.read_csv('orca/data/input_climate_files/%s_input_data.csv'%sc, index_col = 0, parse_dates = True)
		proj_ind_df = process_projection(input_df,gains_loop_df,OMR_loop_df,'orca/data/json_files/gains_regression.json','orca/data/json_files/inf_regression.json',window = window_type) 
		proj_ind_df.to_csv('orca/data/scenario_runs/%s/orca-data-processed-%s.csv'%(sc,sc))


	if climate_forecasts:
		if not climate_indices:
			proj_ind_df = pd.read_csv('orca/data/scenario_runs/%s/orca-data-processed-%s.csv'%(sc,sc),index_col = 0, parse_dates = True)
		WYI_stats_file = pd.read_csv('orca/data/forecast_regressions/WYI_forcasting_regression_stats.csv', index_col = 0, parse_dates = True)
		carryover_stats_file = pd.read_csv('orca/data/forecast_regressions/carryover_regression_statistics.csv', index_col = 0, parse_dates = True)
		forc_df= projection_forecast(proj_ind_df,WYI_stats_file,carryover_stats_file,window_type,window_length, index_exceedence_sac)
		forc_df.to_csv('orca/data/scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc))
	if run_simulation:
		for D in demand_scenarios:
			SHA_baseline = pd.read_csv('orca/data/baseline_storage/SHA_storage.csv',parse_dates = True, index_col = 0)
			SHA_baseline = SHA_baseline[(SHA_baseline.index >= '2019-09-30') & (SHA_baseline.index <= '2099-10-01')]
			ORO_baseline = pd.read_csv('orca/data/baseline_storage/ORO_storage.csv',parse_dates = True, index_col = 0)
			ORO_baseline = ORO_baseline[(ORO_baseline.index >= '2019-09-30') & (ORO_baseline.index <= '2099-10-01')]
			FOL_baseline = pd.read_csv('orca/data/baseline_storage/FOL_storage.csv',parse_dates = True, index_col = 0)
			FOL_baseline = FOL_baseline[(FOL_baseline.index >= '2019-09-30') & (FOL_baseline.index <= '2099-10-01')]
			df =pd.read_csv('orca/data/scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc), index_col = 0, parse_dates = True)
			dfd = demand_data[D]
			for c in dfd.columns:
				df[c] = dfd[c]
			df[discount_indicator] = np.ones(len(df))*discount_vals[rcount]
			df['rdiscount'] = np.ones(len(df))*discount_vals[rcount]
			rdcount += 1
			dfh =pd.read_csv('orca/data/historical_runs_data/results.csv', index_col = 0, parse_dates = True)
			df = df[(df.index >= '2019-10-01') & (df.index <= '2099-10-01')]
			model = Model(P, df, dfh, SHA_baseline[sc], ORO_baseline[sc], FOL_baseline[sc],baseline_run = False, projection = True, sim_gains = True) #climate scenario test
			projection_results = model.simulate(P)[0] # takes a while... save results
			projection_results.to_csv('orca/data/scenario_runs/%s/%s-%s-results.csv'%(sc,sc,D))

if consolidate_outputs: 
	result_ids = ['SHA_storage','SHA_out','SHA_target','SHA_out_to_delta','SHA_tocs','SHA_sodd','SHA_spill','SHA_forecast','SHA_curtail','SHA_gw_storage','SHA_gw_in','SHA_gw_out','SHA_gw_cost','SHA_nodd_shortage',
	'FOL_storage','FOL_out','FOL_target','FOL_out_to_delta','FOL_tocs','FOL_sodd','FOL_spill','FOL_forecast','FOL_curtail','FOL_gw_storage','FOL_gw_in','FOL_gw_out','FOL_gw_cost','FOL_nodd_shortage',
	'ORO_storage','ORO_out','ORO_target','ORO_out_to_delta','ORO_tocs','ORO_sodd','ORO_spill','ORO_forecast','ORO_curtail','ORO_gw_storage','ORO_gw_in','ORO_gw_out','ORO_gw_cost','ORO_nodd_shortage',
	'DEL_in','DEL_out','DEL_TRP_pump','DEL_HRO_pump','DEL_total_pump','DEL_X2','DEL_SODD_CVP','DEL_SODD_SWP','DEL_SWP_shortage','DEL_CVP_shortage','DEL_total_pump_shortage','DEL_Delta_shortage',
	'OFFSTREAM_storage','OFFSTREAM_in','OFFSTREAM_out','build_cost','maintenance_cost','conservation_cost']

	for obj in result_ids:
		df = pd.DataFrame()
		print(obj)
		i = 0
		for sc in scenarios:
			for D in demand_scenarios:
				i+=1
				print('projection # %s' %i)
				dfobj = pd.read_csv('orca/data/scenario_runs/%s/%s-%s-results.csv'%(sc,sc,D), parse_dates = True, index_col = 0)
				df['%s'%sc] = dfobj[obj]
		df.to_csv('orca/data/climate_results/%s.csv'%obj)

if consolidate_inputs: 
	input_ids = ['TLG_fnf','FOL_fnf','MRC_fnf','MIL_fnf','NML_fnf','ORO_fnf','MKM_fnf','BND_fnf','NHG_fnf','SHA_fnf','YRS_fnf','BKL_swe','SHA_pr','ORO_pr','FOL_pr',
	'SHA_tas','ORO_tas','FOL_tas','SHA_tasmax','ORO_tasmax','FOL_tasmax','SHA_tasmin','ORO_tasmin','FOL_tasmin','WY','DOWY','BND_trbt_fnf','BND_trbt_roll','SR_WYI','SR_WYT','SR_WYT_rolling',
	'SJR_WYI','SJR_WYT','8RI','SHA_in_tr','ORO_in_tr','FOL_in_tr','SHA_fci','ORO_fci','FOL_fci','BND_swe','FOL_swe','ORO_swe','YRS_swe','gains_sim','OMR_sim','RES_swe','RES_fnf','WYI_sim','WYT_sim']

	for obj in input_ids:
		df = pd.DataFrame()
		print(obj)
		i = 0
		for sc in scenarios:
			i+=1
			print('projection # %s' %i)
			dfobj = pd.read_csv('orca/data/scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc), parse_dates = True, index_col = 0)
			df['%s'%sc] = dfobj[obj]
		df.to_csv('orca/data/climate_inputs/%s.csv'%obj)		