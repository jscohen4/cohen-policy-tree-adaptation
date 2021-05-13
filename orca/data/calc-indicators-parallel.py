import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI


#this will end up in git repository because it is part of paper methods
with open('scenario_names_all.txt') as f:
  scenarios = f.read().splitlines()
with open('demand_scenario_names_all.txt') as f:
  demand_scenarios = f.read().splitlines()

def water_day(d):
    doy = d.dayofyear
    return doy - 274 if doy >= 274 else doy + 91

def get_wday(s,timing_quantile): #use to get quantiles for flow timing. use args when calling with apply. df.resample('AS-OCT').apply(get_wday,quantile = (0.3))
  total = s.sum()
  cs = s.cumsum()
  day = s.index[cs > timing_quantile*total][0]
  return water_day(day)

def get_indicators(sc):
	ts = ['1D','3D','1M','3M','1Y'] #time scales
	tsm = ['1D','3D'] #time scales
	
	qs = [0.10, 0.30, 0.50, 0.70, 0.90] #quantiles, also used for intra-annual timing
	ws = [5,10,20,30,50] #windows
	ds = [5,10,20,30,50] #deltas - changes in statistics over time
	dff = pd.read_csv('scenario_runs/%s/orca-data-climate-forecasted-%s.csv'%(sc,sc), index_col = 0, parse_dates = True) #forcasted dataframe with inputs for indicator calculations
	dff['RES_tas'] = (dff.SHA_tas + dff.ORO_tas + dff.FOL_tas)/3
	dff['RES_tasmin'] = (dff.SHA_tasmin + dff.ORO_tasmin + dff.FOL_tasmin)/3
	dff['RES_tasmax'] = (dff.SHA_tasmax + dff.ORO_tasmax + dff.FOL_tasmin)/3
	dff['RES_pr'] = (dff.SHA_pr*4324 + dff.ORO_pr*3200 + dff.FOL_pr*1850)/(4324+3200+1850)
	dfi = pd.DataFrame()
	#######start with flow indices
	for w in ws: 
		######annual flows
		dfi['fnf_AN_mu_roll%s'%w] = dff.RES_fnf.resample('AS-OCT').sum().rolling(w).mean()
		dfi['fnf_AN_sig_roll%s'%w] = dff.RES_fnf.resample('AS-OCT').sum().rolling(w).std()
		for d in ds:
			#difference in annual flows
			dfi['fnf_AN_mu_roll%s_D%s'%(w,d)] = dfi['fnf_AN_mu_roll%s'%w].pct_change(periods=d)
			dfi['fnf_AN_sig_roll%s_D%s'%(w,d)] = dfi['fnf_AN_sig_roll%s'%w].pct_change(periods=d)
	for w in ws: 
		for t in ts:
			#flow timescale and percentiles
			for q in qs:
				dfi['fnf_%s_pct%0.0f_roll%s'%(t,q*100,w)] = dff.RES_fnf.resample(t).sum().rolling(w).quantile(q)
				for d in ds:
					#differences in the flow quantile statistics
					dfi['fnf_%s_pct%0.0f_roll%s_D%s'%(t,q*100,w,d)] = dfi['fnf_%s_pct%0.0f_roll%s'%(t,q*100,w)].pct_change(periods=d)
	for w in ws: 		
		for q in qs:
			#flow timings
			dfi['fnf_tim%0.0f_mu_roll%s'%(q*100,w)] = dff.RES_fnf.resample('AS-OCT').apply(get_wday,timing_quantile = q).rolling(w).mean()
			dfi['fnf_tim%0.0f_sig_roll%s'%(q*100,w)] = dff.RES_fnf.resample('AS-OCT').apply(get_wday,timing_quantile = q).rolling(w).std()

			for d in ds:
				#differences in the flow timing statistics
				dfi['fnf_tim%0.0f_mu_roll%s_D%s'%(q*100,w,d)] = dfi['fnf_tim%0.0f_mu_roll%s'%(q*100,w)].pct_change(periods=d)
				dfi['fnf_tim%0.0f_sig_roll%s_D%s'%(q*100,w,d)] = dfi['fnf_tim%0.0f_sig_roll%s'%(q*100,w)].pct_change(periods=d)

	for w in ws:
		#snowpack maximums
		dfi['swe_AN_max_mu_roll%s'%w] = dff.RES_swe.resample('AS-OCT').max().rolling(w).mean()
		for d in ds:
			dfi['swe_AN_max_mu_roll%s_D%s'%(w,d)] = dfi['swe_AN_max_mu_roll%s'%w].pct_change(periods=d)
		for m in [12,1,2,3,4,5,6]:
			dfm = dff.RES_swe.resample('M').max()
			dfm = dfm[dfm.index.month == m]
			dfi['swe_month%s_max_mu_roll%s'%(m,w)] = dfm.resample('AS-OCT').max().rolling(w).mean()
			for d in ds:
				dfi['swe_month%s_max_mu_roll%s_D%s'%(m,w,d)] = dfi['swe_month%s_max_mu_roll%s'%(m,w)].pct_change(periods=d)
			dfm = dff.RES_swe.resample('M').min()
			dfm = dfm[dfm.index.month == m]
			dfi['swe_month%s_min_mu_roll%s'%(m,w)] = dfm.resample('AS-OCT').max().rolling(w).mean()
			for d in ds:
				dfi['swe_month%s_min_mu_roll%s_D%s'%(m,w,d)] = dfi['swe_month%s_max_mu_roll%s'%(m,w)].pct_change(periods=d)

	
	for w in ws:
		#mean resampled
		dfi['tas_avg_roll%s'%(w)] = dff.RES_tas.resample('AS-OCT').mean().rolling(w).mean()
		dfi['tasmax_avg_roll%s'%(w)] = dff.RES_tasmax.resample('AS-OCT').mean().rolling(w).mean()
		for d in ds:
			dfi['tas_avg_roll%s'%(w)] = dfi['tas_avg_roll%s'%(w)].pct_change(periods = d)
			dfi['tasmax_avg_roll%s'%(w)] = dfi['tasmax_avg_roll%s'%(w)].pct_change(periods = d)

		#quantiles
		for q in qs:
			for t in ts:
				dfi['tas_%s_pct%0.0f_roll%s'%(t,q*100,w)] = dff.RES_tas.resample(t).mean().rolling(w).quantile(q)
				dfi['tasmax_%s_pct%0.0f_roll%s'%(t,q*100,w)] = dff.RES_tasmax.resample(t).mean().rolling(w).quantile(q)
				for d in ds:
					dfi['tas_%s_pct%0.0f_roll%s_%s'%(t,q*100,w,d)] = dfi['tas_%s_pct%0.0f_roll%s'%(t,q*100,w)].pct_change(periods = d)
					dfi['tasmax_%s_pct%0.0f_roll%s_%s'%(t,q*100,w,d)] = dfi['tasmax_%s_pct%0.0f_roll%s'%(t,q*100,w)].pct_change(periods = d)

		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
			#mean resampled
			dfm = dff.resample('M').mean()
			dfm = dfm[dfm.index.month == m]
			dfi['tas_month%s_avg_roll%s'%(m,w)] = dfm.RES_tas.resample('AS-OCT').mean().rolling(w).mean()
			dfi['tasmax_month%s_avg_roll%s'%(m,w)] = dfm.RES_tasmax.resample('AS-OCT').mean().rolling(w).mean()
			for d in ds:
				dfi['tas_month%s_avg_roll%s_D%s'%(m,w,d)] = dfi['tas_month%s_avg_roll%s'%(m,w)].pct_change(periods = d)
				dfi['tasmax_month%s_avg_roll%s_D%s'%(m,w,d)] = dfi['tasmax_month%s_avg_roll%s'%(m,w)].pct_change(periods = d)
			#quantile resampled
			for q in qs:
				for t in ts:
					dfm = dff.resample(t).mean()
					dfm = dfm[dfm.index.month == m]
					dfi['tas_month%s_%s_pct%0.0f_roll%s'%(m,t,q*100,w)] = dfm.RES_tas.resample(t).mean().rolling(w).quantile(q)
					dfi['tasmax_month%s_%s_pct%0.0f_roll%s'%(m,t,q*100,w)] = dfm.RES_tasmax.resample(t).mean().rolling(w).quantile(q)
					for d in ds:
						dfi['tas_month%s_%s_pct%0.0f_roll%s_D%s'%(m,t,q*100,w,d)] = dfi['tasmax_month%s_%s_pct%0.0f_roll%s'%(m,t,q*100,w)].pct_change(periods = d)
						dfi['tasmax_month%s_%s_pct%0.0f_roll%s_D%s'%(m,t,q*100,w,d)] = dfi['tasmax_month%s_%s_pct%0.0f_roll%s'%(m,t,q*100,w)].pct_change(periods = d)

	for w in ws: 
		######annual flows
		dfi['precip_AN_mu_roll%s'%w] = dff.RES_pr.resample('AS-OCT').sum().rolling(w).mean()
		dfi['precip_AN_sig_roll%s'%w] = dff.RES_pr.resample('AS-OCT').sum().rolling(w).std()
		for d in ds:
			#difference in annual flows
			dfi['precip_AN_mu_roll%s_D%s'%(w,d)] = dfi['precip_AN_mu_roll%s'%w].pct_change(periods=d)
			dfi['precip_AN_sig_roll%s_D%s'%(w,d)] = dfi['precip_AN_sig_roll%s'%w].pct_change(periods=d)
	for w in ws: 
		for t in ts:
			#flow timescale and percentiles
			for q in qs:
				dfi['precip_%s_pct%0.0f_roll%s'%(t,q*100,w)] = dff.RES_pr.resample(t).sum().rolling(w).quantile(q)
				for d in ds:
					#differences in the flow quantile statistics
					dfi['precip_%s_pct%0.0f_roll%s_D%s'%(t,q*100,w,d)] = dfi['precip_%s_pct%0.0f_roll%s'%(t,q*100,w)].pct_change(periods=d)

	dfi = dfi.dropna(axis = 1, how = 'all')
	dfi.to_csv('indicator_files/%s_indicators.csv'%sc)

comm = MPI.COMM_WORLD # communication object
rank = comm.rank # what number processor am I?
index = rank
sc = scenarios[index]
print(sc)
print(rank)
get_indicators(sc)
