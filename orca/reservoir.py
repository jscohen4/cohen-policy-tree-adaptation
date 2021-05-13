from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
from .util import *

class Reservoir():

	def __init__(self, df, dfh, month, dayofyear, storage_baseline, key, baseline_run = False):

		###################################################################################################    
		################################ initiate parameters and time series ##############################
		###################################################################################################
		self.baseline_run = baseline_run
		T = len(df)
		self.gw = False
		self.dayofyear = dayofyear
		self.month = month

		self.demand_multiplier = df.demand_multiplier.values
		self.key = key

		for k,v in json.load(open('orca/data/json_files/%s_properties.json' % key)).items():
			setattr(self,k,v)
		self.evap_reg = json.load(open('orca/data/json_files/evap_regression.json'))
		self.evap_coeffs = np.asarray(self.evap_reg['%s_evap_coeffs' % key])
		self.evap_int = self.evap_reg['%s_evap_int' % key]
		self.Q = df['%s_in_tr'% key].values * cfs_tafd
		self.E = np.zeros(T)
		self.fci = df['%s_fci' % key].values
		self.slope =  df['%s_slope' % key].values
		self.intercept = df['%s_intercept' % key].values
		self.rem_flow = df['%s_remaining_flow' % key].values
		self.mean = df['%s_mean' % key].values
		self.std = df['%s_std' % key].values  
		self.tas = df['%s_tas' % key].values
		self.obs_flow = df['%s_cum_flow_to_date' % key].values
		self.obs_snow = df['%s_snowpack' % key].values
		# self.BND_trbt_in = df['BND_trbt_fnf'] * cfs_tafd
		#initialize time series arrays
		self.S = np.zeros(T)
		self.gw_S = np.zeros(T)
		self.gw_in = np.zeros(T)
		self.gw_out = np.zeros(T)
		self.gw_cost = np.zeros(T)
		self.R = np.zeros(T)
		self.Rtarget_no_curt = np.zeros(T)
		self.Rtarget = np.zeros(T)
		self.R_to_delta = np.zeros(T)
		self.nodd_shortage = np.zeros(T)
		self.nodd_delivered = np.zeros(T)
		self.nodd_target = np.zeros(T)
		self.S[0] = storage_baseline.values[0]
		self.R[0] = 0
		self.gw_S[0] = 0
		self.storage_bounds = np.zeros(2)
		self.index_bounds = np.zeros(2)
		self.tocs = np.zeros(T)
		self.cum_min_release = np.zeros(366)
		self.forecast = np.zeros(T)
		self.available_storage = np.zeros(T)
		self.soddp = np.zeros(T)
		self.spill = np.zeros(T)
		self.curtailments = np.zeros(T)
		self.shortage_ratio = np.zeros(T)
		# self.car = np.zeros(T)
		###################################################################################################    
		################################ interpolations ###################################################
		###################################################################################################

		#tocs rule variables
		self.tocs_indexS = []

		for i,v in enumerate(self.tocs_rule['index']):        
			self.tocs_indexS.append(np.zeros(366))
			for day in range(0, 366):  
				self.tocs_indexS[i][day] = np.interp(day, self.tocs_rule['dowy'][i], self.tocs_rule['storage'][i])
		self.tocs_index_A = self.tocs_indexS
		self.tocs_index_B = self.tocs_indexS
		self.tocs_indexSA = self.tocs_indexS
		self.tocs_indexSB = self.tocs_indexS

		end = self.tocs_indexS[i][-1]
		for i,fc in enumerate(self.tocs_indexSA):
			self.tocs_index_A[i] = np.delete(self.tocs_indexSA[i], np.s_[61:61+self.FCR_shiftA])
			self.tocs_index_A[i] = np.append(self.tocs_index_A[i],np.tile(end, self.FCR_shiftA))	       
		for i,fc in enumerate(self.tocs_indexSB):
			self.tocs_index_B[i] = np.delete(self.tocs_indexSB[i], np.s_[61:61+self.FCR_shiftB])
			self.tocs_index_B[i] = np.append(self.tocs_index_B[i],np.tile(end, self.FCR_shiftB))
	
		self.nodd_base_int = np.zeros(367)
		for i in range(0,366):
			self.nodd_base_int[i] = np.interp(i, first_of_month, self.nodd_base)



		# self.nodds1 = np.zeros(367)
		# for i in range(0,366):
		# 	self.nodds[i] = np.interp(i, first_of_month, self.nodd1)

		# self.nodds2 = np.zeros(367)
		# for i in range(0,366):
		# 	self.nodds[i] = np.interp(i, first_of_month, self.nodd2)

		# self.nodds3 = np.zeros(367)
		# for i in range(0,366):
		# 	self.nodds3[i] = np.interp(i, first_of_month, self.nodd3)

	def current_tocs(self, tocs_index_p, d, ix):
		for i,v in enumerate(self.tocs_rule['index']):
			if ix > v:
				break
		return tocs_index_p[i][d]


	def step(self, policy, t, d, m, wyt, dowy, y, exceedance, carryover_curtail, floodpool_shift, gw_rate,rdiscount, dmin=0.0, sodd=0.0): 
		if floodpool_shift == 'standard':
			tocs_index_policy = self.tocs_indexS
		elif floodpool_shift == 'policyA':
			tocs_index_policy = self.tocs_index_A
		elif floodpool_shift == 'policyB':
			tocs_index_policy = self.tocs_index_B
		self.nodds = self.nodds
		if dowy == 0:
			self.calc_expected_min_release(t, wyt, self.nodds*self.demand_multiplier[t])
			##what do they expect to need to release for env. requirements through the end of september
			self.forecast[t] = max(0,self.slope[t+1] * self.obs_snow[t+1] + self.intercept[t+1]+ self.std[t]*exceedance[wyt]) #* 1000 #based on forecast regression
        
		envmin = self.env_min_flow[wyt][m-1] * cfs_tafd#minimum allowed environmental flows
		nodd = self.nodd_base_int[d]*	self.demand_multiplier[t]
		self.nodd_target[t] = nodd
  #north of delta demands
		# sodd *= self.sodd_pct_var
		sodd *= self.sodd_pct * self.sodd_curtail_pct[wyt] #south of delta demands
		self.soddp[t] = sodd

		###the variable percentage calculates Folsom & Shasta's contribution to the total releases
		###based on the current 'available storage' in each reservoir, not a fixed percentage based on the total storage of each reservoir
		self.tocs[t] = self.current_tocs(tocs_index_policy,dowy, self.fci[t])
		dout = dmin * self.delta_outflow_pct
		if not self.nodd_meets_envmin:
			envmin += nodd 
		# decide next release
		W = self.S[t-1] + self.Q[t]
		fcr = 0.2*(W-self.tocs[t]) #flood control release
		self.Rtarget[t] = max((fcr, nodd+sodd+dout, envmin)) #target release
		self.Rtarget_no_curt[t] = self.Rtarget[t]
		self.curt = False
		if self.carryover_rule:
			if m >= 5 and m <= 9: #from may to september, accout for carryover storage targets
				if self.forecast[t] + self.S[t-1] - self.Rtarget[t] * (365-dowy) < self.carryover_target[wyt]: #forecasting rest-of-wateryear inflow
					#how much to curtail releases in attempt to meet carryover targets
					self.carryover_curtail_pct = (self.forecast[t] + self.S[t-1] - self.Rtarget[t] * (365-dowy))/self.carryover_target[wyt]
					self.Rtarget[t] = self.Rtarget[t] * max(self.carryover_curtail_pct,carryover_curtail[wyt]) #update target release
					# self.Rtarget[t] = self.Rtarget[t] * max(self.carryover_curtail_pct) #update target release

					self.curt = True
					self.curtailments[t] = max(self.carryover_curtail_pct,carryover_curtail[wyt])
			else:
				self.curtailments[t] = 1
		else:
			self.curtailments[t] = 1
		# then clip based on constraints
		self.R[t] = min(self.Rtarget[t], W - self.dead_pool) # dead-pool constraint
		self.R[t] = min(self.R[t], self.max_outflow * cfs_tafd) #max outflow constraint
		self.spill[t] = max(W - self.R[t] - self.capacity,0)
		#adding gw

		# if self.projection: # if projection mode, calculate evaporation
		# 	X=[]
		# 	storage = self.S[t-1]
		# 	temp = self.tas[t]
		# 	X.append(temp)
		# 	X.append(storage)
		# 	X.append(temp*storage)
		# 	X.append(temp**2)
		# 	X.append(storage**2)
		# 	self.E[t] = max((np.sum(X * self.evap_coeffs) + self.evap_int) * cfs_tafd,0) #evaporation variable
		self.R[t] +=  max(W - self.R[t] - self.capacity, 0) # spill
		self.S[t] = W - self.R[t] #- self.E[t] # mass balance update

		if self.curt == True:
			if gw_rate > 0:
				nodd_used = nodd*max(self.carryover_curtail_pct,carryover_curtail[wyt])
				nodd_lost = nodd*(1/max(self.carryover_curtail_pct,carryover_curtail[wyt],0.1))
				self.gw_out[t] = max(min(gw_rate, self.gw_S[t-1],max(self.gw_cap-self.gw_S[t], 0)),0)
				nodd_in = min(self.gw_out[t], nodd_lost)
				self.nodd_shortage[t] = max(nodd + nodd_lost - nodd_in,0)
				self.nodd_delivered[t] = nodd_lost - nodd_in
				self.R_to_delta += max(0,self.gw_out[t]-nodd_in)
				self.gw_S[t] = self.gw_S[t-1] - self.gw_out[t]

			elif gw_rate == 0:
				nodd_used = nodd*max(self.carryover_curtail_pct,carryover_curtail[wyt])
				self.R_to_delta[t] = max(self.R[t] - nodd_used, 0) # delta calcs need this
				self.gw_S[t] = self.gw_S[t-1] 
				self.nodd_shortage[t] = max(nodd - nodd_used,0)
				self.nodd_delivered[t] = nodd_used

		elif (self.R[t] > max(nodd+sodd+dout, envmin)) & (gw_rate > 0):
			self.gw_in[t] = max(min(gw_rate, self.R[t] - max(nodd+sodd+dout, envmin,max(self.gw_S[t]-self.gw_cap, 0))),0)
			self.gw_S[t] = self.gw_S[t-1] + self.gw_in[t]
			self.R_to_delta[t] = max(self.R[t] - nodd - self.gw_in[t], 0) # delta calcs need this
			self.nodd_shortage[t] = 0
			self.nodd_delivered[t] = nodd

		elif (fcr < max(nodd+sodd+dout, envmin)) & (gw_rate > 0):   
			self.gw_out[t] = min(min(gw_rate,sodd+dout),self.gw_S[t-1])
			self.gw_S[t] = self.gw_S[t-1] - self.gw_out[t]
			self.R_to_delta[t] = max(self.R[t] - nodd + self.gw_out[t], 0) # delta calcs need this
			self.nodd_delivered[t] = nodd
		
		else:
			self.R_to_delta[t] = max(self.R[t] - nodd, 0) # delta calcs need this
			self.gw_S[t] = self.gw_S[t-1]
			self.nodd_delivered[t] = nodd

		self.gw_cost[t] = ((1/(1+rdiscount))**y)*(self.gw_pump_cost * self.gw_out[t]) 
		
		if t > 5:
			return self.R_to_delta[t], self.R_to_delta[t-1], self.R_to_delta[t-3], self.R_to_delta[t-5], self.available_storage[t]
		
		elif t <= 5:
			return self.R_to_delta[t], 0, 0, 0, self.available_storage[t]

	def calc_expected_min_release(self, t, wyt, nodd_dem):

		'''this function calculates the total expected releases needed to meet environmental minimums used in the find_available_storage function
		this is only calculated once per year, at the beginning of the year'''

		self.cum_min_release[0] = 0.0
		##the cum_min_release is the total expected environmental releases between the current day and the end of september in that water year 

		if self.nodd_meets_envmin: #for Shasta and Oroville only 
			for x in range(1,366):
				m = self.month[x-1]
				d = self.dayofyear[x-1]
				#minimum yearly release on first day. Either environmental minimum flow, north of delta demands, or temperature release standards. 
				self.cum_min_release[0] += max(self.env_min_flow[wyt][m-1] * cfs_tafd, nodd_dem[d], self.temp_releases[wyt][m-1] * cfs_tafd) 
			for x in range(1,365):
				m = self.month[x-1]
				#each day the yearly cumulative minimum release is decreased by that days minimum allowed flow. m
				self.cum_min_release[x] = self.cum_min_release[x-1] - max(self.env_min_flow[wyt][m-1] * cfs_tafd, nodd_dem[d], self.temp_releases[wyt][m-1] * cfs_tafd ) 
		else:
			''' same idea, but for folsom. env_min_flow and nodd are combined because flow for agricultural users 
			is diverted before the flow reaches the Lower American River (where the env minimunm flows are to be met)'''
			for x in range(1,366):
				m = self.month[x-1]
				d = self.dayofyear[x-1]
				self.cum_min_release[0] += max(self.env_min_flow[wyt][m-1] * cfs_tafd + nodd_dem[d], self.temp_releases[wyt][m-1] * cfs_tafd)
			for x in range(1,365):
				m = self.month[x-1]
				self.cum_min_release[x] = max(self.cum_min_release[x-1] - self.env_min_flow[wyt][m-1] * cfs_tafd - nodd_dem[d], self.temp_releases[wyt][m-1] * cfs_tafd) 

	def find_available_storage(self, t, d, dowy,wyt,exceedance):#, exceedence_level):
    
		'''this function uses the linear regression variables calculated in find_release_func (called before simulation loop) to figure out how
		much 'excess' storage is available to be released to the delta with the explicit intention of running the pumps.  This function is calculated
		each timestep before the reservoirs' individual step function is called also used to obtain inflow forecasts'''

		#for nodd demand adaptations


		self.forecast[t] = max(0,self.slope[t] * self.obs_snow[t] + self.intercept[t] + self.std[t]*self.exceedance[wyt])#based on forecast regression
		if dowy == 0:
			self.calc_expected_min_release(t,wyt,self.nodds)##what do they expect to need to release for env. requirements through the end of september
			self.forecast[t] = max(0,self.slope[t+1] * self.obs_snow[t+1] + self.intercept[t+1]+ self.std[t]*self.exceedance[wyt]) #* 1000 #based on forecast regression
		self.available_storage[t] = max(0,self.S[t-1] - self.carryover_target[wyt]*self.exceedance[wyt] + self.forecast[t] - self.cum_min_release[dowy])
	
	def results_as_df(self, index):

	##########################################################################################  
	################################ for generating output file ##############################
	##########################################################################################
		df = pd.DataFrame()

		if self.baseline_run == False:	
			names = ['storage', 'out', 'spill', 'gw_cost','NODD_target', 'NODD_shortage']
			things = [self.S, self.R ,self.spill, self.gw_cost, self.nodd_target, self.nodd_shortage]
			for n,t in zip(names,things):
				df['%s_%s' % (self.key,n)] = pd.Series(t, index=index)
		
		elif self.baseline_run == True:
			df = pd.DataFrame()
			names = ['storage', 'out', 'out_to_delta', 'tocs', 'sodd', 'spill', 'forecast', 'curtail', 'gw_storage', 'gw_in', 'gw_out', 'gw_cost', 'NODD_target', 'NODD_delivered','NODD_shortage']
			things = [self.S, self.R, self.R_to_delta, self.tocs,self.soddp,self.spill,self.forecast,self.curtailments, self.gw_S, self.gw_in, self.gw_out, self.gw_cost, self.nodd_target, self.nodd_delivered, self.nodd_shortage]
			for n,t in zip(names,things):
				df['%s_%s' % (self.key,n)] = pd.Series(t, index=index)
		
		return df