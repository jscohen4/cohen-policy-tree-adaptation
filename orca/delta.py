from __future__ import division
import numpy as np 
import pandas as pd
import json
from .util import *

class Delta():

	def __init__(self, df, key, baseline_run = False):

		################################################################################################### 
		################################ basic time and key parameters ####################################
		###################################################################################################
		self.baseline_run = baseline_run
		T = len(df)
		self.key = key
		self.demand_multiplier = df.demand_multiplier.values
		##########################################################################################################################
		################################ Gains, Old and Middle River, and San Joaquin options ####################################
		##########################################################################################################################

		self.OMR_sim = df.OMR_sim.values
		self.netgains = df.gains_sim.values
		# self.sanjoaquin = self.netgains - df.YRS_fnf.values - df.NML_fnf.values
		self.sanjoaquin = df.sanjoaquin.values
		self.san_joaquin_ie_amt = df.san_joaquin_ie_amt.values

		#############################################################################################################   
		################################ extract Delta properties from json file ####################################
		#############################################################################################################

		for k,v in json.load(open('orca/data/json_files/Delta_properties.json')).items():
			setattr(self,k,v)

		############################################################################################     
		################################ initialize time series ####################################
		############################################################################################
		self.dmin = np.zeros(T)
		self.min_rule = np.zeros(T)
		self.gains = np.zeros(T)
		self.sodd_cvp = np.zeros(T)
		self.sodd_swp = np.zeros(T)
		self.cvp_max = np.zeros(T)
		self.swp_max = np.zeros(T)
		self.TRP_pump = np.zeros(T)
		self.HRO_pump = np.zeros(T)
		self.inflow = np.zeros(T)
		self.outflow = np.zeros(T)
		self.CVP_shortage = np.zeros(T)
		self.SWP_shortage = np.zeros(T)
		self.SWP_shortage = np.zeros(T)
		self.Delta_shortage = np.zeros(T)
		# self.x2 = np.zeros(T+1)
		# self.x2[1] = 82.0
		# self.x2[0] = 82.0

		#########################################################################################################    
		################################ initialize arrays for interpolation ####################################
		#########################################################################################################

		self.cvp_targetO = np.zeros(367)
		self.swp_targetO = np.zeros(367)
		self.cvp_pmaxO = np.zeros(367)
		self.swp_pmaxO = np.zeros(367)
		self.swp_intake_maxO = np.zeros(367)
		self.cvp_intake_maxO = np.zeros(367)

		self.san_joaquin_adj = np.zeros(367)
		self.D1641_on_off = np.zeros(367)
		self.san_joaquin_ie_used = np.zeros(367)
		# self.san_joaquin_ie_amt = np.zeros(T)
		self.omr_reqr_int = np.zeros(367)

		############################################################################################     
		################################ interpolation to fill arrays ##############################
		############################################################################################

		# for i in range(0,T):
		# 	self.san_joaquin_ie_amt[i] = np.interp(self.sanjoaquin[i]*tafd_cfs, self.san_joaquin_export_ratio['D1641_flow_target'],self.san_joaquin_export_ratio['D1641_export_limit']) * cfs_tafd

		for i in range(0,365):
			self.san_joaquin_adj[i] = np.interp(water_day(i), self.san_joaquin_add['d'], self.san_joaquin_add['mult']) * max(self.sanjoaquin[i] - 1000.0 * cfs_tafd, 0.0)
			self.san_joaquin_ie_used[i] = np.interp(water_day(i), self.san_joaquin_export_ratio['d'], self.san_joaquin_export_ratio['on_off'])
			self.omr_reqr_int[i] = np.interp(water_day(i), self.omr_reqr['d'], self.omr_reqr['flow']) * cfs_tafd

			self.cvp_targetO[i] = np.interp(i, self.pump_max['cvp']['d'], #calculate pumping target for day of year (based on target pumping for sodd) 
			                          self.pump_max['cvp']['target']) * cfs_tafd
			self.swp_targetO[i] = np.interp(i, self.pump_max['swp']['d'], 
			                          self.pump_max['swp']['target']) * cfs_tafd
			self.cvp_pmaxO[i] = np.interp(i, self.pump_max['cvp']['d'], 
			                          self.pump_max['cvp']['pmax']) * cfs_tafd #calculate pumping targets (based on max allowed pumping) based on time of year 
			self.swp_pmaxO[i] = np.interp(i, self.pump_max['swp']['d'], 
			                          self.pump_max['swp']['pmax']) * cfs_tafd
			self.swp_intake_maxO[i] = np.interp(i, self.pump_max['swp']['d'], self.pump_max['swp']['intake_limit']) * cfs_tafd
			self.cvp_intake_maxO[i] = np.interp(i, self.pump_max['cvp']['d'],self.pump_max['cvp']['intake_limit']) * cfs_tafd



	def find_release(self, dowy, d, t, wyt, orovilleAS, shastaAS, folsomAS):

		#################################################################################################################    
		################################ San Joaquin river import/export ratio constraints ##############################
		#################################################################################################################

		san_joaquin_ie = self.san_joaquin_ie_amt[t] * self.san_joaquin_ie_used[dowy]
		swp_jas_stor = (self.pump_max['swp']['pmax'][5] * cfs_tafd)/self.export_ratio[wyt][8]
		cvp_jas_stor = (self.pump_max['cvp']['pmax'][5] * cfs_tafd)/self.export_ratio[wyt][8]
    
		if dowy <= 274:
			numdaysSave = 92
		else:
			numdaysSave = 1
		if orovilleAS > numdaysSave*swp_jas_stor:
			swp_max = min(max(self.swp_intake_max[d] + self.san_joaquin_adj[d], san_joaquin_ie * 0.45), self.swp_pmax[d])
		else:
			swp_max = 0.0
		if (shastaAS + folsomAS) > numdaysSave*cvp_jas_stor:
			cvp_max = min(max(self.cvp_intake_max[d], san_joaquin_ie * 0.55), self.cvp_pmax[d])
		else:
			cvp_max = 0.0

		return cvp_max, swp_max

	def calc_flow_bounds(self, t, d, m, wyt, dowy, orovilleAS, shastaAS, folsomAS): 

		#######################################################################################################################################################    
		################################ Initial flow constraints based on Delta export ration constraints and reservoir storage ##############################
		#######################################################################################################################################################

		gains = self.netgains[t] 
		self.min_rule[t] = self.min_outflow[wyt][m-1] * cfs_tafd
		export_ratio = self.export_ratio[wyt][m-1]
		self.cvp_max[t] = self.cvp_target[d-1]*self.demand_multiplier[t]
		self.swp_max[t] = self.swp_target[d-1]*self.demand_multiplier[t]
		if d == 366:
			self.cvp_max[t] = self.cvp_target[d-2]*self.demand_multiplier[t]
			self.swp_max[t] = self.swp_target[d-2]*self.demand_multiplier[t]

			'''the sodd_* variables tell the reservoirs how much to release
			for south of delta demands only
			(dmin is the reservoir release needed to meet delta outflows)'''

		if gains > self.min_rule[t]: # extra unstored water available for pumping. in this case dmin[t] is 0
			self.sodd_cvp[t] = max((self.cvp_max[t] - 0.55*(gains - self.min_rule[t])) / export_ratio, 0) #implementing export ratio "tax"
			self.sodd_swp[t] = max((self.swp_max[t] - 0.45*(gains - self.min_rule[t])) / export_ratio, 0)
		else: # additional flow needed
			self.dmin[t] = self.min_rule[t] - gains
			'''amount of additional flow from reservoirs that does not need "export tax"
			because dmin release helps to meet the export ratio requirement'''
			Q = self.min_rule[t]*export_ratio/(1-export_ratio) 

			if self.cvp_max[t] + self.swp_max[t] < Q:
				self.sodd_cvp[t] = self.cvp_max[t]
				self.sodd_swp[t] = self.swp_max[t]
			else:
				self.sodd_cvp[t] = 0.75*Q + (self.cvp_max[t] - 0.75*Q)/export_ratio #implementing export ratio "tax"
				self.sodd_swp[t] = 0.25*Q + (self.swp_max[t] - 0.25*Q)/export_ratio

		#determining percentage of CVP sodd demands from both Shasta and Folsom
		if folsomAS > 0.0 and shastaAS > 0.0:
			self.folsomSODDPCT = folsomAS/(folsomAS + shastaAS)
		elif folsomAS < 0.0:
			self.folsomSODDPCT = 0.0
		else:
			self.folsomSODDPCT = 1.0
		self.shastaSODDPCT = 1.0 - self.folsomSODDPCT
  
	def meet_OMR_requirement(self, Tracy, Banks, t): #old and middle river requirements (hence "OMR")

		#################################################################################################  
		################################ Old and Middle river requirements ##############################
		#################################################################################################

		if Tracy + Banks > self.maxTotPump: 
			'''maxTotPump is calculated in calc_weekly_storage, before this OMR function is called. 
			current simulated puming is more that the total allowed pumping based on Delta requirements
			Tracy (CVP) is allocated 55% of available flow for pumping, Banks (SWP) is allocated 45%. 
			(assuming Delta outflow is greater than it's requirement- I still need to look into where that's determined)'''
				#Tracy is pumping less that it's maximum allocated flow. Harvery should pump less flow now. 
			if Tracy < self.maxTotPump*0.55:
				Banks = self.maxTotPump - Tracy
			elif Banks < self.maxTotPump*0.45: #Banks is pumping less that it's maximum allocated flow. Tracy should pump less flow now. 
				Tracy = self.maxTotPump - Banks
				'''in this case, both pumps would be taking their allocated percentage of flow,
				but the overall flow through the pumps is still greater than the maximum allowed'''
			else:
				Banks = self.maxTotPump*0.45
				Tracy= self.maxTotPump*0.55
		return Tracy, Banks

	def step_init(self, t, d, m, wyt, dowy, cvp_flows, swp_flows, orovilleAS, shastaAS, folsomAS):

		##################################################################################################   
		################################ initial stimulation step at time t ##############################
		##################################################################################################

		self.gains[t] = self.netgains[t] #+ sumnodds
		self.inflow[t] = max(self.gains[t] + cvp_flows + swp_flows, 0) # realinflow * cfs_tafd

		self.outflow_rule = self.min_outflow[wyt][m-1] * cfs_tafd

		self.min_rule[t] = max(self.outflow_rule, 0)
		export_ratio = self.export_ratio[wyt][m-1]

		self.cvp_max[t] = self.cvp_pmax[d-1] #max pumping allowed 
		self.swp_max[t] = self.swp_pmax[d-1]

		omrNat = self.OMR_sim[t]* cfs_tafd
		maxTotPumpInt = omrNat - self.omr_reqr_int[dowy] #- fish_trigger_adj
		self.maxTotPump = max(maxTotPumpInt,0.0)

		self.cvp_max[t], self.swp_max[t] = self.find_release(dowy, d, t, wyt, orovilleAS, shastaAS, folsomAS)
		self.cvp_max[t], self.swp_max[t] = self.meet_OMR_requirement(self.cvp_max[t], self.swp_max[t], t)

		self.required_outflow = max(self.min_rule[t], (1-export_ratio)*self.inflow[t])
		self.surplus = self.gains[t] - self.required_outflow 
		return self.surplus



	def step_pump(self, t, d, m, wyt, dowy, cvp_flows, swp_flows,surplus):

		##################################################################################################   
		################################ second stimulation step at time t ##############################
		##################################################################################################

		if surplus >= 0:
			#gains cover both the min_rule and the export ratio requirement.so, pump the full cvp/swp inflows
			self.TRP_pump[t] = max(min(cvp_flows + 0.55 * surplus, self.cvp_max[t]),0) #Tracy pumping plant, for CVP exports
			self.HRO_pump[t] = max(min(swp_flows + 0.45 * surplus, self.swp_max[t]),0) #Harvey 0. Banks pumping plant, for SWP exports
    
		else:
			'''deficit must be made up from cvp/swp flows. Assume 75/25 responsibility for these
			 (including meeting the export ratio requirement)'''

			deficit = -surplus
			cvp_pump = max(cvp_flows - 0.75 * deficit, 0)
			if cvp_pump == 0:
				swp_pump = max(swp_flows - (deficit - cvp_flows), 0)
			else:
				swp_pump = max(swp_flows - 0.25 * deficit, 0)
			self.TRP_pump[t] = max(min(cvp_pump, self.cvp_max[t]),0) #overall TRP pumping
			self.HRO_pump[t] = max(min(swp_pump, self.swp_max[t]),0) #overall HRO pumping
		if d >= 365:
			self.TRP_pump[t] = self.TRP_pump[t-1]
			self.HRO_pump[t] = self.HRO_pump[t-1]
		self.outflow[t] = self.inflow[t] - self.TRP_pump[t] - self.HRO_pump[t]

		self.CVP_shortage[t] = max(self.cvp_max[t] - self.TRP_pump[t],0)
		self.SWP_shortage[t] = max(self.swp_max[t] - self.HRO_pump[t],0)
		self.Delta_shortage[t] = max(self.min_rule[t] -self.outflow[t],0)

	def results_as_df(self, index):

		##########################################################################################  
		################################ for generating output file ##############################
		##########################################################################################

		df = pd.DataFrame()

		if self.baseline_run == False:
			names = ['SODD_CVP','SODD_SWP','SWP_shortage', 'CVP_shortage']
			things = [self.cvp_max,self.swp_max,self.SWP_shortage, self.CVP_shortage]
			for n,t in zip(names,things):
				df['%s_%s' % (self.key,n)] = pd.Series(t, index=index)

		elif self.baseline_run == True:
			names = ['in','out','TRP_pump','HRO_pump','total_pump','SODD_CVP','SODD_SWP', 'SWP_shortage', 'CVP_shortage','total_pump_shortage', 'Delta_shortage', 'Outflow_requirement']
			things = [self.inflow, self.outflow, self.TRP_pump, self.HRO_pump, self.TRP_pump + self.HRO_pump,self.cvp_max,self.swp_max, self.SWP_shortage, self.CVP_shortage,self.SWP_shortage + self.CVP_shortage, self.Delta_shortage, self.min_rule]
			for n,t in zip(names,things):
				df['%s_%s' % (self.key,n)] = pd.Series(t, index=index)
			
		return df