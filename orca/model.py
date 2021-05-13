import numpy as np
import pandas as pd
from .reservoir import *
from .delta import *
from .offstream import *
from .util import *
import json

class Model(): 

	def __init__(self, P, df, dfind, Y, min_depth, hist_datafile,SHA_baseline, ORO_baseline, FOL_baseline,baseline_run = False):
	    
	########################################################################################               
	################################ initialize time seires ################################
	########################################################################################         

		self.baseline_run = baseline_run
		self.df = df
		self.dfh = hist_datafile
		self.T = len(self.df)
		self.Y = Y
		self.sha_levee_expand = np.zeros(self.Y)
		self.oro_levee_expand = np.zeros(self.Y)
		self.fol_levee_expand = np.zeros(self.Y)
		self.sha_res_expand = np.zeros(self.Y)
		self.oro_res_expand = np.zeros(self.Y)
		self.fol_res_expand = np.zeros(self.Y)
		self.offstream_build = np.zeros(self.Y)
		self.offstream_build_cap = np.zeros(self.Y)
		self.build_cost = np.zeros(self.T)
		self.maintenance_cost = np.zeros(self.T)
		self.conservation_cost = np.zeros(self.T)
		self.YRS_fnf = df.YRS_fnf.values
		self.BND_trbt_fnf = df.BND_trbt_fnf.values
		self.levee_lag = 5
		self.res_lag = 10
		self.offstream_lag = 10
		self.offstream_1 = False
		self.offstream_2 = False
		self.offstream_3 = False
		self.demand_curt = np.zeros(self.T)
		self.pol_reverse = np.zeros(self.T)
		self.dayofyear = self.df.index.dayofyear.values
		self.month = self.df.index.month.values    
		self.year = self.df.index.year.values
		self.wyts = df['WYT_sim'].values# simulated (forecasted)
		self.rdiscount = df.rdiscount.values
		self.min_depth = min_depth
		############################################################################            
		################################ initialize objects ########################
		############################################################################    

		self.shasta = Reservoir(self.df, self.dfh, self.month, self.dayofyear, SHA_baseline, 'SHA', baseline_run = self.baseline_run)
		self.folsom = Reservoir(self.df, self.dfh, self.month, self.dayofyear, ORO_baseline, 'FOL', baseline_run = self.baseline_run)
		self.oroville = Reservoir(self.df, self.dfh, self.month, self.dayofyear, FOL_baseline,'ORO', baseline_run = self.baseline_run)
		self.offstream = Offstream(self.df, self.month, 'OFFSTREAM')
		self.reservoirs = [self.shasta, self.folsom, self.oroville]
		self.delta = Delta(self.df, 'DEL', baseline_run = self.baseline_run)
		self.indicator_arr = dfind.values

	def simulate(self,P):

	###################################################################################################         
	################################ parameters for dynamic adaptation actions ########################
	###################################################################################################  
		Levee_levels = {'Levee_1':False,'Levee_2':False,'Levee_3':False,'Levee_4':False,'Levee_5':False}
		Dam_levels = {'Dam_1':False,'Dam_2':False,'Dam_3':False}

		SHA_exeedance = self.shasta.exceedance
		ORO_exeedance = self.oroville.exceedance
		FOL_exeedance = self.folsom.exceedance

		SHA_carryover_curtail = self.shasta.carryover_curtail
		ORO_carryover_curtail = self.oroville.carryover_curtail
		FOL_carryover_curtail = self.folsom.carryover_curtail

		floodpool_shift = 'standard'

		gw_rate = 0
		cost = 0
		gw_tracker = 0
		sha_levee_tracker, oro_levee_tracker, fol_levee_tracker = 79000, 150000, 115000 #cfs, original max outflows
		sha_capacicy_tracker, oro_capacicy_tracker, fol_capacicy_tracker = 4552, 3537, 975 #TAF, original capacities
		offstream = False
		# #############################################################################################    
		# ################################ tree minimum depth penalty  ################################
		# #############################################################################################    

		if self.baseline_run == False:
			if P.get_depth() < self.min_depth:
				penalty = 10**18
			else:
				penalty = 0
		if self.baseline_run == True:
			penalty = 0

		####################################################################################################   
		################################ start looping through time series  ################################
		####################################################################################################   
		leap_year = False
		y = 0 #start at 9, for year 2010- no infrastructure unitl 2020
		policy_track = ['standard_rule']*(y+1)
		for t in range(1,self.T):
			d = self.dayofyear[t]
			if d == 0:
				leap_year = False
			dowy = water_day(d)
			if leap_year == True: 
				dowy +=1
			if dowy == 92 and doy == 92:
				leap_year = True
			doy = dowy
			m = self.month[t]

			###################################################################################  
			################################ action switching  ################################
			###################################################################################  
			if dowy == 1:
				indicators = self.indicator_arr[y]
				y += 1
				if self.baseline_run == False: 
					policy,rules = P.evaluate(indicators)
				if self.baseline_run == True: 
					policy = 'standard_rule'
				policy_track.append(policy)


					########################################################################################               
					################################ Change operating rules ################################
					########################################################################################         
				if policy == 'standard_rule':
					# self.wyt = self.wyts
					SHA_exeedance = self.shasta.exceedance
					ORO_exeedance = self.oroville.exceedance
					FOL_exeedance = self.folsom.exceedance
					SHA_carryover_curtail = self.shasta.carryover_curtail
					ORO_carryover_curtail = self.oroville.carryover_curtail
					FOL_carryover_curtail = self.folsom.carryover_curtail
					floodpool_shift = 'standard'

				elif policy == 'OpPolA':
					# self.wyt = self.wytA
					SHA_exeedance = self.shasta.exceedanceA
					ORO_exeedance = self.oroville.exceedanceA
					FOL_exeedance = self.folsom.exceedanceA
					SHA_carryover_curtail = self.shasta.carryover_curtailA
					ORO_carryover_curtail = self.oroville.carryover_curtailA
					FOL_carryover_curtail = self.folsom.carryover_curtailA
					floodpool_shift = 'policyA'

				elif policy == 'OpPolB':
					# self.wyt = self.wytB
					SHA_exeedance = self.shasta.exceedanceB
					ORO_exeedance = self.oroville.exceedanceB
					FOL_exeedance = self.folsom.exceedanceB
					SHA_carryover_curtail = self.shasta.carryover_curtailB
					ORO_carryover_curtail = self.oroville.carryover_curtailB
					FOL_carryover_curtail = self.folsom.carryover_curtailB
					floodpool_shift = 'policyB'

				##############################################################################################################################          
				################################ Conservation- will need to convert to percentages eventually ################################
				##############################################################################################################################

				if policy == 'Demand_90':
					mult = 0.9
				elif policy == 'Demand_80':
					mult = 0.8
				elif policy == 'Demand_70':
					mult = 0.7
				elif policy == 'Demand_60':
					mult = 0.6
				elif policy == 'Demand_50':
					mult = 0.5
				else:
					# sumnodds = self.sumnodds
					mult = 1

				self.delta.swp_target = self.delta.swp_targetO * mult
				self.delta.cvp_target = self.delta.cvp_targetO * mult
				self.delta.cvp_pmax = self.delta.cvp_pmaxO * mult
				self.delta.swp_pmax = self.delta.swp_pmaxO * mult
				self.delta.swp_intake_max = self.delta.swp_intake_maxO * mult
				self.delta.cvp_intake_max = self.delta.cvp_intake_maxO * mult
				self.folsom.nodds = self.folsom.nodd_base_int * mult
				self.oroville.nodds = self.oroville.nodd_base_int * mult
				self.shasta.nodds = self.shasta.nodd_base_int * mult
				#add cost here
				nodd_cost = (np.sum(self.folsom.nodd_base) + np.sum(self.shasta.nodd_base)) * self.delta.cvp_cost_tafd * (1-mult)
				swp_cost = np.sum(self.delta.swp_targetO) * self.delta.swp_cost_tafd * (1-mult)
				cvp_cost = np.sum(self.delta.swp_targetO) * self.delta.cvp_cost_tafd * (1-mult)

				self.conservation_cost[t]=(swp_cost + cvp_cost + nodd_cost)*(1/(1+self.rdiscount[t]))**(y)


				#################################################################################      
				################################ Conjunctive use ################################
				#################################################################################

				if policy == 'GW0.1':
					gw_tracker = 0.1 #taf/day
				elif policy == 'GW0.5':
					gw_tracker = 0.5 #taf/day
				elif policy == 'GW1':
					gw_tracker = 1 #taf/day
				elif policy == 'GW2':
					gw_tracker = 2 #taf/day
				elif policy == 'GW4':
					gw_tracker = 4 #taf/day

				if gw_rate < gw_tracker:
					gw_rate = gw_tracker


				#################################################################################      
				################################ Levee expansion ################################
				#################################################################################
				if any([policy =='Levee_1', policy =='Levee_2', policy =='Levee_3', policy =='Levee_4', policy =='Levee_5']):
					if Levee_levels[policy] == False:
						Levee_levels[policy] = True
						self.sha_levee_expand[y] = max(self.shasta.lev_tracker[policy],self.sha_levee_expand[y-1]) #update levee expansion time series
						self.oro_levee_expand[y] = max(self.oroville.lev_tracker[policy],self.oro_levee_expand[y-1])
						self.fol_levee_expand[y] = max(self.folsom.lev_tracker[policy],self.fol_levee_expand[y-1])
						if all([y + self.levee_lag < self.Y,self.sha_levee_expand[y] > self.shasta.max_outflow]):
							expand_cost = self.shasta.base_lev_cost + self.oroville.base_lev_cost + self.folsom.base_lev_cost # base costs
							expand_cost +=  (self.shasta.lev_cost[policy_track[y]])*(self.sha_levee_expand[y] - self.shasta.max_outflow)**self.shasta.alpha_3 #add cost for expanding capacity- diminishing returns
							expand_cost += (self.oroville.lev_cost[policy_track[y]])*(self.oro_levee_expand[y] - self.oroville.max_outflow)**self.oroville.alpha_3
							expand_cost += (self.folsom.lev_cost[policy_track[y]])*(self.fol_levee_expand[y] - self.folsom.max_outflow)**self.folsom.alpha_3
							self.build_cost[t] =  (expand_cost)*(1/(1+self.rdiscount[t]))**(y) #update cost of expansion. Extra maintenance cost added in post-processing
				if self.sha_levee_expand[y-self.levee_lag] > self.shasta.max_outflow: #if lag has passed
					self.shasta.max_outflow = self.sha_levee_expand[y-self.levee_lag] #update capcity
					self.oroville.max_outflow = self.oro_levee_expand[y-self.levee_lag]
					self.folsom.max_outflow = self.fol_levee_expand[y-self.levee_lag]


	        ############################################################################################     
	        ################################ Storage capacity expansion ################################
	        ############################################################################################

				if policy == 'Dam_1':
					if Dam_levels[policy] == False:
						Dam_levels[policy] = True
						self.sha_res_expand[y] = max(self.shasta.cap_tracker[policy],self.sha_res_expand[y-1]) #update capacity expansion time series
						self.fol_res_expand[y] = max(self.folsom.cap_tracker[policy],self.fol_res_expand[y-1])			
						expand_cost = self.shasta.cap_cost[policy_track[y]]+ self.shasta.base_cost
						expand_cost += self.folsom.cap_cost[policy_track[y]] + self.folsom.base_cost

						if all([y + self.res_lag < self.Y,self.sha_res_expand[y] > self.shasta.capacity]):
							self.build_cost[t] =  (expand_cost)*(1/(1+self.rdiscount[t]))**(y) #update cost of expansion. Extra maintenance cost added in post-processing

				if policy == 'Dam_2':
					if Dam_levels[policy] == False:
						Dam_levels[policy] = True
						self.sha_res_expand[y] = max(self.shasta.cap_tracker[policy],self.sha_res_expand[y-1]) #update capacity expansion time series
						self.fol_res_expand[y] = max(self.folsom.cap_tracker[policy],self.fol_res_expand[y-1])
						if Dam_levels['Dam_1'] == True: 						
							expand_cost = self.shasta.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_1'] + self.shasta.base_cost
							expand_cost += self.folsom.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_1'] + self.folsom.base_cost
							expand_cost += self.shasta.alpha_1*(self.shasta.alpha_2*(self.shasta.cap_tracker[policy] - self.shasta.capacity))**0.7 #add cost for incramental penalty- diminishing returns
							expand_cost += self.folsom.alpha_1*(self.folsom.alpha_2*(self.folsom.cap_tracker[policy] - self.folsom.capacity))**0.7

						else:
							expand_cost = self.shasta.cap_cost[policy_track[y]] + self.shasta.base_cost
							expand_cost += self.folsom.cap_cost[policy_track[y]] + self.folsom.base_cost

						if all([y + self.res_lag < self.Y,self.sha_res_expand[y] > self.shasta.capacity]):
							self.build_cost[t] =  (expand_cost)*(1/(1+self.rdiscount[t]))**(y) #update cost of expansion. Extra maintenance cost added in post-processing
				
				if policy == 'Dam_3':
					if Dam_levels[policy] == False:
						Dam_levels[policy] = True
						self.sha_res_expand[y] = max(self.shasta.cap_tracker[policy],self.sha_res_expand[y-1]) #update capacity expansion time series
						self.fol_res_expand[y] = max(self.folsom.cap_tracker[policy],self.fol_res_expand[y-1])
						if Dam_levels['Dam_1'] == True: 						
							expand_cost = self.shasta.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_2'] + self.shasta.base_cost
							expand_cost += self.folsom.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_2'] + self.folsom.base_cost
							expand_cost += self.shasta.alpha_1*(self.shasta.alpha_2*(self.shasta.cap_tracker[policy] - self.shasta.capacity))**self.shasta.alpha_3 #add cost for incramental penalty- diminishing returns
							expand_cost += self.folsom.alpha_1*(self.folsom.alpha_2*(self.folsom.cap_tracker[policy] - self.folsom.capacity))**self.folsom.alpha_3

						elif Dam_levels['Dam_2'] == True: 	
							expand_cost = 0					
							expand_cost += self.shasta.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_1'] + self.shasta.base_cost
							expand_cost += self.folsom.cap_cost[policy_track[y]] - self.shasta.cap_cost['Dam_1'] + self.folsom.base_cost							
							expand_cost += self.shasta.alpha_1*(self.shasta.alpha_2*(self.shasta.cap_tracker[policy] - self.shasta.capacity))**0.7 #add cost for incramental penalty- diminishing returns
							expand_cost += self.folsom.alpha_1*(self.folsom.alpha_2*(self.folsom.cap_tracker[policy] - self.folsom.capacity))**0.7

						else:
							expand_cost = self.shasta.cap_cost[policy_track[y]] + self.shasta.base_cost
							expand_cost += self.folsom.cap_cost[policy_track[y]] + self.folsom.base_cost

						if all([y + self.res_lag < self.Y,self.sha_res_expand[y] > self.shasta.capacity]):
							self.build_cost[t] =  (expand_cost)*(1/(1+self.rdiscount[t]))**(y) #update cost of expansion. Extra maintenance cost added in post-processing
				if self.sha_res_expand[y-self.res_lag] > self.shasta.capacity: #if lag has passed
					self.shasta.capacity = self.sha_res_expand[y-self.res_lag] #update capcity
					self.oroville.capacity = self.oro_res_expand[y-self.res_lag]
					self.folsom.capacity = self.fol_res_expand[y-self.res_lag]

				############################################################################################     
				################################ offstream storage capacity ################################
				############################################################################################
				if all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_1, self.offstream_1 == False, self.offstream_2 == False, self.offstream_3 == False]):
					#initialize offstream reservoir capacity
					offstream = True
					self.offstream_1 = True

					self.offstream.capacity = self.offstream.capacity_1
					self.offstream.in_capacity = self.offstream.in_capacity_1
					self.offstream.out_capacity = self.offstream.out_capacity_1
					self.build_cost[t-365*self.offstream_lag] =  (self.offstream.cost_1)*(1/(1+self.rdiscount[t]))**(y)

				elif all([policy == 'Offstream_1',self.offstream_1 == False, self.offstream_2 == False, self.offstream_3 == False]):
					self.offstream_build_cap[y] = self.offstream.capacity_1


				elif all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_2, self.offstream_1 == False, self.offstream_2 == False, self.offstream_3 == False]):
					offstream = True
					self.offstream_2 = True
					self.offstream.capacity = self.offstream.capacity_2
					self.offstream.in_capacity = self.offstream.in_capacity_2
					self.offstream.out_capacity = self.offstream.out_capacity_2
					self.build_cost[t-365*self.offstream_lag] =  (self.offstream.cost_2)*(1/(1+self.rdiscount[t]))**(y)
					#######add economies of scale

				elif all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_2, self.offstream_1 == True, self.offstream_2 == False, self.offstream_3 == False]):
					offstream = True
					self.offstream_2= True
					self.offstream.capacity = self.offstream.capacity_2
					self.offstream.in_capacity = self.offstream.in_capacity_2
					self.offstream.out_capacity = self.offstream.out_capacity_2
					self.build_cost[t-365*self.offstream_lag] = (self.offstream.cost_expand_1_2)*(1/(1+self.rdiscount[t]))**(y)


				elif all([policy == 'Offstream_2',self.offstream_2 == False, self.offstream_3 == False]):
					self.offstream_build_cap[y] = self.offstream.capacity_2

				elif all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_3, self.offstream_1 == False, self.offstream_2 == False, self.offstream_3 == False]):
					offstream = True
					self.offstream_3 = True
					self.offstream.capacity = self.offstream.capacity_3
					self.offstream.in_capacity = self.offstream.in_capacity_3
					self.offstream.out_capacity = self.offstream.out_capacity_3
					self.build_cost[t-365*self.offstream_lag] = (self.offstream.cost_3)*(1/(1+self.rdiscount[t]))**(y)

				elif all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_3, self.offstream_1 == True, self.offstream_2 == False, self.offstream_3 == False]):
					offstream = True
					self.offstream_3= True
					self.offstream.capacity = self.offstream.capacity_3
					self.offstream.in_capacity = self.offstream.in_capacity_3
					self.offstream.out_capacity = self.offstream.out_capacity_3
					self.build_cost[t] = (self.offstream.cost_expand_1_3)*(1/(1+self.rdiscount[t]))**(y)

				elif all([self.offstream_build_cap[y-self.offstream_lag] == self.offstream.capacity_3, self.offstream_2 == True, self.offstream_3 == False]):
					self.offstream.capacity = self.offstream.capacity_3
					self.offstream.in_capacity = self.offstream.in_capacity_3
					self.offstream.out_capacity = self.offstream.out_capacity_3
					self.build_cost[t-365*self.offstream_lag] = (self.offstream.cost_expand_2_3)*(1/(1+self.rdiscount[t]))**(y)
					offstream = True
					self.offstream_3 = True

				elif all([policy == 'Offstream_3', self.offstream_3 == False]): 
					self.offstream_build_cap[y] = self.offstream.capacity_3
	#offstream option 3
				#############################################################################################################################  
				################################ maintenance costs for reservoir and levee expansion, offstream  ################################
				#############################################################################################################################  
				self.maintenance_cost[t] += (self.folsom.max_outflow-150000)*self.folsom.levee_maint_cost**(y )
				self.maintenance_cost[t] += (self.oroville.max_outflow-79000)*self.oroville.levee_maint_cost**(y)
				self.maintenance_cost[t] += (self.shasta.max_outflow-115000)*self.shasta.levee_maint_cost**(y)
				self.maintenance_cost[t] += (self.folsom.capacity-975)*self.folsom.dam_maint_cost**(y)
				self.maintenance_cost[t] += (self.shasta.capacity-4552)*self.shasta.dam_maint_cost**(y)
				self.maintenance_cost[t] += (self.offstream.capacity)*self.offstream.maint_cost**(y)
				self.maintenance_cost[t] = (self.maintenance_cost[t])*(1/(1+self.rdiscount[t]))**(y)
			################################################################################     
			################################ run simulation ################################
			################################################################################

			wyt = self.wyts[t]
			### Delta values: 
			self.oroville.find_available_storage(t,d,dowy,wyt,ORO_exeedance)
			self.folsom.find_available_storage(t,d,dowy,wyt,FOL_exeedance)
			self.shasta.find_available_storage(t,d,dowy,wyt,SHA_exeedance)
			self.delta.calc_flow_bounds(t, d, m, wyt, dowy, self.oroville.available_storage[t], self.shasta.available_storage[t], self.folsom.available_storage[t])
			self.shasta.sodd_pct = self.delta.shastaSODDPCT
			self.folsom.sodd_pct = self.delta.folsomSODDPCT
			SHA_R_Delta, SHA_R_Delta_lag_1, SHA_R_Delta_lag_3, SHA_R_Delta_lag_5, SHA_av_storage = self.shasta.step(policy, t, d, m, wyt, dowy, y, SHA_exeedance, SHA_carryover_curtail, floodpool_shift, gw_rate, self.delta.dmin[t], self.delta.sodd_cvp[t])
			FOL_R_Delta, FOL_R_Delta_lag_1, FOL_R_Delta_lag_3, FOL_R_Delta_lag_5, FOL_av_storage = self.folsom.step(policy, t, d, m, wyt, dowy, y, FOL_exeedance, FOL_carryover_curtail, floodpool_shift, gw_rate, self.delta.dmin[t], self.delta.sodd_cvp[t])
			ORO_R_Delta, ORO_R_Delta_lag_1, ORO_R_Delta_lag_3, ORO_R_Delta_lag_5, ORO_av_storage = self.oroville.step(policy, t, d, m, wyt, dowy, y, ORO_exeedance, ORO_carryover_curtail, floodpool_shift, gw_rate, self.delta.dmin[t], self.delta.sodd_swp[t])
			DEL_surplus = self.delta.step_init(t, d, m, wyt, dowy, SHA_R_Delta + FOL_R_Delta, ORO_R_Delta, ORO_av_storage, SHA_av_storage, ORO_av_storage)
			
			if offstream == True:
				self.offstream.freeport_flow[t] = SHA_R_Delta_lag_5 + self.BND_trbt_fnf[t-5]+ ORO_R_Delta_lag_3 + self.YRS_fnf[t-2]* cfs_tafd + FOL_R_Delta
				self.offstream.red_bluff_flow[t] = SHA_R_Delta_lag_1 + self.BND_trbt_fnf[t-1] * cfs_tafd
				OFFSTREAM_in_lag, OFFSTREAM_out_lag = self.offstream.step(t,m,DEL_surplus)
				self.delta.step_pump(t, d, m, wyt, dowy, SHA_R_Delta + FOL_R_Delta, ORO_R_Delta, DEL_surplus - OFFSTREAM_in_lag + OFFSTREAM_out_lag)
			
			else: 
				self.delta.step_pump(t, d, m, wyt, dowy, SHA_R_Delta + FOL_R_Delta, ORO_R_Delta, DEL_surplus)

		return self.results_as_df(self.build_cost, self.maintenance_cost, self.conservation_cost), penalty, policy_track

	def results_as_df(self,build_cost,maintenance_cost,conservation_cost):

		##########################################################################################  
		################################ for generating output file ##############################
		##########################################################################################
		df = pd.DataFrame(index=self.df.index)
		if self.baseline_run == False:
			for x in [self.shasta, self.folsom, self.oroville, self.delta]:
				df = pd.concat([df, x.results_as_df(df.index)], axis=1)

		elif self.baseline_run == True:
			for x in [self.shasta, self.folsom, self.oroville, self.delta, self.offstream]:
				df = pd.concat([df, x.results_as_df(df.index)], axis=1)

		names = ['build_cost','maintenance_cost','conservation_cost']
		things = [build_cost, maintenance_cost, conservation_cost]
		for n,t in zip(names,things):
			df['%s'%(n)] = t
		# df = df.fillna(0)
		return df