from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
from .util import *

class Offstream():

	def __init__(self, df, month, key):

		######################################################################################################               
		################################ initialize time seires and parameters################################
		######################################################################################################         

		T = len(df)
		self.key = key
		self.storage = np.zeros(T)
		self.inflow = np.zeros(T)
		self.outflow = np.zeros(T)
		for k,v in json.load(open('orca/data/json_files/Offstream_properties.json')).items():
			setattr(self,k,v)
		self.month = month
		self.BND_trbt_fnf = df.BND_trbt_fnf.values #sac river tributary flow between Keswich Dam and Red Bluff
		self.BND_trbt_roll = df.BND_trbt_roll.values  #pulse protection uses 3 day rolling averages
		self.freeport_flow = np.zeros(T)
		self.red_bluff_flow = np.zeros(T)
		self.YRS_fnf = df.YRS_fnf.values
		self.pump_allowed = np.zeros(T,dtype=bool)
		self.pulse_protection = np.zeros(T,dtype=bool)
		self.pulse_protection_delay = np.zeros(T,dtype=bool) #for 3 daays prior to initiation of pulse protection period
		self.month_pulse_done =False


	def step(self,t,m,surplus):
		#################################################################################################          
		################################ run offstream simulation for time t ################################
		################################################################################################# 

		if all([self.freeport_flow[t]*2 >= self.freeport_constraint[m-1], self.red_bluff_flow[t]/cfs_tafd > self.red_bluff_constraint]): 
		#Freeport flow constrait based on month, and red bluff constraint. Both must be satisfied for inflow to offstream reservoir. 
			if surplus >= 0: #only pump if surplus exists
				self.pulse_constraint(t) #find pulse constraint
				if self.pump_allowed[t] == True:
					self.inflow[t] = max(min(surplus,self.in_capacity*cfs_tafd,self.capacity - self.storage[t-1],self.BND_trbt_fnf[t]),0)

		elif all([m in [7,8,9,10,11], surplus < -15]): #if major deficit, release from offstream reservoir

			 self.outflow[t] = max(min(-surplus,self.out_capacity*cfs_tafd,self.storage[t-1]),0)
		
		else: 			
			self.pump_allowed[t] = False
			self.pulse_protection[t] = False
			self.pulse_protection_delay[t] = False
			self.inflow[t] = 0
		#update offstream reservoir storage	 
		self.storage[t] = self.storage[t-1] + self.inflow[t] - self.outflow[t]
		return self.inflow[t-3], self.outflow[t-3]
	
	def pulse_constraint(self, t):

		###########################################################################################################################          
		################################ Constraints for pulse protection on sac river tributaries ################################
		###########################################################################################################################   

		m = self.month[t]
		if self.month[t] != self.month[t-1]:
			self.month_pulse_done = False

 		#end of seven day continuous pulse protection period
		if (self.pulse_protection[t-lag] == True for lag in [1,2,3,4,5,6,7]) and (self.month_pulse_done == False):
				self.pump_allowed[t] = True
				self.pulse_protection[t] = False
				self.pulse_protection_delay[t] = False
				self.month_pulse_done = True

		#pulse protection ends if 3 day average flow exeeds 25000 cfs
		elif self.BND_trbt_roll[t] > 25000: #pulse flow constraint
			self.pump_allowed[t] = True # pumping allowed
			self.pulse_protection[t] = False # no pulse protection (see USBR NODOS feasibility report)
			self.pulse_protection_delay[t] = False

		#crossing 15000 cfs threshold to initiate 3 days prior to potential pulse protection period
		elif all([self.BND_trbt_roll[t] >= 15000, self.BND_trbt_roll[t-1] < 15000, self.month_pulse_done == False]): #potential pulse flow occuring
				self.pump_allowed[t] = True
				self.pulse_protection[t] = False
				self.pulse_protection_delay[t] = True

		#end of 3 days prior to initiation of pulse protection. Pulse protection begins
		elif all([self.pulse_protection_delay[t-1] == True, self.pulse_protection_delay[t-3] == True,self.BND_trbt_roll[t] >= 15000]): 
				self.pump_allowed[t] = False
				self.pulse_protection[t] = True
				self.pulse_protection_delay[t] = False
		
		# during 3 days prior to initiation of pulse protection period
		elif all([self.pulse_protection_delay[t-1] == True, self.BND_trbt_roll[t] >= 15000]):
				self.pump_allowed[t] = True
				self.pulse_protection[t] = False
				self.pulse_protection_delay[t] == True

		# flow too low for pulse pulse protection
		elif self.BND_trbt_roll[t] < 15000:  
				self.pump_allowed[t] = True
				self.pulse_protection[t] = False
				self.pulse_protection_delay[t] = False
		
		# during pulse protection period 
		elif self.pulse_protection[t-1] == True: 
				self.pump_allowed[t] = False
				self.pulse_protection[t] = True
				self.pulse_protection_delay[t] = False

	def results_as_df(self, index):

		##########################################################################################  
		################################ for generating output file ##############################
		##########################################################################################

		df = pd.DataFrame()
		names = ['storage','in','out']
		things = [self.storage,self.inflow, self.outflow]
		for n,t in zip(names,things):
			df['%s_%s' % (self.key,n)] = pd.Series(t, index=index)
		return df


