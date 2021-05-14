import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
import random
from mpi4py import MPI
import math
import json
def dominates(a, b):
	return (np.all(a <= b) and np.any(a < b))

def pareto_sort(P):
	N = len(P)
	keep = np.ones(N, dtype=bool) # all True to start

	for i in range(N):
		for j in range(i+1,N):
			if keep[j] and dominates(P[i,:], P[j,:]):
				keep[j] = False

			elif keep[i] and dominates(P[j,:], P[i,:]):
				keep[i] = False

	return P[keep,:]

seed_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust.json'))
count = 0
for i in range(0,10):
		for j in range(len(seed_policy_adjust['%s'%i])):
			if count == 0:
				seed_pol_arr = [[i,j]]
			else:
				seed_pol_arr = np.concatenate((seed_pol_arr,[[i,j]]),axis = 0)
			count+= 1
for sens in range(0,1000):
	nondom_pol_array = []
	sensitivity_objectives = np.zeros([len(seed_pol_arr),4])
	for i,seedpol in enumerate(seed_pol_arr[0:len(seed_pol_arr)]):
		seed = seedpol[0]
		pol_num = seedpol[1]
		df = pd.read_csv('SA_files/SA_testing_outputs/SA_testing_seed_%s_pol_%s.csv'%(seed,pol_num), index_col = 0)
		tr = np.transpose(df.values)
		objs = tr[sens]
		sensitivity_objectives[i] = objs
	sort = pareto_sort(sensitivity_objectives)
	for i in range(0,len(sort)):
		pols = np.where(sensitivity_objectives[:,0] == sort[i][0]) 
		if len(pols) == 1:
			nondom_pol_array.append(pols[0][0])
	nondom_pol_array = np.array(nondom_pol_array)
	np.save('SA_files/SA-nondom-policies/params_%s.npy'%sens,nondom_pol_array)