import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import call
import random
import json
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import pickle
from matplotlib import cm

seed_policy_adjust = json.load(open('nondom-tracker/seed_policy_adjust.json'))
basline_ind = json.load('misc-files/basline_testing_scenarios.json')

robust_dist = np.array([])
robust_pols_dicts = {}
for seed in range(10): #9 also
	robust_seed_pols = []
	optrun = 'training_scenarios_seed_%s'%seed
	snapshots = pickle.load(open('snapshots/%s.pkl'%optrun, 'rb'))
	f = snapshots['best_f'][-1]
	P = snapshots['best_P'][-1]

	if seed == 0:
		f_all = f[seed_policy_adjust['%s'%seed]]
	else:
		f_all = np.concatenate((f_all, f[seed_policy_adjust['%s'%seed]]))
	for j,pol_num in enumerate(seed_policy_adjust['%s'%seed]):
		df = pd.DataFrame()
		for scset in range(5):
			dfscset = pd.read_csv('testing_outputs/ind_sc/seed_%s_pol_%s_scset_%s.csv'%(seed,pol_num,scset), index_col = 0)
			df = pd.concat([df, dfscset], axis=1, sort=False)
		meets_baseline = 0
		for sc in df.columns:
			baseline = basline_ind[sc]
			opt = df[sc].values
			if (-1*opt[1] > baseline[1]*0.9) & (opt[2]  < baseline[2]*1.3) & (opt[3] < baseline[3]) & (opt[0] < 900):
				meets_baseline +=1
		score = meets_baseline/(47*5)
		robust_dist = np.append(robust_dist,score)
		if score > 0.8:
			P[pol_num].graphviz_export('trees/nondom/seed_%s/pol_%s.pdf'%(seed,pol_num))
			robust_seed_pols.append(pol_num)
	robust_pols_dicts[seed] = robust_seed_pols	
with open('nondom-tracker/seed_policy_adjust_robust.json', 'w') as fp:
	json.dump(robust_pols_dicts, fp, indent=4)
