import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns

def init_plotting():
  sns.set_style("darkgrid", {"axes.facecolor": "0.8"}) 
  # plt.rcParams['figure.figsize'] = (15, 8)
  plt.rcParams['figure.figsize'] = (9,7)
  plt.rcParams['font.family'] = 'DejaVu Sans'
  plt.rcParams['font.weight'] = 'bold'

  plt.rcParams['font.size']  = 12
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = 0.85*plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 0.85*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
init_plotting()



###############levee
fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharex = True)

dfcount = pd.read_csv('../misc-files/Levee_distributionsexpansion.csv', index_col = 0)
# dfcount['Levee1'] = dfcount['Levee1']*2
Levee_count = {}



levee1_count = []
dfcount1adj = dfcount['Levee5']
dfcount['Levee1'] = dfcount1adj

for i,action_count in enumerate(dfcount['Levee1'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		levee1_count.append(year-2)
Levee_count['Levee 1'] = levee1_count

levee2_count = []
dfcount2adj = dfcount['Levee2'] 
dfcount['Levee2'] = dfcount2adj

for i,action_count in enumerate(dfcount['Levee2'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		levee2_count.append(year-2)
Levee_count['Levee 2'] = levee2_count

levee3_count = []
dfcount3adj = dfcount['Levee3'] 
dfcount['Levee3'] = dfcount3adj*1.27

for i,action_count in enumerate(dfcount['Levee3'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		levee3_count.append(year-2)
Levee_count['Levee 3'] = levee3_count

dfcount4adj = dfcount['Levee4'] 

levee4_count = []
for i,action_count in enumerate(dfcount['Levee4'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		levee4_count.append(year-2)
Levee_count['Levee 4'] = levee4_count

dfcount5adj = dfcount['Levee5']

levee5_count = []
for i,action_count in enumerate(dfcount['Levee5'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		levee5_count.append(year-2)
Levee_count['Levee 5'] = levee5_count
sns.histplot(Levee_count,ax = ax0, bins =16,legend=True,element="poly",fill = True, alpha = 0.15)

################offstream
dfcount = pd.read_csv('../misc-files/Offstream_distributionsexpand.csv', index_col = 0)
Offstream_count = {}
dfcount3adj = dfcount['Sites3']
dfcount['Sites3'] = dfcount['Sites3']

dfcount2adj = dfcount['Sites2']
dfcount['Sites2'] = dfcount['Sites2']


Offstream1_count = []
for i,action_count in enumerate(dfcount['Sites1'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Offstream1_count.append(year-2)
Offstream_count['Offstream 1'] = Offstream1_count

Offstream2_count = []
for i,action_count in enumerate(dfcount['Sites2'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Offstream2_count.append(year-2)
Offstream_count['Offstream 2'] = Offstream2_count

Offstream3_count = []
for i,action_count in enumerate(dfcount['Sites3'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Offstream3_count.append(year-2)
Offstream_count['Offstream 3'] = Offstream3_count
Offstream3_adj  = Offstream_count['Offstream 3']
# Offstream3_adj = Offstream_count
sns.histplot(Offstream_count,ax = ax1, bins = 16,element="poly",fill = True, alpha = 0.15)
# sns.kdeplot(Offstream_count['Offstream 1'])
# sns.kdeplot(Offstream_count['Offstream 2'])
# sns.kdeplot(Offstream_count['Offstream 3'])
# for key in Levee_count:
	# sns.histplot(Levee_count[key],bins = 8)

##########Demand
dfcount = pd.read_csv('../misc-files/demand_distributions.csv', index_col = 0)
Demand_count = {}

dfcount70adj = dfcount['Demand70']
Demand70_count = []
for i,action_count in enumerate(dfcount['Demand70'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Demand70_count.append(year-2)
Demand_count['Demand 70'] = Demand70_count

dfcount80adj = dfcount['Demand80']

Demand80_count = []
for i,action_count in enumerate(dfcount['Demand80'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Demand80_count.append(year-2)
Demand_count['Demand 80'] = Demand80_count

dfcount90adj = dfcount['Demand90']

Demand90_count = []
for i,action_count in enumerate(dfcount['Demand90'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		Demand90_count.append(year-2)
Demand_count['Demand 90'] = Demand90_count


sns.histplot(Demand_count,ax = ax2, bins =16,element="poly",fill = True, alpha = 0.15)



##########oppol


dfcount = pd.read_csv('../misc-files/OpPol_distributions.csv', index_col = 0)
OpPol_count = {}

OpPolA_count = []
dfcountAadj = dfcount['OpPolA']

for i,action_count in enumerate(dfcount['OpPolA'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		OpPolA_count.append(year-2)
OpPol_count['Hedging A'] = OpPolA_count

OpPolB_count = []
dfcountBadj = dfcount['OpPolB']
for i,action_count in enumerate(dfcount['OpPolB'].values):
	year = i+2020
	for j in range(0,int(action_count)):
		OpPolB_count.append(year-2)
OpPol_count['Hedging B'] = OpPolB_count

sns.histplot(OpPol_count,ax = ax3,bins =16,element="poly",fill = True, alpha = 0.15)
# for key in Levee_count:
	# sns.histplot(Levee_count[key],bins = 8)
ax0.set_title('(a) Levee', weight = 'bold')
ax1.set_title('(b) Offstream storage', weight = 'bold')
ax2.set_title('(c) Demand', weight = 'bold')
ax3.set_title('(d) Hedging', weight = 'bold')

ax3.set_xlabel('')
ax2.set_xlabel('')
ax0.set_ylabel('Count',weight = 'bold')
ax1.set_ylabel('Count',weight = 'bold')
ax2.set_ylabel('Count',weight = 'bold')
ax3.set_ylabel('Count',weight = 'bold')
# ax0.set_xticks(np.arange(2020,2100,10))
# plt.setp(ax0.get_xticklabels(), Rotation=35) 
# plt.setp(ax1.get_xticklabels(), Rotation=35) 
# plt.setp(ax2.get_xticklabels(), Rotation=35) 
# plt.setp(ax3.get_xticklabels(), Rotation=35) 
ax0.xaxis.set_tick_params(labelbottom=True)
ax1.xaxis.set_tick_params(labelbottom=True)

plt.tight_layout()
plt.savefig('figures/Figure-9.pdf')

plt.show()



