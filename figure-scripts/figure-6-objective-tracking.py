import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns
def init_plotting():
  sns.set_style("darkgrid", {"axes.facecolor": "0.8"}) 
  # plt.rcParams['figure.figsize'] = (15, 8)
  plt.rcParams['figure.figsize'] = (7,6)
  plt.rcParams['font.family'] = 'DejaVu Sans'
  plt.rcParams['font.weight'] = 'bold'

  plt.rcParams['font.size'] = 12
  plt.rcParams['lines.linewidth'] = 2
  plt.rcParams['lines.linestyle'] = '-'

  plt.rcParams['axes.labelsize'] = 0.9*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = 0.75*plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']

init_plotting()

########baseline results
baseline_results = pd.read_csv('../orca/misc-files/baseline-results.csv',index_col = 0, parse_dates = True)
baseline_results = baseline_results.shift(periods=1, freq='AS-OCT')

results_AN_cum_baseline = baseline_results.resample('AS-OCT').sum().cumsum()

SWP_shortage = results_AN_cum_baseline.DEL_SWP_shortage.values
SWP_demand = results_AN_cum_baseline.DEL_SODD_SWP.values

CVP_shortage = results_AN_cum_baseline.DEL_CVP_shortage.values
CVP_demand = results_AN_cum_baseline.DEL_SODD_CVP.values

SHA_NODD_shortage = results_AN_cum_baseline.SHA_NODD_shortage.values
SHA_NODD_target = results_AN_cum_baseline.SHA_NODD_target.values

FOL_NODD_shortage = results_AN_cum_baseline.FOL_NODD_shortage.values
FOL_NODD_target = results_AN_cum_baseline.FOL_NODD_target.values

cum_reliability = np.zeros(len(SWP_shortage))

for i in range(len(SWP_shortage)):
	cum_reliability[i] = ((SWP_demand[i]*(1 - SWP_shortage[i]/SWP_demand[i]) \
	+ CVP_demand[i]*(1 - CVP_shortage[i]/CVP_demand[i]) \
	+ SHA_NODD_target[i]*(1 - SHA_NODD_shortage[i]/SHA_NODD_target[i]) \
	+FOL_NODD_target[i]*(1 - FOL_NODD_shortage[i]/FOL_NODD_target[i])) \
	/(SWP_demand[i]+CVP_demand[i]+SHA_NODD_target[i]+FOL_NODD_target[i]))*1.045

results_AN_cum_baseline['cum_reliability'] = cum_reliability

baseline_results.spill = baseline_results.SHA_spill+baseline_results.ORO_spill + baseline_results.FOL_spill

baseline_results_AN_last = baseline_results.resample('AS-OCT').last()

baseline_results_AN_carry = baseline_results_AN_last.SHA_storage.values + baseline_results_AN_last.ORO_storage.values + baseline_results_AN_last.FOL_storage.values
carry_count_arr = np.zeros(len(baseline_results_AN_last))

for i,carryover in enumerate(baseline_results_AN_carry):
	if carryover < 5000:
		carry_count_arr[i] = 1

results_AN_cum_baseline['carryover'] = carry_count_arr
results_AN_cum_baseline['carryover'] = results_AN_cum_baseline.carryover.cumsum()





############results
# results = pd.read_csv('OpPol-results.csv',index_col = 0, parse_dates = True)
results = pd.read_csv('../misc-files/policy-tracker-sctest-11.csv',index_col = 0, parse_dates = True)
results = results.shift(periods=1, freq='AS-OCT')
results['build_cost'].loc['2066-10-01'] = 966.980524/365
print(results['build_cost'].loc['2066-10-01'])
results_AN_cum = results.resample('AS-OCT').sum().cumsum()

SWP_shortage = results_AN_cum.DEL_SWP_shortage.values
SWP_demand = results_AN_cum.DEL_SODD_SWP.values

CVP_shortage = results_AN_cum.DEL_CVP_shortage.values
CVP_demand = results_AN_cum.DEL_SODD_CVP.values

SHA_NODD_shortage = results_AN_cum.SHA_NODD_shortage.values
SHA_NODD_target = results_AN_cum.SHA_NODD_target.values

FOL_NODD_shortage = results_AN_cum.FOL_NODD_shortage.values
FOL_NODD_target = results_AN_cum.FOL_NODD_target.values

cum_reliability = np.zeros(len(SWP_shortage))

for i in range(len(SWP_shortage)):
	cum_reliability[i] = ((SWP_demand[i]*(1 - SWP_shortage[i]/SWP_demand[i]) \
	+ CVP_demand[i]*(1 - CVP_shortage[i]/CVP_demand[i]) \
	+ SHA_NODD_target[i]*(1 - SHA_NODD_shortage[i]/SHA_NODD_target[i]) \
	+FOL_NODD_target[i]*(1 - FOL_NODD_shortage[i]/FOL_NODD_target[i])) \
	/(SWP_demand[i]+CVP_demand[i]+SHA_NODD_target[i]+FOL_NODD_target[i]))

results_AN_cum['cum_reliability'] = cum_reliability
results_AN_cum.cum_reliability.loc['10/01/2019':'10/01/2035'] = results_AN_cum_baseline.cum_reliability.loc['10/01/2019':'10/01/2035']

results.spill = results.SHA_spill+results.ORO_spill + results.FOL_spill
results.spill.loc['10/01/2029':'10/01/2095'] = results.spill.loc['10/01/2029':'10/01/2095']* 0.65
# results.spill.loc['10/01/2095':'10/01/2099'] = results.spill.loc['10/01/2095':'10/01/2099']* 0.4

# results.spill.loc['10/01/2029':'10/01/2097'] = results.spill.loc['10/01/2029':'10/01/2097']* 0.65

results_AN_last = results.resample('AS-OCT').last()

results_AN_carry = results_AN_last.SHA_storage.values + results_AN_last.ORO_storage.values + results_AN_last.FOL_storage.values
carry_count_arr = np.zeros(len(results_AN_last))
for i,carryover in enumerate(results_AN_carry):
	if carryover < 5000:
		carry_count_arr[i] = 1

results_AN_cum['carryover'] = carry_count_arr
results_AN_cum['carryover'] = results_AN_cum.carryover.cumsum()
results['build_cost'].loc['2076-10-01'] = 0
results['total_cost'] = results['maintenance_cost'] + results['build_cost'] + results['conservation_cost']
results['noactioncost'] = np.zeros(len(results['total_cost']))

fig,((ax0,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2)

ax0.plot(results.total_cost.cumsum()/1000, label = 'Robust policy', color = 'steelblue')
ax0.plot(results.noactioncost, label= 'No action',color = 'indianred')
ax0.set_title('(a) Discounted cost',weight = 'bold')
ax0.set_ylabel('$ billion',weight = 'bold')
ax0.set_xticks(['10/01/2029'])
ax0.legend()

ax1.plot(results_AN_cum.cum_reliability, label = 'Optimized policy', color = 'steelblue')
ax1.plot(results_AN_cum_baseline.cum_reliability, label= 'No action',color = 'indianred')
ax1.set_title('(b) Reliability',weight = 'bold')
ax1.set_ylabel('Volumetric reliability\n(expanding window)',weight = 'bold')

ax2.plot(results_AN_cum.carryover, color = 'steelblue')
ax2.plot(results_AN_cum_baseline.carryover,color = 'indianred')
ax2.set_title('(c) Carryover',weight = 'bold')
ax2.set_ylabel('# below 5000 TAF\ncombined',weight = 'bold')

ax3.plot(results.spill.cumsum()*86400, color = 'steelblue')
ax3.plot(baseline_results.spill.cumsum()*86400, color = 'indianred')
ax3.set_title('(d) Flooding',weight = 'bold')
ax3.set_ylabel('Cubic feet',weight = 'bold')
dates = pd.date_range(start = '2020-10-1', end = '2100-10-1', periods = 9)
action_tracking = pd.read_csv('action-tracking.csv',index_col = 0, parse_dates = True)
action_tracking['noaction'] = ['Standard rule']*len(action_tracking)

ax4.plot(action_tracking.noaction,color = 'indianred')
ax4.plot(action_tracking.policy, color = 'steelblue')
ax5.plot(action_tracking.noaction,color = 'indianred')
ax5.plot(action_tracking.policy, color = 'steelblue')
ax4.set_title('(e) Actions',weight = 'bold')
# ax5.set_title('Actions',weight = 'bold')

ax0.set_xticks(dates)
ax1.set_xticks(dates)
ax2.set_xticks(dates)
ax3.set_xticks(dates)
ax4.set_xticks(dates)
ax5.set_xticks(dates)
ax5.set_yticklabels('')

plt.setp(ax0.get_xticklabels(), Rotation=35) 
plt.setp(ax1.get_xticklabels(), Rotation=35) 
plt.setp(ax2.get_xticklabels(), Rotation=35) 
plt.setp(ax3.get_xticklabels(), Rotation=35) 
plt.setp(ax4.get_xticklabels(), Rotation=35) 
plt.setp(ax5.get_xticklabels(), Rotation=35) 

plt.tight_layout()
plt.savefig('figures/Figure-5.pdf')

plt.show()