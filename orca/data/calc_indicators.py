import numpy as np 
import scipy.stats as sp
import pandas as pd
from .util import *
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.datasets import load_iris
from .write_json import modify
import json
import matplotlib.pyplot as plt
import pickle

# import matplotlib.pyplot as plt
# calc WYT and 8RI. add columns to datafile from cdec_scraper.
# confirm against http://cdec.water.ca.gov/cgi-progs/iodir/WSIHIST
cfsd_mafd = 2.29568411*10**-5 * 86400 / 10 ** 6

cfs_tafd = 2.29568411*10**-5 * 86400 / 1000
tafd_cfs = 1000 / 86400 * 43560

def get_wday(s,timing_quantile): #use to get quantiles for flow timing. use args when calling with apply. df.resample('AS-OCT').apply(get_wday,quantile = (0.3))
  total = s.sum()
  cs = s.cumsum()
  day = s.index[cs > timing_quantile*total][0]
  return water_day(day)


def process_indicators(df):
  df = daily_df(df)
  ann_df = df.resample('AS-OCT').first()
  return ann_df

def daily_df(df): #used to process climate projection data
  dfi = indicator_calculation(df.index, df['RES_fnf'],df['RES_swe'])
  for ind in dfi.columns:
    df[ind] = dfi[ind]
  return df


def indicator_calculation(index, fnf, swe):
  indicators = json.load(open('orca/data/json_files/indicators_whole.json'))
  tnums = {'1D':365,'3D':122,'1M':12,'3M':4,'1Y':1}
  dfind = pd.DataFrame(index = index)
  for i in indicators:
    ind = indicators[i]
    
    if ind['type'] == 'fnf_annual':
      if ind['delta'] == 'no':
        if ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').sum().rolling(ind['window']).mean()
        elif ind['stat'] == 'sig': 
          dfind[i] = fnf.resample('AS-OCT').sum().rolling(ind['window']).std()
      else: 
        if ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').sum().rolling(ind['window']).mean().pct_change(periods=ind['delta'])
        elif ind['stat'] == 'sig': 
          dfind[i] = fnf.resample('AS-OCT').sum().rolling(ind['window']).std().pct_change(periods=ind['delta'])
  
    elif ind['type'] == 'fnf_timescale':
      if ind['delta'] == 'no':
        dfind[i]=fnf.resample(ind['scale']).sum().rolling(ind['window']*tnums[ind['scale']]).quantile(ind['pct']).resample('AS-OCT').last()/1000
      else:
        dfind[i]=fnf.resample(ind['scale']).sum().rolling(ind['window']*tnums[ind['scale']]).quantile(ind['pct']).resample('AS-OCT').last().pct_change(periods=ind['delta'])/1000

    elif ind['type'] == 'fnf_timing':
      if ind['delta'] == 'no':
        if ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').apply(get_wday,timing_quantile = ind['tim']).rolling(ind['window']).mean()
        elif ind['stat'] == 'sig': 
          dfind[i] = fnf.resample('AS-OCT').apply(get_wday,timing_quantile = ind['tim']).rolling(ind['window']).std()
      else:
        if ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').apply(get_wday,timing_quantile = ind['tim']).rolling(ind['window']).mean().pct_change(periods=ind['delta'])
        elif ind['stat'] == 'sig': 
          dfind[i] = fnf.resample('AS-OCT').apply(get_wday,timing_quantile = ind['tim']).rolling(ind['window']).std().pct_change(periods=ind['delta'])

    elif ind['type'] == 'swe_max':
      if ind['delta'] == 'no':
        if ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').max().rolling(ind['window']).mean()
        elif ind['stat'] == 'mu':
          dfind[i] = fnf.resample('AS-OCT').max().rolling(ind['window']).std()
      else:
        if ind['stat'] == 'mu':
          dfind[i] = swe.resample('AS-OCT').max().rolling(ind['window']).mean().pct_change(periods=ind['delta'])
        elif ind['stat'] == 'mu':
          dfind[i] = swe.resample('AS-OCT').max().rolling(ind['window']).std().pct_change(periods=ind['delta'])
  dfind = dfind.ffill( )
  return dfind