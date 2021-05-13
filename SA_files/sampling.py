import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import latin
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import random
		############################################################################            
		################################ initialize objects ########################
		############################################################################   
np.random.seed(0)
problem = {
  'num_vars': 5,
  'names': ['Levee', 'Dam', 'Offstream', 'Demand', 'GW'],
  'bounds': [[0.8, 1.2]]*5
}

param_values = latin.sample(problem, 10)
np.save('latin_samples_2.npy', param_values)