# Dynamic adaptation with policy tree optimization

This repository contains all code corresponding to methods and figure generation in the following paper:

Cohen, J.S. & Herman, J.D., Dynamic adaptation of water resources systems under uncertainty using policy tree optimization (submitted manuscript).

This work is is an extension of two existing repositories: [ORCA](https://github.com/jscohen4/orca) and [policy tree optimization](https://github.com/jdherman/ptreeopt). 

## Requirements:
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [Scipy](http://www.scipy.org/), [Scikit-learn](http://scikit-learn.org/), [SALIB](https://github.com/SALib/SALib), [Seaborn](https://seaborn.pydata.org/), [MPI for Python](https://mpi4py.readthedocs.io/en/stable/) (optional), [PyGraphviz](https://pygraphviz.github.io/) (optional).

## Directories:
`orca`: Contains code and input data for the simulation more used in this study: Operations of Reservoirs in California. The model is briefly described in **2.2 Model**. For further infocmation on ORCA see the [original ORCA repository](https://github.com/jscohen4/orca) and [Cohen et. al., 2020](https://doi.org/10.1061/(ASCE)WR.1943-5452.0001300). Additions to the original model for this study are described in **Paper methods and figures** in this README file.

`ptreeopt`: Contains code for multi-objective policy tree optimization, described in **3.2.3 Multi-objective optimization**. For further details see the [original policy tree optimization repository](https://github.com/jdherman/ptreeopt) and [Herman and Giuliani, 2018](https://doi.org/10.1016/j.envsoft.2017.09.016). The repository has been extended to run more efficiently in parallel, via lines 144-150 in `ptreeopt/opt.py`. 

`misc-files`: Directory containing several data files required for running several python scripts to generate results and figures.

`snapshots`: Directory containing results of multi-objective policy tree optimization. These include the policies and objective values generated in **3.2.3** and analyzed in **3.3 Policy analysis**, **3.4 Actions and indicator analysis** and **4 Results and Discussion**.

`SA_files`: Directory containing python scripts and data used for **3.4.2 Sensitivity to cost estimates** and **4.3 Cost sensitivity**. 

`figure-scripts`: Directory containing python scripts to generate **Figures 3-11**.


## Paper methods and figures:
The following instrutctions correspond to subsections in **Section 3 Methods**, and their corrsponding subsections in **Section 4 Results and discussion**.

**3.1.1 Climate projections**: : The original input climate data files should first be obtained from the repository [jscohen4/orca_cmip5_inputs](https://github.com/jscohen4/orca_cmip5_inputs) and the directory `input_climate_files` put in the directory `orca/data`. To process climate data, run `baseline-cc-parallel.py` remotely on 97 processors or `baseline-cc.py` on 1 processor. To ensure that climate data is processed to be input to ORCA, ensure that `calc_indices = True` and `climate_forecasts = True` in these scripts. This will cause the sript to run `orca/data/calc_indices.py` and `orca/data/forecasting.py`. The original data for [USBR CMIP5 climate and hydrology projections](https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html) are also publically available. 


**3.1.2 Water demand and land use projections**: [Demand data](https://drive.google.com/drive/folders/1w8r_4D7e96Yw6I-GvURalE3cw84Yn-Ma?usp=sharing) should be added as `orca/data/demand_files`. The demand data is further processed by running one of the baseline simulation scripts with `tree_input_files = True` set. Original demand data is also publically available for [USGS FORE-SCE CONUS](https://www.sciencebase.gov/catalog/item/5b96c2f9e4b0702d0e826f6d), [USGS LUCAS California](https://www.sciencebase.gov/catalog/item/587fb408e4b085de6c11f389), and 
[DOE GCAM CONUS](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1216). 

**3.1.3 Scenario ensemble**: The full scenarios necessary for baseline simulations and policy-tree optimization runs can be obtained by running one of the two baseline scripts with the following options set: `calc_indices = True`, `climate_forecasts = True`, `tree_input_files = True`, `indicator_data_file = True`. Setting `simulation = True` will also run the baseline simulations. 

**3.2.1 Indicators**: Indicators are calculated from climate projections, demand data, and randomly generated discount rates in `orca/data/calc_indicators.py`. To ensure indicators are correctly processed for simulation or optimization, run one of the baseline scripts with  `calc_indices = True` set. 

**3.2.2 Actions**: various scripts

**3.2.3 Multi-objective optimization**:`optimization-parallel.py` performs the policy search over the 235 testing scenarios. Optimized policies and objective values are stored as pickle files in the `snapshots` directory. The number of processors used for this optimization must be equal to the population size (line 111 in )
 
**3.3.1 Robusness testing**: `testing-outofsample.py` and `robustness-calculation.py`. Figure 3 script

**3.3.2 Policy dynamics**: figure 4, 5 and 6 scripts

**3.4.1 Action occurance in robust policiess**: figure 7 scripts

**3.4.2 Sensitivity to cost estimates**: SA files

**3.4.3 Action timing**: action hits and figure 8

**3.4.4 Indicator occurance in robust policies**: figure 10 and 11
