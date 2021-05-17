# Dynamic adaptation with policy tree optimization

This repository contains all code corresponding to methods and figure generation in the following paper:

Cohen, J.S. & Herman, J.D., Dynamic adaptation of water resources systems under uncertainty using policy tree optimization (submitted manuscript).

This work is is build off and couples two existing repositories: [ORCA](https://github.com/jscohen4/orca) and [policy tree optimization](https://github.com/jdherman/ptreeopt). 

## Requirements:
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [Scipy](http://www.scipy.org/), [Scikit-learn](http://scikit-learn.org/), [SALIB](https://github.com/SALib/SALib), [Seaborn](https://seaborn.pydata.org/), [MPI for Python](https://mpi4py.readthedocs.io/en/stable/) (optional), [PyGraphviz](https://pygraphviz.github.io/) (optional).

## Directories:
`orca`: Contains code and input data for the simulation more used in this study: Operations of Reservoirs in California. The model is briefly described in **2.2 Model**. For further infocmation on ORCA see the [original ORCA repository](https://github.com/jscohen4/orca) and [Cohen et. al., 2020](https://doi.org/10.1061/(ASCE)WR.1943-5452.0001300).

`ptreeopt`: Contains code for multi-objective policy tree optimization, described in **3.2.3 Multi-objective optimization**. For further details see the [original policy tree optimization repository](https://github.com/jdherman/ptreeopt) and [Herman and Giuliani, 2018](https://doi.org/10.1016/j.envsoft.2017.09.016)

`misc-files`: Directory containing several data files required for running several python scripts to generate results and figures.

`snapshots`: Directory containing results of multi-objective policy tree optimization. These include the policies and objective values generated in **3.2.3** and analyzed in **3.3 Policy analysis**, **3.4 Actions and indicator analysis** and **4 Results and Discussion**.

`SA_files`: Directory containing python scripts and data used for **3.4.2 Sensitivity to cost estimates** and **4.3 Cost sensitivity**. 

`figure-scripts`: Directory containing python scripts to generate **Figures 3-11**.


## Paper methods and figures:
The following instrutctions correspond to subsections in **Section 3 Methods**, and their corrsponding subsections in **Section 4 Results and discussion**.

**3.1.1 Climate projections**: : cmip5 inputs

**3.1.2 Water demand and land use projections**: demand

**3.1.3 Scenario ensemble**: `orca/data/calc.py`

**3.2.1 Indicators**: `orca/data/calc_indicators.py`

**3.2.2 Actions**: various scripts

**3.2.3 Multi-objective optimization**`:optimization-parallel.py` performs the policy search: Multi-objective optimization. Optimized policies are stored as pickle files in the `snapshots` folder.

**3.3.1 Robusness testing**: `testing-outofsample.py` and `robustness-calculation.py`. Figure 3 script

**3.3.2 Policy dynamics**: figure 4, 5 and 6 scripts

**3.4.1 Action occurance in robust policiess**: figure 7 scripts

**3.4.2 Sensitivity to cost estimates**: SA files

**3.4.3 Action timing**: action hits and figure 8

**3.4.4 Indicator occurance in robust policies**: figure 10 and 11
