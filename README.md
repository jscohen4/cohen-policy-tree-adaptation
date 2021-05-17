# Dynamic adaptation with policy tree optimization

This repository contains all code corresponding to methods and figure generation in the following paper:

Cohen, J.S. & Herman, J.D., Dynamic adaptation of water resources systems under uncertainty using policy tree optimization (submitted manuscript).

This work is is build off and couples two existing repositories: [ORCA](https://github.com/jscohen4/orca) and [policy tree optimization](https://github.com/jdherman/ptreeopt). 

## Requirements:
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [Scipy](http://www.scipy.org/), [Scikit-learn](http://scikit-learn.org/), [SALIB](https://github.com/SALib/SALib), [Seaborn](https://seaborn.pydata.org/), [MPI for Python](https://mpi4py.readthedocs.io/en/stable/) (optional), [PyGraphviz](https://pygraphviz.github.io/) (optional).

## Directories:
`orca`: Contains code and input data for the simulation more used in this study: Operations of Reservoirs in California. The model is briefly described in **Section 2.2: Model**. For further infocmation on ORCA see the [original ORCA repository](https://github.com/jscohen4/orca) and [Cohen et. al., 2020](https://ascelibrary.org/doi/10.1061/%28ASCE%29WR.1943-5452.0001300).

`ptreeopt`:

`misc-files`:

`snapshots`:

`SA_files`:

`figure-scripts`:


## Paper methods and figures:
The following instrutctions correspond to the methods section of the paper.

**Section 3.1.1 Climate projections**: : cmip5 inputs

**Section 3.1.2 Water demand and land use projections**: demand

**Section 3.1.3 Scenario ensemble**: `orca/data/calc.py`

**Section 3.2.1 Indicators**: `orca/data/calc_indicators.py`

**Section 3.2.2 Actions**: various scripts

**Section 3.2.3 Multi-objective optimization**`:optimization-parallel.py` performs the policy search: Multi-objective optimization. Optimized policies are stored as pickle files in the `snapshots` folder.

**Section 3.3.1 Robusness testing**: `testing-outofsample.py` and `robustness-calculation.py`. Figure 3 script

**Section 3.3.2 Policy dynamics**: figure 4, 5 and 6 scripts

**Secction 3.4.1 Action occurance in robust policiess**: figure 7 scripts

**Section 3.4.2 Sensitivity to cost estimates**: SA files

**Section 3.4.3 Action timing**: action hits and figure 8

**Section 3.4.4 Indicator occurance in robust policies**: figure 10 and 11
