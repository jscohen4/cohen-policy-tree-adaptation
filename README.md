# Dynamic adaptation with policy tree optimization

This repository contains all code corresponding to methods and figure generation in the following paper:

Cohen, J.S. & Herman, J.D., Dynamic adaptation of water resources systems under uncertainty using policy tree optimization (submitted manuscript).

This work is is build off and couples two existing repositories: [ORCA](https://github.com/jscohen4/orca) and [policy tree optimization](https://github.com/jdherman/ptreeopt). 

## Requirements:
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [Scipy](http://www.scipy.org/), [Scikit-learn](http://scikit-learn.org/), [Seaborn](https://seaborn.pydata.org/), [MPI for Python](https://mpi4py.readthedocs.io/en/stable/) (optional), [PyGraphviz](https://pygraphviz.github.io/) (optional).

## Contents
`ORCA`:

`ptreeopt`:

`misc-files`:

`snapshots`:

`SA_files`:

`figure-scripts`:

`baseline-cc.py` and `baseline-cc-parallel.py`: these scripts run baseline ORCA climate simulations, either locally on one processor or in parallel on 97 processors. The initially formated climate projection data must first be downloaded from the [orca_cmip5_inputs](https://github.com/jscohen4/orca_cmip5_inputs) repository. 

## Paper methods and figures:
The following instrutctions correspond to the methods section of the paper.

`optimization-parallel.py` performs the policy search described in Section 2.3.2: Multi-objective optimization. Optimized policies are stored as pickle files in the `snapshots` folder.

`
