# Description
Code to conduct experiments for the paper [**Modified Gauss-Newton method for solving a smooth system of nonlinear equations**](https://www.mathnet.ru/links/5c8e34f7dca934e26b7f98b8eeb713be/crm911.pdf).

## Overview

* *run_experiments.py* — main script to perform experiments;
* *oracles.py* — contains classes for optimization criteria;
* *opt_utils.py* — auxiliary functions for optimizers;
* *optimizers.py* — contains Gauss-Newton optimization algorithms;
* *benchmark_utils.py* — routines for designed experiments;
* *plotting.py* — routines for plotting results.

Print help in command line in repository directory to list all hyperparameters of the experiments:
```
    python run_experiments.py -h
```
Run the following command in command line in repository directory to obtain all experiment data in current directory:
```
    python run_experiments.py
```

## Requirements

* [NumPy](https://numpy.org/);
* [Matplotlib](https://matplotlib.org/);
* [Seaborn](https://seaborn.pydata.org/).

## References

<a id="1">[1]</a> Yudin N.E. Modified Gauss–Newton method for solving a smooth system of nonlinear equations // Computer Research and Modeling, 2021, vol. 13, no. 4, pp. 697-723, [doi: https://doi.org/10.20537/2076-7633-2021-13-4-697-723](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=crm&paperid=911&option_lang=eng).
