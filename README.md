Mixed Cell Population Identification Package
=======================
[![PyPI version](https://badge.fury.io/py/pyphenopop.svg)](https://badge.fury.io/py/pyphenopop)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7323577.svg)](https://doi.org/10.5281/zenodo.7323577)



This package contains methods designed to determine the existance of 
subpopulations with different responses to a given drug from a dataset of 
screenings with that drug on the total cell population. The implementation is
based on the method presented in the article 
["Phenotypic deconvolution in heterogeneous cancer cell populations using drug screening data"](https://doi.org/10.1101/2022.01.17.476604).
The dataset should contatin cell viability data tested on C different drug concentrations over 
N time points and containing R replicates. The package can then estimate the 
number of subpopulations, their mixture proportions and a dose-reponse curve 
for each of the subpopulations. 

## Install
The package can be easily install via `pip install pyphenopop`. You can also install it from the [Github repository](https://github.com/ocbe-uio/pyPhenoPop) using 

`pip install git+https://github.com/ocbe-uio/pyPhenoPop.git` 

or by cloning the repository

`git clone https://github.com/ocbe-uio/pyPhenoPop` 

and installing from the local repository via

`pip install .`

## Usage

A tutorial using data from the original [publication](https://doi.org/10.1101/2022.01.17.476604) is provided in `examples/tutorial.ipynb`. Additional information can be obtained by executing

```
from pyphenopop.mixpopid import mixture_id
help(mixture_id)
