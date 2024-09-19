Mixed Cell Population Identification Package
=======================
[![PyPI version](https://badge.fury.io/py/pyphenopop.svg)](https://badge.fury.io/py/pyphenopop)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7323577.svg)](https://doi.org/10.5281/zenodo.7323577)
[![codecov](https://codecov.io/gh/ocbe-uio/pyPhenoPop/graph/badge.svg?token=fihRjmYkMy)](https://codecov.io/gh/ocbe-uio/pyPhenoPop)


This package contains methods designed to determine the existance of
subpopulations with different responses to a given drug from a dataset of
screenings with that drug on the total cell population. The implementation is
based on the method presented in the article
["Phenotypic deconvolution in heterogeneous cancer cell populations using drug screening data"](https://doi.org/10.1016/j.crmeth.2023.100417).
The dataset should contatin cell viability data tested on C different drug concentrations over
N time points and containing R replicates. The package can then estimate the
number of subpopulations, their mixture proportions and a dose-reponse curve
for each of the subpopulations. A Matlab package of the method is available at https://github.com/ocbe-uio/PhenoPop.

## Install
The package can be easily install via `pip install pyphenopop`. You can also install it from the [Github repository](https://github.com/ocbe-uio/pyPhenoPop) using

`pip install git+https://github.com/ocbe-uio/pyPhenoPop.git`

or by cloning the repository

`git clone https://github.com/ocbe-uio/pyPhenoPop`

and installing from the local repository via

`pip install .`

## Usage

A tutorial using data from the original [publication](https://doi.org/10.1016/j.crmeth.2023.100417) is provided in [`examples/tutorial.ipynb`](https://github.com/ocbe-uio/pyPhenoPop/blob/main/examples/tutorial.ipynb). Additional information can be obtained by executing

```
from pyphenopop.mixpopid import mixture_id
help(mixture_id)
```

## Publication

When using pyPhenoPop in your project, please cite
* Köhn-Luque, A., Myklebust, E. M., Tadele, D. S., Giliberto, M., Schmiester, L., Noory, J., ... & Foo, J. (2023). Phenotypic deconvolution in heterogeneous cancer cell populations using drug-screening data. Cell Reports Methods, 3(3).
```
@article{kohn2023phenotypic,
  title={Phenotypic deconvolution in heterogeneous cancer cell populations using drug-screening data},
  author = {Alvaro Köhn-Luque and Even Moa Myklebust and Dagim Shiferaw Tadele and Mariaserena Giliberto and Leonard Schmiester and Jasmine Noory and Elise Harivel and Polina Arsenteva and Shannon M. Mumenthaler and Fredrik Schjesvold and Kjetil Taskén and Jorrit M. Enserink and Kevin Leder and Arnoldo Frigessi and Jasmine Foo},
  journal={Cell Reports Methods},
  volume={3},
  number={3},
  year={2023},
  doi={https://doi.org/10.1016/j.crmeth.2023.100417},
  publisher={Elsevier}
}

```
