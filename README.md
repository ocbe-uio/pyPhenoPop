Mixed Cell Population Identification Package
=======================
This package contains methods designed to determine the existance of 
subpopulations with different responses to a given drug from a dataset of 
screenings with that drug on the total cell population. The dataset should 
contatin cell viability data tested on C different drug concentrations over 
N time points and containing R replicates. The package can then estimate the 
number of subpopulations, their mixture proportions and a dose-reponse curve 
for each of the subpopulations.
 
This version of the package contains a single module with following functions:
-   rateexpo
-   popexpo
-   objective
-   mixtureID

It also contains a dictionary 'bounds_expo', which is used by mixtureID by 
default. You can find a detailed discription in function documentation.
      
Some of the functions are model specific. This version only has an exponential 
model option, more are going to be added in the future. You can contribute to
the project by adding other models and bounds options.
      
Find all package files on: https://github.com/Apollinaria45/mixpopid
