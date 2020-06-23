# COVID19 synthetic control analysis
This repository performs a synthetic control analysis on COVID19 data, and the repo includes tslib which has the base synthetic control functions. The jupyter notebook COVID_clean.ipynb has sample functions that show the loading, cleaning, and analysis of the COVID data.

There is a wrapper function called synth\_control\_predictions() which calls the underlying synthetic control library function.


There are different datasets used for this analysis which are not included to this repository for space constraints, once the load function is executed the latet copy of the datasets is downloaded and a local copy kept.

![Alt text](https://github.com/niloofarbayat/COVID19-synthetic-contro-analysis/blob/master/Data%20Hierarchy.png?raw=true "Data Hierarchy")

