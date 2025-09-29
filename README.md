# Our Paper

This repository contains the codebase for our experiments our paper. It is based on [FSNET](https://github.com/salesforce/fsnet) repository. Please refer to this repo to get the datasets. The synthetic datasets we used are saved in the [`main/data/DATA`](main/data/DATA/) folder.

## Reproducing the Results

To reproduce the experimental results, run the main scripts located in [`main/scripts`](main/scripts).
Make sure to execute all relevant scripts to reproduce the full set of results.
The list of possible arguments can be found in [`main/main.py`](main/main.py) file. For our experiments, we used mainly the default values of the parameters, except for the ones that change in the scripts and the ones in the [`best_configs`](best_configs/) folder.
Note that running all the experiments take a lot of time. If you only want to test the code, make sure to change the list of configurations in the bash file.

---

## Analyzing Results

After running the experiments, you can analyze and evaluate the results using the provided notebooks:

- [`analyze_res.ipynb`](analyze_res.ipynb):  
  Contains the analysis code, including computation of key metrics such as **MASE** and other evaluation statistics.

- [`test_outliers.ipynb`](test_outliers.ipynb):  
  Used to analyze results on **synthetic data** and to generate some of the **figures** presented in the paper.

- [`test_regimes.ipynb`](test_regimes.ipynb):  
  Also used for **synthetic data analysis** and producing **figures** from the paper related to regime-change behavior.
The outputs we obtained on the synthetic datasets are already present in the [`synthetic_output_runs`](synthetic_output_runs/) folder. An example of the results obtained after a run is in the [`result_example`](result_example/) folder.


---

## Environment Setup

All dependencies and environment specifications are provided in the [`OCL_TS.yml`](OCL_TS.yml) file.

