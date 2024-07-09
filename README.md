# Macro Finance

This repository contains the replication files for the research conducted by [Hansen, Khorrami, and Tourre (2024)](https://github.com/lphansen/macro-finance/blob/main/Comparing_DSGE_Models_June11.pdf). The files are organized into five subfolders, each corresponding to specific sections of the paper:

1. **Section 4.4:** [`1_single_capital_with_stochastic_volatility`](https://github.com/lphansen/macro-finance/tree/main/1_single_capital_with_stochastic_volatility)
2. **Section 4.5:** [`2_single_capital_with_structure_ambiguity`](https://github.com/lphansen/macro-finance/tree/main/2_single_capital_with_structure_ambiguity)
3. **Section 4.6:** [`3_heterogenous_capital_with_stochastic_volatility`](https://github.com/lphansen/macro-finance/tree/main/3_heterogenous_capital_with_stochastic_volatility)
4. **Sections 5.3.1, 5.3.2, 5.3.3:** [`4_heterogenous_agents_with_frictions_NN`](https://github.com/lphansen/macro-finance/tree/main/4_heterogenous_agents_with_frictions_NN)
5. **Section 5.3.4:** [`5_heterogenous_agents_with_frictions_FDM (mfrSuite)`](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite))

We recommend running the code on clusters; our computations were performed on [UChicago Midway3](https://rcc.uchicago.edu/midway3). To replicate the calibrations used in the paper, please refer to the environment settings and run the bash files in each subfolder sequentially. The results from each subfolder are independent and can be reviewed in the Jupyter notebook named `results.ipynb` located in each subfolder. The [`figure_replication`](https://github.com/lphansen/macro-finance/tree/main/figure_replication) subfolder contains the code used to generate all the figures in the paper.

Estimated running times for each bash file are included in the readme file of each subfolder.

## Notes: Quick Start 

### Connect to Uchicago Midway Server

- Guidance on setting up a UChicago Midway Account can be found [here](https://rcc.uchicago.edu/accounts-allocations/request-account). Please change the following line in the bash script to your account:
  ```bash
  #SBATCH --account=pi-lhansen
  ```
- Guidance on running Jupyter Notebooks on Midway can be found [here](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/).

- Git clone the project in your local folder. 

### Set up the environment

- Please note that the [`5_heterogenous_agents_with_frictions_FDM (mfrSuite)`](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite)) subfolder requires the installation of the [_mfrSuite_](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite)/src/mfrSuite) module before solving the model. 
- We recommend configuring both Python and Julia environments before starting your work. 
  - **One-Step Setup**: Run the `setup_environment.sh` script to set up all environments and `mfrSuite` in one step.

  If you would prefer setting up each environment separately:
  - **Python**: Use the `requirements.txt` file to set up your Python environment. Run the following command:
    ```bash
    python -m pip install -r requirements.txt
    ```
  - **Julia**: Use the provided `Manifest.toml` and `Project.toml` files to set up your Julia environment. Run the following commands in the Julia REPL:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```
  
### Replicate the result in each subfolder 
[To Edit] Before you customize parameters, we recommend you to first test the code in each subfolder to make sure your environment is correctly set up. For example, for the first subfolder:

1. Run the following code in your terminal: 
cd 1_single_capital_with_stochastic_volatility
bash run_1.sh
Once this has finished running, run:
bash run_2.sh

the error messages will be saved in job-outs folder. 

2. Then run results.ipynb to view the plots.

Please see the detailed instruction in subfolders' Readme file. 
It is important that you do not run each bash file until the previous bash file has finished running.
E.g., only execute run_3.sh once run_2.sh has successfully completed.



### Test customized parameters

# Simple customization
In the first subfolder `1_single_capital_with_stochastic_volatility`, there is a bash file `simple.sh` which can be used
to test a single specification in the first environment. More details can be found in the `Readme.md` for that environment.


# Further customization
- To test customized parameters, a quick way is to modify the arrays at the beginning of the bash script. For example, consider `run_1.sh` in [`1_single_capital_with_stochastic_volatility`](https://github.com/lphansen/macro-finance/tree/main/1_single_capital_with_stochastic_volatility):
  ```bash
  Deltaarray=(1.0)
  gammaarray=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
  deltaarray=(0.01)
  rhoarray=(0.67 1.0 1.5)
  alphaarray=(0.0922)

  for Delta in ${Deltaarray[@]}; do
    for delta in ${deltaarray[@]}; do
      for rho in ${rhoarray[@]}; do
        for gamma in ${gammaarray[@]}; do
          for alpha in ${alphaarray[@]}; do
            ...
              action_name="large_test"
            ...
          end
        end
      end
    end
  end
  ```
  This script runs every combination of `gamma` from 1.0 to 8.0 and `rho` equal to 0.67, 1.0, and 1.5, given delta equal to 0.01 and alpha equal to 0.0922.
  - You can also add other parameters you want to iterate over into the bash files and use `ArgParse` (for Julia) or `argparse` (for Python) to pass them to the scripts.

- If you encounter issues with scripts not being executable, please run
  ```bash
  chmod +x script_name.sh
  ```


