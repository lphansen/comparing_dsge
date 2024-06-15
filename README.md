# Macro Finance

This repository contains the replication files for the research conducted by [Hansen, Khorrami, and Tourre (2024)](https://github.com/lphansen/macro-finance/blob/main/Comparing_DSGE_Models_June11.pdf). The files are organized into five subfolders, each corresponding to specific sections of the paper:

1. **Section 4.4:** [`1_single_capital_with_stochastic_volatility`](https://github.com/lphansen/macro-finance/tree/main/1_single_capital_with_stochastic_volatility)
2. **Section 4.5:** [`2_single_capital_with_structure_ambiguity`](https://github.com/lphansen/macro-finance/tree/main/2_single_capital_with_structure_ambiguity)
3. **Section 4.6:** [`3_heterogenous_capital_with_stochastic_volatility`](https://github.com/lphansen/macro-finance/tree/main/3_heterogenous_capital_with_stochastic_volatility)
4. **Sections 5.3.1, 5.3.2, 5.3.3:** [`4_heterogenous_agents_with_frictions_NN`](https://github.com/lphansen/macro-finance/tree/main/4_heterogenous_agents_with_frictions_NN)
5. **Section 5.3.4:** [`5_heterogenous_agents_with_frictions_FDM (mfrSuite)`](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite))

We recommend running the code on clusters; our computations were performed on [UChicago Midway3](https://rcc.uchicago.edu/midway3). To replicate the calibrations used in the paper, please refer to the environment settings and run the bash files in each subfolder sequentially. The results from each subfolder are independent and can be reviewed in the Jupyter notebook named `results.ipynb` located in each subfolder. The [`figure_replication`](https://github.com/lphansen/macro-finance/tree/main/figure_replication) subfolder contains the code used to generate all the figures in the paper.

Readers can modify the bash files and main files directly to test customized parameters. Estimated running times for each bash file are included in the readme file of each subfolder.

## Notes

- Guidance on setting up a UChicago Midway Account can be found [here](https://rcc.uchicago.edu/accounts-allocations/request-account).
- Guidance on running Jupyter Notebooks on Midway can be found [here](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/).
- We recommend configuring both Python and Julia environments before starting your work. 
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
- Please note that the [`5_heterogenous_agents_with_frictions_FDM (mfrSuite)`](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite)) subfolder requires the installation of the [_mfrSuite_](https://github.com/lphansen/macro-finance/tree/main/5_heterogenous_agents_with_frictions_FDM%20(mfrSuite)/src/mfrSuite) module before solving the model. 