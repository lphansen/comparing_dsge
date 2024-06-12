# Macro Finance

This repository contains the replication files for the research by Hansen, Khorrami, and Tourre (2024). The files are organized into five subfolders, each corresponding to specific sections of the paper:

1. **Section 4.4:** `single_capital_with_stochastic_volatility`
2. **Section 4.5:** `single_capital_with_structure_ambiguity`
3. **Section 4.6:** `heterogenous_capital_with_stochastic_volatility`
4. **Sections 5.3.1, 5.3.2, 5.3.3:** `heterogenous_agents_with_frictions_NN`
5. **Section 5.3.4:** `heterogenous_agents_with_frictions_FDM (mfrSuite)`

We recommend running the code on clusters. To replicate the calibrations used in the paper, please run the bash files in each subfolder sequentially. The results from each subfolder are independent and can be reviewed in the Jupyter notebook named `results.ipynb` located in each subfolder. The `heterogenous_agents_with_frictions_FDM (mfrSuite)` subfolder requires the installation of the _mfrSuite_ module before solving the model (this module is included in the subfolder). The `figure_replication` subfolder contains the code used to generate all the figures in the paper.

Readers can modify the bash files and main files directly to test customized parameters.
