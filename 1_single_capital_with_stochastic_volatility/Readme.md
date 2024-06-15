# Single Capital with Stochastic Volatility

The subfolder solves single capital production economies with stochastic volatility in section 4.4.

## Scripts Overview

1. **run_1.sh**: solve for the $\rho=1$ case
    - **main_onecapital.jl**: Solves the three dimensional model using Finite Difference.
        - **utils_onecapital.jl**: Contains source files for HJB equation computations.
    - **main_pde_shock_elasticity.py**: Solve the investment-output ratio, consumption, uncertainty price shock elasticity using Finite Difference.
        - **utils_pde_shock_elasticity.py**: Contains source files for continuous-time shock elasticity PDE computations.
        - **utils_FDM.py**: Computes finite differences for input data across various dimensions and orders.
2. **run_2.sh**: Use $\rho=1$ result as preload to solve the case with $\rho = 0.67$ and $\rho = 1.5$
3. **results.ipynb**: Load model solutions, plot shock elasticities.

## Generated Directory Structure

Running the above bash scripts organizes the outputs and logs into specific directories to streamline troubleshooting and monitoring of script execution:

1. **job-outs**
   - Contains all log and error files associated with the script runs.
2. **bash**
   - Contains sbatch files for each parameter set, which are used to submit jobs to a computing cluster.
3. **output**
   - Contains the computed model solutions and other outputs.
        - ***res.npz**: Contains state variables, control variables, stationary densities, important derivatives, etc.
        - **elasticity_logimo.npz**: Contains the investment-output ratio shock elasticity
        - **elasticity_logc.npz**: Contains the consumption shock elasticity
        - **uncertainty_priceelas.npz**: Contains the uncertainty price shock elasticity

## Estimated Running Time
1. **run_1.sh**: < 3 mins
2. **run_2.sh**: < 5 mins

Each task was tested on a single core of Intel Xeon Gold 6248R using the parameters in the bash files, with multiple tasks run simultaneously.