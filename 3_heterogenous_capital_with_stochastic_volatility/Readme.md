# Heterogenous Capital with Stochastic Volatility

The subfolder solves heterogenous capital production economies with stochastic volatility in section 4.6.

## Scripts Overview

1. **run_1.sh**: solve for the $\rho=1$ case in two dimensional model without stochastic volatility
   - **main_twocapitals_two_dimensions.jl**: Solves the two dimensional model without stochastic volatility using Finite Diffence.
      - **utils_twocapitals_two_dimensions.jl**: Contains source files for HJB equation computations.
2. **run_2.sh**: use $\rho=1$ result as preload to solve the case with $\rho = 0.67$ and $\rho = 1.5$ in two dimensional model without stochastic volatility
3. **run_3.sh**: use two dimensional model without stochastic volatility as preload to solve three dimensional model with stochastic volatility 
   - **main_twocapitals_three_dimensions.jl**: Solves the three dimensional model with stochastic volatility using Finite Diffence.
      - **utils_twocapitals_three_dimensions.jl**: Contains source files for HJB equation computations.
4. **run_4.sh**:
   - **main_pde_shock_elasticity.py**: Solve the investment-output ratio, consumption shock elasticity using Finite Diffence.
         - **utils_pde_shock_elasticity.py**: Contains source files for continuous-time shock elasticity PDE computations.
         - **utils_FDM.py**: Computes finite differences for input data across various dimensions and orders.
5. **results.ipynb**: Load model solutions, plot stationary density and shock elasticities.

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

## Estimated Running Time
1. **run_1.sh**: < 2 hours
2. **run_2.sh**: < 4 hours
3. **run_3.sh**: < 7 hours
4. **run_4.sh**: < 20 mins

Each task was tested on a single core of Intel Xeon Gold 6248R using the parameters in the bash files, with multiple tasks run simultaneously.