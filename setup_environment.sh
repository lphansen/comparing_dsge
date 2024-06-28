#!/bin/bash

# Description: This script sets up the Python and Julia environments.

# Step 1: Set up Python environment
echo "Setting up Python environment..."
cd figure_replication
python -m pip install -r requirements.txt
echo "Python environment setup complete."
cd ..

# Step 2: Set up Julia environment
echo "Setting up Julia environment..."
cd 3_heterogenous_capital_with_stochastic_volatility
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
echo "Julia environment setup complete."
cd ..

# Step 3: Install mfrSuite
echo "Setting up mfrSuite..."
cd "5_heterogenous_agents_with_frictions_FDM (mfrSuite)/src/mfrSuite"
bash install.sh
cd ../../..
echo "mfrSuite setup complete."

echo "Environment setup finished."
