#!/bin/bash
echo off
echo =======================================================================================
echo The MFR suite
echo Questions: please contact Macro-Financial Modeling \(MFM\) Team at Becker Friedman Institute
echo URL: https://bfi.uchicago.edu/mfm
echo =======================================================================================

echo Starting installation process...
echo ===============================================================================
echo Step 0: Install numba and pybind11
pip install pybind11
pip install numba

echo ===============================================================================
echo Step 1: Install toolboxes to compute shock elasticities and stationary density
pip install ./src/sdm
pip install ./src/sem

echo ===============================================================================
echo Step 2: Install model solution core
pip install ./src/cppCore

echo ===============================================================================
echo Step 3: Install model solution
pip install ./src/modelSoln

echo ===============================================================================
echo Step 4: Install jupyter notebook widgets
pip install ./src/jupyterWidgets

echo ===============================================================================
echo Step 5: Check installation
python testInstallation.py
