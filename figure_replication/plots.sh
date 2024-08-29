#!/bin/bash

# Description: This script creates all the plots used in the paper.

module load python/anaconda-2022.05

python3 src/plot_figs_1_3.py
python3 src/plot_figs_4_6.py
python3 src/plot_figs_7_9.py
python3 src/plot_figs_10_15.py
