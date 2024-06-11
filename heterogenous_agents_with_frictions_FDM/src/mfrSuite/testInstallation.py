##################################################################################
## This file tests, one by one, whether the packages are installed successfully ##
## If not, it will prompt a message                                             ##
##################################################################################

import subprocess
import sys

## Step 0: Get list of packages

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'mfr-sem' in installed_packages:
    print('mfr.sem was installed successfully.')
else:
    print('========================================================\n' \
    'ERROR: mfr.sem WAS NOT INSTALLED SUCCESSFULLY.\nRe-run \'pip install ./src/sem\' in the command line and \nlook for explanation in documentation on potential errors (section 2.1.4).\n'
    '========================================================\n')

if 'mfr-sdm' in installed_packages:
    print('mfr.sdm was installed successfully.')
else:
    print('========================================================\n' \
    'ERROR: mfr.sdm WAS NOT INSTALLED SUCCESSFULLY.\nRe-run \'pip install ./src/sdm\' in the command line and \nlook for explanation in documentation on potential errors (section 2.1.4).\n'
    '========================================================\n')

if 'mfr-modelSoln' in installed_packages:
    print('mfr-modelSoln was installed successfully.')
else:
    print('========================================================\n' \
    'ERROR: mfr.modelSoln WAS NOT INSTALLED SUCCESSFULLY.\nRe-run \'pip install ./src/modelSoln\' in the command line and \nlook for explanation in documentation on potential errors (section 2.1.4).\n'
    '========================================================\n')

if 'mfr-jupyterWidgets' in installed_packages:
    print('mfr-jupyterWidgets was installed successfully.')
else:
    print('========================================================\n' \
    'ERROR: mfr.jupyterWidgets WAS NOT INSTALLED SUCCESSFULLY.\nRe-run \'pip install ./src/jupyterWidgets\' in the command line and \nlook for explanation in documentation on potential errors (section 2.1.4).\n'
    '========================================================\n')

if 'modelSolnCore' in installed_packages:
    print('modelSolnCore was installed successfully.')
else:
    print('========================================================\n' \
    'ERROR: modelSolnCore WAS NOT INSTALLED SUCCESSFULLY.\nRe-run \'pip install ./src/modelSolnCore\' in the command line and \nlook for explanation in documentation on potential errors (section 2.1.4).\n'
    '========================================================\n')
