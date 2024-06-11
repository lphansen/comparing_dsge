# README

This repository contains the Python interface that calls C++ code used to solve the Long Run Risk
Model, based on Brunnermeier & Sannikov (2012) and Bassac & Cucco (1998).

This code has been tested with the Intel C Compiler (ICC) 17.0.4 and Eigen 3.3.4.

## License

All source code and software in this repository are made available
under the terms of the
[MIT license](https://opensource.org/licenses/mit-license.html).


## Quick Start

*Note:* These instructions are specific to the midway2
high-performance computing cluster operated by the
[Research Computing Center](http://rcc.uchicago.edu), and may need to
be adapted to work for different computing environments.

1. Connect to a midway2 login node; e.g., `ssh midway2.rcc.uchicago.edu`.

2. Load the Intel C compiler module and the Intel MKL library module:

   ```bash
   module load intel/16.0
   module load mkl/11.3
   ```

3. Load the correct Python version:

   ```bash
   module load python/3.5.2+intel-16.0
   ```

4. If you plan to load the standalone C++ program, make sure that you load the Boost package:

   ```bash
   module load boost/1.62.0
   ```

5. There is no need to load the Eigen package, as the Github repo contains the
header files of Eigen (we are currently using Eigen 3.3.4).

6. Run `make -f Makefile.shared` in the `src` directory to build the `longrunrisk`
shared library. If you'd like to run the standalone C++ program, run `make -f Makefile.test`

7. To test the Python interface, run `python test1D.py`. It would solve a 1 dimensional
model using the Python interface. To solve the same model using the standalone C++ program, run
`source runTestCPP.sh`

8. The Python interface will export the numerical solution to
folder `model0Python` and the standalone C++ program will export the numerical
solution to folder `model0`.

## Compare Results
To compare results, you only need to compare `zeta_e_final.dat` and
`zeta_h_final.dat` in the two folders, as all the variables are functions
of these two vectors.

## Credits

*Add information about contributors here.*
