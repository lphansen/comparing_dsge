/*
This file belongs to a set of C++ files that serve as the model solution for
Hansen, Khorrami, and Tourre (working paper). The C++ program is not meant to be
called directly but is connected to a Python interface. Advanced users can
modify the C++ program and call it without Python.

Any questions/suggestions, please contact
Joe Huang:       jhuang12@uchicago.edu
Paymon Khorrami: paymon@uchicago.edu
Fabrice Tourre:  fabrice@uchicago.edu
*/


/******************************************************/
/* This file contains all the libraries and external  */
/* functions needed to preload. It should be included */
/* in all files.                                      */
/******************************************************/

#ifndef common_h
#define common_h

/*****************************************************/
/* Basic header files                                */
/*****************************************************/

#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <ctime>
#include <stack>
#include <chrono>
#include <iostream>
#include <cmath>
// #include <omp.h>
#include <string>
using namespace std::chrono;

/*****************************************************/
/* Eigen libraries                                   */
/*****************************************************/

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
//typedef Eigen::SparseMatrix<double, Eigen::RowMajor > SpMat;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor > SpMat;

typedef Eigen::Triplet<double> T;
#include <typeinfo>
#include "SparseExtra"
using Eigen::nbThreads;


// Handling name spaces
using namespace std;


/********************************************************/
/* Derivative functions                                 */
/********************************************************/

// Compute (n choose k)
int choose(int n, int k);

#endif /* common_h */
