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


/**********************************************************/
/**********************************************************/
/* This file contains the header file of the model class. */
/* It will be the class interfaced in Python              */
/**********************************************************/
/**********************************************************/


#ifndef model_h
#define model_h

#include "common.h"
#include "Parameters.h"
#include "derivs.h"
#include "stateVars.h"
#include "valueVars.h"
#include "Vars.h"
#include "matrixVars.h"
#include "Python.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <stdio.h>



class model{
public:
    /*******************************************/
    /*      Objects needed in a model          */
    /*******************************************/

    Parameters parameters;
    matrixVars matrix_vars;
    stateVars  state_vars;
    valueVars  value_vars;
    Vars       vars;
    derivs     derivsXiE;
    derivs     derivsXiH;
    derivs     derivsLogQ;
    derivs     derivsQ;
    derivs     derivsKappa;
    derivs     derivsLogABar;

    Eigen::ArrayXd xiEGuess;
    Eigen::ArrayXd xiHGuess;
    Eigen::ArrayXd chiGuess;
    Eigen::ArrayXd kappaGuess;

    int status;

    std::vector<double> timeItersVec;
    std::vector<double> timeItersLinSysVec;
    std::vector<double> eErrorsVec;
    std::vector<double> hErrorsVec;
    std::vector<int> cgEIters;
    std::vector<int> cgHIters;

    /*******************************************/
    /*      Objects used to store data         */
    /*******************************************/

    // Derivatives

    // First partials
    Eigen::MatrixXd derivsXiE_first;
    Eigen::MatrixXd derivsXiH_first;
    Eigen::MatrixXd derivsQ_first;
    Eigen::MatrixXd derivsLogQ_first;

    // Second partials
    Eigen::MatrixXd derivsXiE_second;
    Eigen::MatrixXd derivsXiH_second;
    Eigen::MatrixXd derivsQ_second;
    Eigen::MatrixXd derivsLogQ_second;

    // Cross partials
    // Drifts of state
    Eigen::MatrixXd muX;

    Eigen::MatrixXd derivsXiE_cross;
    Eigen::MatrixXd derivsXiH_cross;
    Eigen::MatrixXd derivsQ_cross;
    Eigen::MatrixXd derivsLogQ_cross;

    // Vols of state
    Eigen::MatrixXd sigmaX;
    /*******************************************/
    /*      Methods                            */
    /*******************************************/

    // Constructor
    model(int numSds, double sigma_K_norm, double sigma_Z_norm, double sigma_V_norm,
          double sigma_H_norm, int logW, double wMin, double wMax,
          int nDims, int nWealth, int nZ, int nV, int nH, int nShocks,
          int verbatim, string folderName, string preLoad, int method,
          double dt, double dtInner, int maxIters, int maxItersInner,
          double tol, double innerTol, int equityIss, int hhCap, int iparm_2,
          int iparm_3, int iparm_28, int iparm_31, double lambda_d, double nu_newborn,
          double lambda_Z, double lambda_V, double lambda_H, double Z_bar,
          double V_bar, double H_bar, double delta_e, double delta_h,
          double a_e, double a_h, double rho_e, double rho_h, double phi,
          double alpha_K, double gamma_e, double gamma_h, double chiUnderline,
          double cov11, double cov12, double cov13, double cov14,
          double cov21, double cov22, double cov23, double cov24,
          double cov31, double cov32, double cov33, double cov34,
          double cov41, double cov42, double cov43, double cov44, int exportFreq,
          Eigen::ArrayXd xiEGuessInput, Eigen::ArrayXd xiHGuessInput, Eigen::ArrayXd chiGuessInput,
          Eigen::ArrayXd kappaGuessInput, double, int);

    // Function to solve model
    int solveModel();

    // Function to handle data
    int organizeData();

    // Function to dump data
    void dumpData();

    // Function to smooth data (in the case where capital misallocation is possible)
    void smoothDataCPP(Eigen::ArrayXd, Eigen::ArrayXd);

    void reset(int numSds, double sigma_K_norm, double sigma_Z_norm, double sigma_V_norm,
          double sigma_H_norm, int logW, double wMin, double wMax,
          int nDims, int nWealth, int nZ, int nV, int nH, int nShocks,
          int verbatim, string folderName, string preLoad, int method,
          double dt, double dtInner, int maxIters, int maxItersInner,
          double tol, double innerTol, int equityIss, int hhCap, int iparm_2,
          int iparm_3, int iparm_28, int iparm_31, double lambda_d, double nu_newborn,
          double lambda_Z, double lambda_V, double lambda_H, double Z_bar,
          double V_bar, double H_bar, double delta_e, double delta_h,
          double a_e, double a_h, double rho_e, double rho_h, double phi,
          double alpha_K, double gamma_e, double gamma_h, double chiUnderline,
          double cov11, double cov12, double cov13, double cov14,
          double cov21, double cov22, double cov23, double cov24,
          double cov31, double cov32, double cov33, double cov34,
          double cov41, double cov42, double cov43, double cov44, int exportFreq,
               Eigen::ArrayXd xiEGuessInput, Eigen::ArrayXd xiHGuessInput, Eigen::ArrayXd chiGuessInput,
               Eigen::ArrayXd kappaGuessInput, double, int);

    /*********************************/
    /*********************************/
    // Passing data to python        //
    /*********************************/
    /*********************************/

    /*********************************/
    // Data that does not depend on # state variables and # shocks

    //// (1) Value functions, policy functions, constraints
    Eigen::ArrayXd &getXi_e() { return value_vars.xi_e; }
    Eigen::ArrayXd &getXi_h() { return value_vars.xi_h; }
    Eigen::ArrayXd &getKappa() { return value_vars.kappa; }
    Eigen::ArrayXd &getChi() { return value_vars.chi; }
    Eigen::ArrayXd &getBetaE() { return vars.beta_e; }
    Eigen::ArrayXd &getBetaH() { return vars.beta_h; }
    Eigen::ArrayXd &getChatE() { return vars.cHat_e; }
    Eigen::ArrayXd &getChatH() { return vars.cHat_h; }
    Eigen::ArrayXd &CeOverCh() { return vars.CeOverCh; }
    Eigen::ArrayXd &getCoverI() { return vars.CoverI; }
    Eigen::ArrayXd &getIoverK() { return vars.IoverK; }
    Eigen::ArrayXd &getCoverK() { return vars.CoverK; }
    Eigen::ArrayXd &getIoverY() { return vars.IoverY; }
    Eigen::ArrayXd &getCoverY() { return vars.CoverY; }

    //// (2) Prices and interest rates
    Eigen::ArrayXd &getQ() { return vars.q; }
    Eigen::ArrayXd &getR() { return vars.r; }
    Eigen::ArrayXd &getDeltaE() { return vars.deltaE; }
    Eigen::ArrayXd &getDeltaH() { return vars.deltaH; }
    Eigen::ArrayXd &getI() { return vars.I; }

    //// (3) Drifts
    Eigen::ArrayXd &getMuQ()  { return vars.muQ; }
    Eigen::ArrayXd &getMuK()  { return vars.muK; }
    Eigen::ArrayXd &getMuY()  { return vars.muY; }
    Eigen::ArrayXd &getMuRe() { return vars.muRe; }
    Eigen::ArrayXd &getMuRh() { return vars.muRh; }


    //// (4) The rest
    Eigen::ArrayXd &getLeverage() { return value_vars.leverageExperts; }
    Eigen::ArrayXd &getMuC() { return vars.muC; }
    Eigen::ArrayXd &getMuPhi() { return vars.muPhi; }
    Eigen::ArrayXd &getMuCe() { return vars.muCe; }
    Eigen::ArrayXd &getMuCh() { return vars.muCh; }
    Eigen::ArrayXd &getMuSe() { return vars.muSe; }
    Eigen::ArrayXd &getMuSh() { return vars.muSh; }
    Eigen::ArrayXd &getPiETilde() { return vars.piETilde; }
    Eigen::ArrayXd &getPiHTilde() { return vars.piHTilde; }
    /*********************************/
    // Data that depends on # state variables only

    //// (1) Derivatives

    Eigen::MatrixXd &getderivsXiE_first() { return derivsXiE_first; }
    Eigen::MatrixXd &getderivsXiH_first() { return derivsXiH_first; }
    Eigen::MatrixXd &getDerivsQ_first() { return derivsQ_first; }
    Eigen::MatrixXd &getDerivsLogQ_first() { return derivsLogQ_first; }

    Eigen::MatrixXd &getderivsXiE_second() { return derivsXiE_second; }
    Eigen::MatrixXd &getderivsXiH_second() { return derivsXiH_second; }
    Eigen::MatrixXd &getDerivsQ_second() { return derivsQ_second; }
    Eigen::MatrixXd &getDerivsLogQ_second() { return derivsLogQ_second; }

    Eigen::MatrixXd &getderivsXiE_cross() { return derivsXiE_cross; }
    Eigen::MatrixXd &getderivsXiH_cross() { return derivsXiH_cross; }
    Eigen::MatrixXd &getDerivsQ_cross() { return derivsQ_cross; }
    Eigen::MatrixXd &getDerivsLogQ_cross() { return derivsLogQ_cross; }

    //// (2) Drifts

    Eigen::MatrixXd &getMuX() { return muX; }

    /*********************************/
    // Data that depends on # shocks only

    Eigen::MatrixXd &getPiE() { return vars.PiE; }
    Eigen::MatrixXd &getPiH() { return vars.Pi; }
    Eigen::MatrixXd &getSigmaQ() { return vars.sigmaQ; }
    Eigen::MatrixXd &getSigmaR() { return vars.sigmaR; }
    Eigen::MatrixXd &getSigmaK() { return vars.sigmaK; }
    Eigen::MatrixXd &getSigmaC() { return vars.sigmaC; }
    Eigen::MatrixXd &getSigmaCe() { return vars.sigmaCe; }
    Eigen::MatrixXd &getSigmaCh() { return vars.sigmaCh; }
    Eigen::MatrixXd &getSigmaSe() { return vars.sigmaSe; }
    Eigen::MatrixXd &getSigmaSh() { return vars.sigmaSh; }
    Eigen::MatrixXd &getSigmaY() { return vars.sigmaY; }
    Eigen::MatrixXd &getSigmaPhi() { return vars.sigmaPhi; }


    /*********************************/
    // Data that depends on # state variables and # shocks
    Eigen::MatrixXd &getSigmaX() { return sigmaX; }

    /*********************************/
    // State variables data

    Eigen::ArrayXd &getW() { return state_vars.omega; }
    Eigen::ArrayXd &getLogW() { return state_vars.logW; }
    Eigen::ArrayXd &getZ() { return state_vars.Z; }
    Eigen::ArrayXd &getV() { return state_vars.V; }
    Eigen::ArrayXd &getH() { return state_vars.H; }


};


#endif /* model_h */
