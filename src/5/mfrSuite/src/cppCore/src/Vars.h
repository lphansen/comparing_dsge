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

#ifndef Vars_h
#define Vars_h

#include <stdio.h>
#include "derivs.h"
#include "stateVars.h"
#include "valueVars.h"
#include "common.h"
#include "Parameters.h"


class Vars {

public:

    //capital price and investment rate
    Eigen::ArrayXd q; Eigen::ArrayXd q_old; Eigen::ArrayXd qStar;
    Eigen::ArrayXd logQ;
    Eigen::ArrayXd oneOmegaQ;
    Eigen::ArrayXd omegaQ;
    Eigen::ArrayXd I;

    //Volatilities
    Eigen::MatrixXd sigmaK;
    Eigen::MatrixXd sigmaQ;
    Eigen::MatrixXd sigmaR;

    std::map <string, Eigen::MatrixXd> sigmaXMap;
    std::map <string, Eigen::ArrayXd>  muXMap;

    //Drifts
    Eigen::ArrayXd muK;
    Eigen::ArrayXd muQ;
    Eigen::MatrixXd muX;
    Eigen::ArrayXd muRe;
    Eigen::ArrayXd muRh;


    Eigen::ArrayXd normR2;
    Eigen::MatrixXd Pi; Eigen::MatrixXd PiE;
    Eigen::ArrayXd piHTilde; Eigen::ArrayXd piETilde;
    Eigen::ArrayXd deltaE;
    Eigen::ArrayXd deltaEStar;
    Eigen::ArrayXd deltaH;
    Eigen::ArrayXd deltaE_last;
    Eigen::ArrayXd trace;
    Eigen::ArrayXd r;
    Eigen::ArrayXd cHat_e;
    Eigen::ArrayXd cHat_h;
    Eigen::ArrayXd CeOverCh;
    Eigen::ArrayXd beta_e;
    Eigen::ArrayXd beta_h;
    Eigen::ArrayXd betaEDeltaE;
    Eigen::ArrayXd betaHDeltaH;

    Eigen::MatrixXd Dx;
    Eigen::ArrayXd DzetaOmega;
    Eigen::ArrayXd DzetaX;

    Eigen::ArrayXd muCe; Eigen::MatrixXd sigmaCe; Eigen::ArrayXd muSe; Eigen::MatrixXd sigmaSe;
    Eigen::ArrayXd muCh; Eigen::MatrixXd sigmaCh; Eigen::ArrayXd muSh; Eigen::MatrixXd sigmaSh;
    Eigen::ArrayXd muC; Eigen::MatrixXd sigmaC;
    Eigen::ArrayXd muY; Eigen::ArrayXd muLogA; Eigen::ArrayXd muPhi;
    Eigen::MatrixXd sigmaY;
    Eigen::MatrixXd sigmaLogY; Eigen::MatrixXd sigmaLogA; Eigen::MatrixXd sigmaPhi;
    Eigen::ArrayXd traceE; Eigen::ArrayXd traceH; Eigen::ArrayXd traceKappa; Eigen::ArrayXd traceQ;
    Eigen::ArrayXd traceLogA;
    Eigen::ArrayXd CeOverC; Eigen::ArrayXd ChOverC;
    Eigen::ArrayXd aBar; Eigen::ArrayXd logABar;
    Eigen::ArrayXd IoverK;
    Eigen::ArrayXd CoverI; Eigen::ArrayXd CoverK; Eigen::ArrayXd IoverY; Eigen::ArrayXd CoverY;

    Eigen::MatrixXd idenMat; Eigen::VectorXd derivs_temp; Eigen::MatrixXd sigmaX_temp;

    int totalCrossNum;
    int k;

    Vars();
    Vars(stateVars & , Eigen::ArrayXd, Parameters &);
    void updateVars(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                    derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters);
    void updateSigmaPi(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                       derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters);
    void updateMuAndR(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                      derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters);
    void updateDeltaEtAl(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                         derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters);
    void updateDerivs(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                      derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters);
    void updateRest(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH,
                    derivs & derivsLogQ, derivs & derivsQ, derivs & derivsKappa, derivs & derivsLogABar, Parameters & parameters);
};


#endif /* Vars_hpp */
