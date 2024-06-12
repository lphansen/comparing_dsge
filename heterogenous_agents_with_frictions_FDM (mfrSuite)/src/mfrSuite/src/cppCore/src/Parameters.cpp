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


#include "common.h"
#include "Parameters.h"

Parameters::Parameters() {

}

void Parameters::save_output() {
    std::ofstream out (folderName + "parameters.txt");

    // Parameters for state variables

    out << "useLogW: "<<useLogW<<"\n";
    out << "nDims: "<<nDims<<"\n";
    out << "nWealth: "<<nOmega<<"\n";
    out << "nZ: "<<nZ<<"\n";
    out << "nV: "<<nV<<"\n";
    out << "nVtilde: "<<nH<<"\n";
    out << "numSds: "<<numSds<<"\n";
    out << "nShocks: "<<nShocks<<"\n";

    // Iteration parameters

    out << "verbatim: "<<verbatim<<"\n";
    out << "run: "<<run<<"\n";
    out << "folderName: "<<folderName<<"\n";
    out << "preLoad: "<<preLoad<<"\n";
    out << "method: "<<method<<"\n";
    out << "dt: "<<dt<<"\n";
    out << "dtInner: "<<dtInner<<"\n";
    out << "maxIters: "<<maxIters<<"\n";
    out << "maxItersInner: "<<maxItersInner<<"\n";
    out << "tol: "<<tol<<"\n";
    out << "innerTol: "<<innerTol<<"\n";
    out << "equityIss: "<<equityIss<<"\n";
    out << "hhCap: "<<hhCap<<"\n";
    out << "exportFreq: "<<exportFreq<<"\n";
    out << "CGscale: "<<CGscale<<"\n";
    out << "precondFreq: "<<precondFreq<<"\n";
    
    // OLG parameters
    out << "nu_newborn: "<<nu_newborn<<"\n";
    out << "lambda_d: "<<lambda_d<<"\n";


    // Persistence parameters

    out << "lambda_Z: "<<lambda_Z<<"\n";
    out << "lambda_V: "<<lambda_V<<"\n";
    out << "lambda_Vtilde: "<<lambda_H<<"\n";

    // Means
    out << "Z_bar: "<<Z_bar<<"\n";
    out << "V_bar: "<<V_bar<<"\n";
    out << "Vtilde_bar: "<<H_bar<<"\n";

    // Rates of time preferences
    out << "delta_e: "<<delta_e<<"\n";
    out << "delta_h: "<<delta_h<<"\n";

    // Productivity parameters
    out << "a_e: "<<a_e<<"\n";
    out << "a_h: "<<a_h<<"\n";

    // Inverses of EIS
    out << "rho_e: "<<rho_e<<"\n";
    out << "rho_h: "<<rho_h<<"\n";


    // Adjustment cost and depreciation
    out << "phi: "<<phi<<"\n";
    out << "alpha_K: "<<alpha_K<<"\n";


    // Risk aversion
    out << "gamma_e: "<<gamma_e<<"\n";
    out << "gamma_h: "<<gamma_h<<"\n";


    // Norm of vol
    out << "sigma_K_norm: "<<sigma_K_norm<<"\n";
    out << "sigma_Z_norm: "<<sigma_Z_norm<<"\n";
    out << "sigma_V_norm: "<<sigma_V_norm<<"\n";
    out << "sigma_Vtilde_norm: "<<sigma_H_norm<<"\n";

    // Equity issuance constraint
    out << "chiUnderline: "<<chiUnderline<<"\n";

    // Upper and lower boundaries of the state variables
    out << "omegaMin: "<<omegaMin<<"\n";
    out << "omegaMax: "<<omegaMax<<"\n";

    out << "zMin "<<zMin<<"\n";
    out << "zMax: "<<zMax<<"\n";

    out << "vMin: "<<vMin<<"\n";
    out << "vMax: "<<vMax<<"\n";

    out << "VtildeMin: "<<hMin<<"\n";
    out << "VtildeMax: "<<hMax<<"\n";

    // Correlation matrix
    Eigen::MatrixXd covMat;
    covMat.resize(4, 4);
    covMat << cov11, cov12, cov13, cov14, cov21, cov22, cov23, cov24,
    cov31, cov32, cov33, cov34, cov41, cov42, cov43, cov44;

    for (int i = 0; i < nShocks; i++) {
        for (int j = 0; j < nShocks; j++) {
            out << "cov"<<i + 1<<j + 1<<": "<<covMat(i,j)<<"\n";
        }
    }


    out.close();

}
