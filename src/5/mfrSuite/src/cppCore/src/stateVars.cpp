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


#include "stateVars.h"
stateVars::stateVars () {

}
stateVars::stateVars (Eigen::ArrayXd upper, Eigen::ArrayXd lower, Eigen::ArrayXd gridSizes, Parameters & parameters) {

    // Reading in parameters to create state variables
    upperLims = upper;
    lowerLims = lower;
    N = upper.size();
    S = gridSizes.prod();

    stateMat.resize(S,N);
    dVec.resize(N);
    increVec.resize(N);
    increVec(0) = 1;

    omega.resize(S); logW.resize(S);
    Z.resize(S);
    V.resize(S);
    H.resize(S);

    sqrtV.resize(S);
    sqrtH.resize(S);

    // fill in the state object; similar to the ndgrid function in MATLAB

    for (int n = 0; n < N; ++n) {

        if (n != 0) {
            increVec(n) = gridSizes(n - 1) * increVec(n - 1);
        }
        dVec(n) = (upper(n) - lower(n)) / (gridSizes(n) - 1);

        for (int i = 0; i < S; ++i) {
            stateMat(i,n) = lower(n) + dVec(n) * ( int(i /  increVec(n) ) % int( gridSizes(n) ) );
        }

    }

    num2State[0] = "w"; // set up map that maps numbers to state variables

    if (N == 1) {

        Z      = Eigen::MatrixXd::Constant(S, 1, parameters.Z_bar);
        V      = Eigen::MatrixXd::Constant(S, 1, parameters.V_bar);
        H      = Eigen::MatrixXd::Constant(S, 1, parameters.H_bar);

    } else if (N == 2) {

        if (parameters.sigma_Z_norm > 0.00000001) {

            Z      = stateMat.col(1);
            V      = Eigen::MatrixXd::Constant(S, 1, parameters.V_bar);
            H      = Eigen::MatrixXd::Constant(S, 1, parameters.H_bar);

            num2State[1] = "Z";

        } else if ( parameters.sigma_V_norm > 0.00000001 ) {

            Z      = Eigen::MatrixXd::Constant(S, 1, parameters.Z_bar);
            V      = stateMat.col(1);
            H      = Eigen::MatrixXd::Constant(S, 1, parameters.H_bar);

            num2State[1] = "V";


        } else if ( parameters.sigma_H_norm > 0.00000001 ) {

            Z      = Eigen::MatrixXd::Constant(S, 1, parameters.Z_bar);
            V      = Eigen::MatrixXd::Constant(S, 1, parameters.V_bar);
            H      = stateMat.col(1);

            num2State[1] = "H";

        }


    } else if (N == 3) {
        if (parameters.sigma_H_norm < 0.0000001) {
            Z      = stateMat.col(1);
            V      = stateMat.col(2);
            H      = Eigen::MatrixXd::Constant(S, 1, parameters.H_bar);

            num2State[1] = "Z";
            num2State[2] = "V";


        } else if (parameters.sigma_Z_norm < 0.0000001) {
            Z      = Eigen::MatrixXd::Constant(S, 1, parameters.Z_bar);
            V      = stateMat.col(1);
            H      = stateMat.col(2);

            num2State[1] = "V";
            num2State[2] = "H";

        } else if (parameters.sigma_V_norm < 0.0000001) {
            Z      = stateMat.col(1);
            V      = Eigen::MatrixXd::Constant(S, 1, parameters.V_bar);
            H      = stateMat.col(2);

            num2State[1] = "Z";
            num2State[2] = "H";

        }



    } else if (N == 4) {
        Z      = stateMat.col(1);
        V      = stateMat.col(2);
        H      = stateMat.col(3);

        num2State[1] = "Z";
        num2State[2] = "V";
        num2State[3] = "H";

    }

    if (parameters.useLogW) {
        logW = stateMat.col(0);
        omega = logW.exp();
    } else {
        omega = stateMat.col(0);
        logW  = omega.log();

    }
    // Create vol vectors

    //// Initialize sigma_K, sigma_Z, sigma_V, and sigma_H to have size of 4.
    //// They will only be used to create the vols of the state variables and
    //// sigmaK, and .leftCols(parameters.nShocks) will be applied on them.

    sigma_K.resize(4); sigma_Z.resize(4); sigma_V.resize(4); sigma_H.resize(4);

    sigma_K           << parameters.cov11 * parameters.sigma_K_norm, parameters.cov12 * parameters.sigma_K_norm, parameters.cov13 * parameters.sigma_K_norm, parameters.cov14 * parameters.sigma_K_norm;

    if (N == 1) {

        sigma_Z        << parameters.cov22 * parameters.sigma_Z_norm, parameters.cov22 * parameters.sigma_Z_norm, parameters.cov23 * parameters.sigma_Z_norm, parameters.cov24 * parameters.sigma_Z_norm;
        sigma_V        << parameters.cov33 * parameters.sigma_V_norm, parameters.cov32 * parameters.sigma_V_norm, parameters.cov33 * parameters.sigma_V_norm, parameters.cov34 * parameters.sigma_V_norm;
        sigma_H        << parameters.cov44 * parameters.sigma_H_norm, parameters.cov42 * parameters.sigma_H_norm, parameters.cov43 * parameters.sigma_H_norm, parameters.cov44 * parameters.sigma_H_norm;

    } else if (N == 2) {

        sigma_Z        << parameters.cov21 * parameters.sigma_Z_norm, parameters.cov22 * parameters.sigma_Z_norm, parameters.cov23 * parameters.sigma_Z_norm, parameters.cov24 * parameters.sigma_Z_norm;
        sigma_V        << parameters.cov21 * parameters.sigma_V_norm, parameters.cov22 * parameters.sigma_V_norm, parameters.cov23 * parameters.sigma_V_norm, parameters.cov24 * parameters.sigma_V_norm;
        sigma_H        << parameters.cov21 * parameters.sigma_H_norm, parameters.cov22 * parameters.sigma_H_norm, parameters.cov23 * parameters.sigma_H_norm, parameters.cov24 * parameters.sigma_H_norm;


    } else if (N == 3) {
        if (parameters.sigma_Z_norm < 0.0000001) {
            sigma_V        << parameters.cov21 * parameters.sigma_V_norm, parameters.cov22 * parameters.sigma_V_norm, parameters.cov23 * parameters.sigma_V_norm, parameters.cov24 * parameters.sigma_V_norm;
            sigma_H        << parameters.cov31 * parameters.sigma_H_norm, parameters.cov32 * parameters.sigma_H_norm, parameters.cov33 * parameters.sigma_H_norm, parameters.cov34 * parameters.sigma_H_norm;
            sigma_Z        << parameters.cov21 * parameters.sigma_Z_norm, parameters.cov22 * parameters.sigma_Z_norm, parameters.cov23 * parameters.sigma_Z_norm, parameters.cov24 * parameters.sigma_Z_norm;
        } else {
            sigma_Z        << parameters.cov21 * parameters.sigma_Z_norm, parameters.cov22 * parameters.sigma_Z_norm, parameters.cov23 * parameters.sigma_Z_norm, parameters.cov24 * parameters.sigma_Z_norm;
            sigma_V        << parameters.cov31 * parameters.sigma_V_norm, parameters.cov32 * parameters.sigma_V_norm, parameters.cov33 * parameters.sigma_V_norm, parameters.cov34 * parameters.sigma_V_norm;
            sigma_H        << parameters.cov31 * parameters.sigma_H_norm, parameters.cov32 * parameters.sigma_H_norm, parameters.cov33 * parameters.sigma_H_norm, parameters.cov34 * parameters.sigma_H_norm;
        }

    } else if (N == 4) {
        sigma_K        << parameters.cov11 * parameters.sigma_K_norm, parameters.cov12 * parameters.sigma_K_norm, parameters.cov13 * parameters.sigma_K_norm, parameters.cov14 * parameters.sigma_K_norm;
        sigma_Z        << parameters.cov21 * parameters.sigma_Z_norm, parameters.cov22 * parameters.sigma_Z_norm, parameters.cov23 * parameters.sigma_Z_norm, parameters.cov24 * parameters.sigma_Z_norm;
        sigma_V        << parameters.cov31 * parameters.sigma_V_norm, parameters.cov32 * parameters.sigma_V_norm, parameters.cov33 * parameters.sigma_V_norm, parameters.cov34 * parameters.sigma_V_norm;
        sigma_H        << parameters.cov41 * parameters.sigma_H_norm, parameters.cov42 * parameters.sigma_H_norm, parameters.cov43 * parameters.sigma_H_norm, parameters.cov44 * parameters.sigma_H_norm;
    }

    // Compute sqrt of V and H
    sqrtV = V.sqrt().matrix().array();
    sqrtH = H.sqrt().matrix().array();
    // Find boundary points
    for (int n = (N - 1); n >= 0; --n) {

        for (int i = 0; i < S; i++) {
            if ( abs(stateMat(i, n) - upperLims(n)) < dVec(n)/2 ) {
                // point at the upper boundary
                upperBdryPts[n].push_back(i);

            } else if ( abs(stateMat(i, n) - lowerLims(n)) < dVec(n)/2 ) {
                // point at the lower boundary
                lowerBdryPts[n].push_back(i);

            } else if ( abs(stateMat(i,n) - upperLims(n)) < dVec(n) * 1.5) {
                // point next to the upper boundary
                adjUpperBdryPts[n].push_back(i);
                nonBdryPts[n].push_back(i);

            } else if ( abs(stateMat(i, n) - lowerLims(n)) < dVec(n) * 1.5 ) {
                // point next to the upper boundary
                adjLowerBdryPts[n].push_back(i);
                nonBdryPts[n].push_back(i);

            } else {
                // point not at the boundary
                centralPts[n].push_back(i);
                nonBdryPts[n].push_back(i);

            }
        }
    }


    for (int n = (N - 1); n >= 0; --n) {
        upperBdryCt[n]        = upperBdryPts[n].size();
        lowerBdryCt[n]        = lowerBdryPts[n].size();
        adjUpperBdryCt[n]     = adjUpperBdryPts[n].size();
        adjLowerBdryCt[n]     = adjLowerBdryPts[n].size();
        nonBdryCt[n]          = nonBdryPts[n].size();
        centralCt[n]          = centralPts[n].size();

    }
}
