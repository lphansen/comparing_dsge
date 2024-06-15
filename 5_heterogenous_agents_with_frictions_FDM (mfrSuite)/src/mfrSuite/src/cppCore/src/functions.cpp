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


#include "functions.h"
#include "Parameters.h"
#include <unsupported/Eigen/Polynomials>
#include <fstream>
#include <iostream>
#include <iterator>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <signal.h>
#endif


static int s_interrupted = 0;

#ifdef _WIN32
  BOOL WINAPI consoleHandler(DWORD fdwCtrlType) {
    if(fdwCtrlType == CTRL_C_EVENT) {
      s_interrupted = 1;
      return TRUE;
    } else {
      return FALSE;
    }
  }
#else
  static void s_signal_handler (int signal_value)
  {
      s_interrupted = 1;
  }

  static void s_catch_signals (void)
  {
      struct sigaction action;
      action.sa_handler = s_signal_handler;
      action.sa_flags = 0;
      sigemptyset (&action.sa_mask);
      sigaction (SIGINT, &action, NULL);
      sigaction (SIGTERM, &action, NULL);
  }
#endif

/******************************************************/
/* Timer functions                                    */
/******************************************************/

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
    << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
    << std::endl;
    tictoc_stack.pop();
}

/******************************************************/
/* Derivative functions                               */
/******************************************************/

void computeFirstDerUpwind(stateVars & state_vars, valueVars & value_vars, matrixVars & matrix_vars, derivs & derivsXiE, derivs & derivsXiH) {
    /* This function computes first partials of value functions via upwinding. */
    /* Only do this when computing PDE error (last step of the program).       */
    /* Assumes thtat the derivs objects already contain derivatives computed   */
    /* using the conventional approach.                                        */

    /* This function is deprecated. It's saved here for testing purposes. */

    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int i = 0; i < state_vars.S; i++) {

            // Only computes derivatives when not at the boundary.
            if ( (abs(state_vars.stateMat(i, n) - state_vars.upperLims(n)) > state_vars.dVec(n)/2)
                && ( abs(state_vars.stateMat(i, n) - state_vars.lowerLims(n)) > state_vars.dVec(n)/2 ) ) {

                // Experts

                if (matrix_vars.firstCoefsE(i,n) <= 0 ) {
                    // backward diff
                    derivsXiE.firstPartials(i,n) = (value_vars.xi_e(i) - value_vars.xi_e(i - int(state_vars.increVec(n)))) / state_vars.dVec(n);
                } else {
                    // forward diff
                    derivsXiE.firstPartials(i,n) = (value_vars.xi_e(i + int(state_vars.increVec(n))) - value_vars.xi_e(i)) / state_vars.dVec(n);
                }

                // Households

                if (matrix_vars.firstCoefsH(i,n) <= 0 ) {
                    // backward diff
                    derivsXiH.firstPartials(i,n) = (value_vars.xi_h(i) - value_vars.xi_h(i - int(state_vars.increVec(n)))) / state_vars.dVec(n);
                } else {
                    // forward diff
                    derivsXiH.firstPartials(i,n) = (value_vars.xi_h(i + int(state_vars.increVec(n))) - value_vars.xi_h(i)) / state_vars.dVec(n);
                }


            }


        }


    }


}
/********************************************************/
/* Export functions                                     */
/********************************************************/


void exportInformation(std::vector<double> & timeItersVec, std::vector<double> & timeItersLinSysVec,
                       std::vector<double> & eErrorsVec, std::vector<double> & hErrorsVec,
                       std::vector<int> & cgEIters, std::vector<int> & cgHIters, Parameters & parameters) {

    /********************************************************/
    /* Export infomration on time, error, CG iterations     */
    /********************************************************/

    std::cout<<"Exporting solution information: time used, convergence error, etc."<<std::endl;

    std::ofstream timeItersFile( parameters.folderName +    "timePerIters" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> timeItersFile_iterator(timeItersFile, "\n");

    std::ofstream timeItersLinSysFile( parameters.folderName +    "timePerLinSysIters" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> timeItersLinSysFile_iterator(timeItersLinSysFile, "\n");

    std::ofstream eErrorsFile( parameters.folderName +    "eErrors" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> eErrorsFile_iterator(eErrorsFile, "\n");

    std::ofstream hErrorsFile( parameters.folderName +    "hErrors" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> hErrorsFile_iterator(hErrorsFile, "\n");

    std::ofstream cgEItersFile( parameters.folderName +    "cgEIters" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> cgEItersFile_iterator(cgEItersFile, "\n");

    std::ofstream cgHItersFile( parameters.folderName +    "cgHIters" + ".dat" , std::ios::out | std::ofstream::binary );
    std::ostream_iterator<double> cgHItersFile_iterator(cgHItersFile, "\n");


    std::copy(timeItersVec.begin(), timeItersVec.end(), timeItersFile_iterator);
    std::copy(timeItersLinSysVec.begin(), timeItersLinSysVec.end(), timeItersLinSysFile_iterator);
    std::copy(eErrorsVec.begin() + 1, eErrorsVec.end(), eErrorsFile_iterator);
    std::copy(hErrorsVec.begin() + 1, hErrorsVec.end(), hErrorsFile_iterator);

    if ( parameters.method.compare("2") == 0 ) {
        std::copy(cgEIters.begin(), cgEIters.end(), cgEItersFile_iterator);
        std::copy(cgHIters.begin(), cgHIters.end(), cgHItersFile_iterator);
    }


}

void exportPDE(matrixVars & matrix_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, Parameters & parameters, stateVars & state_vars, string suffix) {
    /********************************************************/
    /* This function is called at the end of the iterations */
    /* if there's no error in the value functions. It will  */
    /* export the coefficients of the matrices and compute  */
    /* the PDE errors for both hhs and experts              */
    /********************************************************/
    Eigen::ArrayXd pdeErrorE; pdeErrorE.resize(state_vars.S);
    Eigen::ArrayXd pdeErrorH; pdeErrorH.resize(state_vars.S);


    /* Export matrix coefficients */

    std::cout<<"Exporting matrices' information"<<std::endl;
    std::cout<<"=========================================="<<std::endl;

    std::cout<<"(1) Exporting matrix coefficients"<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    // Constant terms
    saveMarketVector(matrix_vars.Fe.array(), parameters.folderName + "Fe_" + suffix + ".dat");
    saveMarketVector(matrix_vars.Fh.array(), parameters.folderName + "Fh_" + suffix + ".dat");

    // First Partials' Coefficients
    saveMarketVector(matrix_vars.firstCoefsE.leftCols(parameters.nDims), parameters.folderName + "firstCoefsE_" + suffix + ".dat");
    saveMarketVector(matrix_vars.firstCoefsH.leftCols(parameters.nDims), parameters.folderName + "firstCoefsH_" + suffix + ".dat");

    // Second Partials' Coefficients
    saveMarketVector(matrix_vars.secondCoefsE.leftCols(parameters.nDims), parameters.folderName + "secondCoefsE_" + suffix + ".dat");
    saveMarketVector(matrix_vars.secondCoefsH.leftCols(parameters.nDims), parameters.folderName + "secondCoefsH_" + suffix + ".dat");

    std::cout<<"(2) Exporting matrices and RHSs."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    // Export matrices
    saveMarket(matrix_vars.Le,parameters.folderName + "Le_" + suffix + ".dat");
    saveMarket(matrix_vars.Lh,parameters.folderName + "Lh_" + suffix + ".dat");

    // Export RHSs
    saveMarketVector(matrix_vars.Ue.array(), parameters.folderName + "Ue_" + suffix + ".dat");
    saveMarketVector(matrix_vars.Uh.array(), parameters.folderName + "Uh_" + suffix + ".dat");


}
void exportData(valueVars & value_vars, Vars & vars,  derivs & derivsXiE, derivs & derivsXiH, derivs & derivsQ, derivs & derivsLogQ, string suffix, stateVars & state_vars, Parameters & parameters) {

    /********************************************************/
    /* Export model solution                                */
    /********************************************************/

    std::cout<<"Exporting state variables."<<std::endl;
    std::cout<<"=========================================="<<std::endl;

    /*  Export state variables */
    saveMarketVector(state_vars.omega,parameters.folderName +   "W" + ".dat");
    saveMarketVector(state_vars.logW,parameters.folderName +   "logW" + ".dat");
    saveMarketVector(state_vars.Z,parameters.folderName +   "Z" + ".dat");
    saveMarketVector(state_vars.V, parameters.folderName +   "V" + ".dat");
    saveMarketVector(state_vars.H, parameters.folderName +   "Vtilde" + ".dat"); // In code, Vtilde is the same as H

    /* Exporting data */
    std::cout<<"Exporting numerical results."<<std::endl;
    std::cout<<"=========================================="<<std::endl;

    //Risk prices and interest rate
    std::cout<<"(1): Exporting risk prices and interest rate: q, piH, piE, deltaE, deltaH, r, I."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    saveMarketVector(vars.q, parameters.folderName + "q_" + suffix + ".dat");
    saveMarketVector(vars.Pi.leftCols(parameters.nShocks),parameters.folderName +   "PiH_" + suffix  + ".dat");
    saveMarketVector(vars.PiE.leftCols(parameters.nShocks),parameters.folderName +   "PiE_" + suffix + ".dat");
    saveMarketVector(vars.deltaE,parameters.folderName +   "deltaE_" + suffix +  ".dat");
    saveMarketVector(vars.deltaH,parameters.folderName +   "deltaH_" + suffix + ".dat");
    saveMarketVector(vars.r,parameters.folderName +  "r_" + suffix + ".dat");
    saveMarketVector(vars.I,parameters.folderName +  "I_" + suffix + ".dat");

    //Drifts
    std::cout<<"(2) Exporting drifts: muQ, muX, muK, muRe, muRh."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    saveMarketVector(vars.muQ,parameters.folderName +   "muQ_" + suffix + ".dat");

    for (int n = 0; n < state_vars.N; n++) {
        saveMarketVector(vars.muXMap[state_vars.num2State[n]], parameters.folderName +   "mu" + state_vars.num2State[n] +  "_" + suffix + ".dat");
    }

    saveMarketVector(vars.muK,parameters.folderName +   "muK_" + suffix + ".dat");
    saveMarketVector( vars.muRe,parameters.folderName +   "muRe_" + suffix + ".dat");
    saveMarketVector( vars.muRh,parameters.folderName +   "muRh_" + suffix + ".dat");

    //Volatilities
    std::cout<<"(3) Exporting vols: sigmaQ, sigmaR, sigmaK."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    saveMarketVector( vars.sigmaQ.leftCols(parameters.nShocks),parameters.folderName +   "sigmaQ_" + suffix + ".dat");
    saveMarketVector( vars.sigmaR.leftCols(parameters.nShocks),parameters.folderName +   "sigmaR_" + suffix + ".dat");
    saveMarketVector( vars.sigmaK.leftCols(parameters.nShocks),parameters.folderName +   "sigmaK_" + suffix + ".dat");

    for (int n = 0; n < state_vars.N; n++) {
        saveMarketVector(vars.sigmaXMap[state_vars.num2State[n]].leftCols(parameters.nShocks), parameters.folderName +   "sigma" + state_vars.num2State[n] + "_" + suffix + ".dat");
    }

    //Value functions
    std::cout<<"(4) Exporting value and policy functions: XiE, XiH, cHatE, cHatH, kappa, chi, betaE, betaH."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    saveMarketVector(value_vars.xi_e, parameters.folderName + "xi_e_" + suffix + ".dat");
    saveMarketVector(value_vars.xi_h, parameters.folderName + "xi_h_" + suffix + ".dat");
    saveMarketVector(vars.cHat_e, parameters.folderName + "cHatE_" + suffix + ".dat");
    saveMarketVector(vars.cHat_h, parameters.folderName + "cHatH_" + suffix + ".dat");

    //Constraints
    saveMarketVector( value_vars.kappa,parameters.folderName +   "kappa_" + suffix + ".dat");
    saveMarketVector( value_vars.chi,parameters.folderName +   "chi_" + suffix + ".dat");
    saveMarketVector( vars.beta_e,parameters.folderName +   "betaE_" + suffix + ".dat");
    saveMarketVector( vars.beta_h,parameters.folderName +   "betaH_" + suffix + ".dat");

    //Derivatives
    std::cout<<"(5) Exporting derivs: first, second, and cross derivs of XiE and XiH."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    for (int n = 0; n < state_vars.N; n++) {

        // value functions
        saveMarketVector( derivsXiE.firstPartialsMap[state_vars.num2State[n]],parameters.folderName +   "dxi_e_d" + state_vars.num2State[n] + "_" + suffix + ".dat");
        saveMarketVector( derivsXiH.firstPartialsMap[state_vars.num2State[n]],parameters.folderName +   "dxi_h_d" + state_vars.num2State[n] + "_" + suffix + ".dat");
        saveMarketVector( derivsXiE.secondPartialsMap[state_vars.num2State[n]],parameters.folderName +   "d2xi_e_d" + state_vars.num2State[n] + "2_" + suffix + ".dat");
        saveMarketVector( derivsXiH.secondPartialsMap[state_vars.num2State[n]],parameters.folderName +   "d2xi_h_d" + state_vars.num2State[n] + "2_" + suffix + ".dat");

        // q and log(q)
        saveMarketVector( derivsQ.firstPartialsMap[state_vars.num2State[n]],parameters.folderName + "dq_dx_" + state_vars.num2State[n] + "_" + suffix + ".dat");

        saveMarketVector( derivsQ.secondPartialsMap[state_vars.num2State[n]],parameters.folderName + "d2q_d" + state_vars.num2State[n] + "2_" + suffix + ".dat");

        saveMarketVector( derivsLogQ.firstPartialsMap[state_vars.num2State[n]],parameters.folderName + "dlogQ_d" + state_vars.num2State[n] + "_" + suffix + ".dat");

        saveMarketVector( derivsLogQ.secondPartialsMap[state_vars.num2State[n]],parameters.folderName + "d2logQ_d" + state_vars.num2State[n] + "2_" + suffix + ".dat");


    }
    int k = choose(state_vars.N, 2);

    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            saveMarketVector( derivsXiE.crossPartialsMap[k].array(),parameters.folderName +   "d2xi_e_d" + state_vars.num2State[n] + "d" + state_vars.num2State[n_sub] + "_" + suffix + ".dat");
            saveMarketVector( derivsXiH.crossPartialsMap[k].array(),parameters.folderName +   "d2xi_h_d" + state_vars.num2State[n] + "d" + state_vars.num2State[n_sub] + "_" + suffix + ".dat");
            saveMarketVector( derivsQ.crossPartialsMap[k].array(),parameters.folderName +   "d2zeta_q_d" + state_vars.num2State[n] + "d" + state_vars.num2State[n_sub] + "_" + suffix + ".dat");
            saveMarketVector( derivsLogQ.crossPartialsMap[k].array(),parameters.folderName +   "d2zeta_logQ_d" + state_vars.num2State[n] + "d" + state_vars.num2State[n_sub] + "_" + suffix + ".dat");


        }
    }



    //Export the rest
    std::cout<<"(6) Exporting the rest: experts' leverage, sigmaC, sigmaCe, sigmaCh, sigmaSe, sigmaSh, simgaLogY, muC, muCe, muCh, muY, muSe, muSh."<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;

    saveMarketVector( value_vars.leverageExperts, parameters.folderName +   "levExp_" + suffix + ".dat");
    saveMarketVector( vars.sigmaC.leftCols(parameters.nShocks),parameters.folderName +   "sigmaC_" + suffix + ".dat");
    saveMarketVector( vars.sigmaCe.leftCols(parameters.nShocks),parameters.folderName +   "sigmaCe_" + suffix + ".dat");
    saveMarketVector( vars.sigmaCh.leftCols(parameters.nShocks),parameters.folderName +   "sigmaCh_" + suffix + ".dat");

    saveMarketVector( vars.sigmaSe.leftCols(parameters.nShocks),parameters.folderName +   "sigmaSe_" + suffix + ".dat");
    saveMarketVector( vars.sigmaSh.leftCols(parameters.nShocks),parameters.folderName +   "sigmaSh_" + suffix + ".dat");

    saveMarketVector( vars.sigmaY.leftCols(parameters.nShocks),parameters.folderName +   "sigmaY_" + suffix + ".dat");

    saveMarketVector( vars.muC,parameters.folderName +   "muC_" + suffix + ".dat");
    saveMarketVector( vars.muCe,parameters.folderName +   "muCe_" + suffix + ".dat");
    saveMarketVector( vars.muCh,parameters.folderName +   "muCh_" + suffix + ".dat");

    saveMarketVector( vars.muY,parameters.folderName +   "muY_" + suffix + ".dat");

    saveMarketVector( vars.muSe,parameters.folderName +   "muSe_" + suffix + ".dat");
    saveMarketVector( vars.muSh,parameters.folderName +   "muSh_" + suffix + ".dat");

}


double f(double x, double coef1, double coef2, double coef3, double coef4)
{
    // we are taking equation as x^3+x-1
    double f = coef4 * pow(x, 3) + coef3 * pow(x,2) + coef2 * x + coef1;
    return f;
}

double rootFinder(double L, double U, double E, double coef1, double coef2, double coef3, double coef4, int maxIters)
{
    // Root-finder used to solve for chi when chi is nonlinear //
    // Not used at the moment because iterative method works //

    double x0, m;
    int i = 0;
    while (i < maxIters) {
        if (std::abs( f(x0, coef1, coef2, coef3, coef4) ) < E ) {
            // Tolerance met; break the loop
            return x0;
        } else if ( (f(L, coef1, coef2, coef3, coef4) < 0) && (f(U, coef1, coef2, coef3, coef4) > 0) ) {
            // Find midpoint

            x0 = 0.5 * L + 0.5 * U;
            m  = f(x0, coef1, coef2, coef3, coef4);
            if (m > 0) {
                U = x0;
            } else {
                L = x0;
            }
        } else if ( (f(L, coef1, coef2, coef3, coef4) > 0) && (f(U, coef1, coef2, coef3, coef4) > 0)  ) {
            // Returning L, because the min operator is solved.
            //std::cout<<"Returning L; min solved"<<std::endl;

            return L;
        } else if ( (f(L, coef1, coef2, coef3, coef4) > 0) && (f(U, coef1, coef2, coef3, coef4) < 0)  ) {
            // Returning L, but second root could exist.

            return L;
        } else if ( (f(L, coef1, coef2, coef3, coef4) < 0) && (f(U, coef1, coef2, coef3, coef4) < 0)  ) {
            // Degenerate case; no roots
            return -99.0;
        }
        i += 1;
    }

    return x0;
}



//iteration function

int iterFunc(stateVars & state_vars, valueVars & value_vars, Vars & vars, matrixVars & matrix_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, derivs & derivsKappa, derivs & derivsLogABar, Parameters & parameters, std::vector<double> & timeItersVec, std::vector<double> & timeItersLinSysVec,
             std::vector<double> & eErrorsVec, std::vector<double> & hErrorsVec,
             std::vector<int> & cgEIters, std::vector<int> & cgHIters) {
    s_interrupted = 0;
    int terminationCode = 0;

    #ifdef _WIN32
      if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
          std::cout<<"ERROR: Could not set control handler"<<std::endl;
      } else {
          std::cout<<"Control hanlder installed"<<std::endl;
      }
    #else
      s_catch_signals ();
    #endif

    /* Explanation on parameters.equityIss and parameters.hhCap */

    /* parameters.equityIss = 0 if equity issuance is not allowed; chi = 1 always */
    /*                      = 1 if equity issuance is allowed and skin-in-the-game constraint binds always; chi = parameters.chiUnderline < 1.0 */
    /*                      = 2 if equity issuance is allowed and skin-in-the-game constraint binds occasionally */

    /* parameters.hhCap     = 0 if households are not allowed to hold capital; kappa = 1 always */
    /*                      = 1 if households are allowed to hold capital; */

    double innerError      = 0;

    /******************************************************/
    /********* Initialize for the algorithm       *********/
    /******************************************************/


    /* Initialize arrays to store temporary data */
    Eigen::ArrayXd allErrorE; allErrorE.resize(state_vars.S);
    Eigen::ArrayXd allErrorH; allErrorH.resize(state_vars.S);
    Eigen::MatrixXf::Index index;
    Eigen::ArrayXd timeIters; timeIters.resize(parameters.maxIters);
    Eigen::ArrayXd H; H.resize(state_vars.S); Eigen::ArrayXd Hsaved; Hsaved.resize(state_vars.S);
    Eigen::ArrayXd sigmaRsigmaXDerivs; sigmaRsigmaXDerivs.resize(state_vars.S);
    sigmaRsigmaXDerivs = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0);

    Eigen::ArrayXd qTilde; qTilde.resize(state_vars.S); Eigen::ArrayXd logQTilde; logQTilde.resize(state_vars.S);
    derivs derivsQTilde (state_vars, parameters); derivs derivsLogQTilde (state_vars, parameters);

    Eigen::VectorXd XiEVector; Eigen::VectorXd XiHVector;

    /* Initialize clocks to record time inforamtion */

    high_resolution_clock::time_point t2; high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t4; high_resolution_clock::time_point t3;
    high_resolution_clock::time_point t6; high_resolution_clock::time_point t5;


    /* Initialize Eigen's cg solver */
    Eigen::LeastSquaresConjugateGradient<SpMat > cgE;
    Eigen::LeastSquaresConjugateGradient<SpMat > cgH;

    std::cout<<"Start to run program. Run name: "<<parameters.run<<std::endl;

    /***********************************************************************/
    /* Update equilibrium quantities invariant throughout iterations       */
    /***********************************************************************/

    // Drifts and vols of K and exogenous state variables

    //// This part computes the drifts and vols of the variables that do not need to be repeatedly updated.
    vars.sigmaK.resize(state_vars.S, parameters.nShocks);
    vars.sigmaK = ((state_vars.sigma_K.segment(0,parameters.nShocks).transpose().replicate(state_vars.S, 1)).array().colwise()
     * state_vars.sqrtV.array()).matrix();
    vars.sigmaXMap["Z"] = ((state_vars.sigma_Z.segment(0,parameters.nShocks).transpose().replicate(state_vars.S, 1)).array().colwise()
     * state_vars.sqrtV.array()).matrix();
    vars.sigmaXMap["V"] = ((state_vars.sigma_V.segment(0,parameters.nShocks).transpose().replicate(state_vars.S, 1)).array().colwise()
     * state_vars.sqrtV.array()).matrix();
    vars.sigmaXMap["H"] = (state_vars.sigma_H.segment(0,parameters.nShocks).transpose().replicate(state_vars.S, 1)).array().colwise() * state_vars.sqrtH.array();
    vars.muXMap["Z"]    = parameters.lambda_Z * (parameters.Z_bar - state_vars.Z);
    vars.muXMap["V"]    = parameters.lambda_V * (parameters.V_bar - state_vars.V);
    vars.muXMap["H"]    = parameters.lambda_H * (parameters.H_bar - state_vars.H);
    Eigen::ArrayXd zeroNonlinearCoefs; Eigen::ArrayXd firstNonlinearCoefs; Eigen::ArrayXd secondNonlinearCoefs; Eigen::ArrayXd thirdNonlinearCoefs;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> icholE;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> icholH;
    /* Start iterations */
    for(int i = 0; i < parameters.maxIters; i ++) {

        if (s_interrupted) {
            std::cout << "Terminating outer loop per user instruction..." << std::endl;
            terminationCode = -3;
            break;
        }
        t1 = high_resolution_clock::now();

        //******* Compute error and store data ************//
        std::cout<<"Computing for error"<<std::endl;
        allErrorE = (value_vars.xi_e - value_vars.xi_e_old).array() / parameters.dt;
        allErrorH = (value_vars.xi_h - value_vars.xi_h_old).array() / parameters.dt;

        allErrorE.abs().maxCoeff(&index); eErrorsVec.push_back( allErrorE(index) );
        allErrorH.abs().maxCoeff(&index); hErrorsVec.push_back( allErrorH(index) );

        std::cout<<"Updating value functions"<<std::endl;

        value_vars.xi_e_old = value_vars.xi_e; value_vars.xi_h_old = value_vars.xi_h;
        value_vars.kappa_old = value_vars.kappa; value_vars.chi_old = value_vars.chi;




        /*********************************************************/
        /* Execute numerical algorithm */
        /*********************************************************/

        if ( std::isnan(value_vars.xi_e.sum()) || std::isnan(value_vars.xi_h.sum())
            || std::isinf(value_vars.xi_e.sum()) || std::isinf(value_vars.xi_e.sum()) ) {
            std::cout<<std::endl;
            std::cout<<"========================================================="<<std::endl;
            std::cout<<"END OF ITERATIONS: VALUE FUNCTIONS CONTAIN NAN OR INF VALUES"<<std::endl;
            std::cout<<"========================================================="<<std::endl;

            // Remove first element of errors vectors
            if (!eErrorsVec.empty()) {
                eErrorsVec.erase (eErrorsVec.begin());
                hErrorsVec.erase (hErrorsVec.begin());
            }
            /****************************************************************/
            /* Value functions contain NaN or inf values. End iterations. ***/
            /****************************************************************/

            return -1;
        }

        if (i == 0  || ( ( abs(eErrorsVec[i]) > parameters.tol) || ( abs(hErrorsVec[i]) > parameters.tol) ) ) {

            /************************************************************/
            /* Tolerance not met. Continue program                    ***/
            /************************************************************/

            /*********************************************************/
            /* Step 0: Print out information */
            /*********************************************************/
            std::cout<<std::endl;
            std::cout<<"========================= Iteration: "<<i<<" ============================="<<std::endl;

            if (i == 0) {
                std::cout<<"First iteration: no error information."<<std::endl;;
            } else {
                std::cout<<"Error for xi_e: "<<eErrorsVec[i]<<std::endl;
                std::cout<<"Error for xi_h: "<<hErrorsVec[i]<<std::endl;
                std::cout<<"Tolerance not met; keep iterating. "<< std::endl;
            }

            t3 = high_resolution_clock::now();

            derivsXiE.computeDerivs(value_vars.xi_e, state_vars);
            derivsXiH.computeDerivs(value_vars.xi_h, state_vars);

            int j = 0;
            while (j < parameters.maxItersInner) {

                if (s_interrupted) {
                    std::cout << "Terminating inner loop per user instruction..." << std::endl;
                    break;
                }

                if (parameters.verbatim && j > 0) {
                    std::cout<<"-----------------------------"<<std::endl;
                    std::cout<<"Inner iteration "<<j<<std::endl;
                    std::cout<<"Error for chi: "<<(value_vars.chi - value_vars.chi_old).abs().maxCoeff() / parameters.dtInner <<std::endl;
                    std::cout<<"Error for kappa: "<<(value_vars.kappa - value_vars.kappa_old).abs().maxCoeff() / parameters.dtInner <<std::endl;

                }

                /*********************************************************/
                /* Going into inner loop                                 */
                /*********************************************************/

                if ( (innerError/parameters.dtInner) >= parameters.innerTol || j == 0) {
                    /*********************************************************/
                    /* Step a: Initialize                                    */
                    /*********************************************************/
                    if ( (i == 0) & (j == 0) ) {
                        if (parameters.hhCap == 0 || (parameters.a_h > 0 && parameters.chiUnderline < 0.000000001) ) {
                            std::cout<<"Households not allowed to hold capital or chiUnderline is zero, setting kappa to 1 everywhere."<<std::endl;
                            value_vars.kappa = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
                        } else {
                            std::cout<<"Households allowed to hold capital, use kappa = w or the preloaded solution as guess."<<std::endl;
                        }

                        // First iteration of the inner loop of the first iteration of the outer loop. Need to
                        // initialize. When i > 0, use the solution from previous outerloop as starting point.
                        value_vars.chi   = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
                        vars.q           = ( (1.0 - value_vars.kappa) * parameters.a_h + value_vars.kappa * parameters.a_e + 1/parameters.phi ) / ( (1.0 -state_vars.omega) * pow(parameters.delta_h, 1/parameters.rho_h) * (value_vars.xi_h.exp()).pow(1-1/parameters.rho_h) + state_vars.omega * pow(parameters.delta_e,1/parameters.rho_e) * value_vars.xi_e.exp().pow(1 - 1/parameters.rho_e) + 1/parameters.phi );
                        vars.logQ        = vars.q.log();
                        vars.I        = vars.logQ / parameters.phi;
                        vars.deltaH      = (1.0 - value_vars.kappa) / (1.0 - state_vars.omega) * parameters.gamma_h * state_vars.H;
                        vars.deltaE      = (vars.deltaH + (parameters.a_e - parameters.a_h) / vars.q) / parameters.chiUnderline;

                    }
                    value_vars.kappa_old = value_vars.kappa;
                    value_vars.chi_old   = value_vars.chi;

                    /*********************************************************/
                    /* Step b: Start iterating in inner loops                */
                    /*********************************************************/

                    // Step b(i)
                    vars.beta_h      = ( 1.0 - value_vars.kappa ) / (1.0 - state_vars.omega );
                    vars.beta_e      = ( value_vars.chi * value_vars.kappa ) / ( state_vars.omega );
                    derivsLogQ.computeDerivs(vars.logQ, state_vars);
                    derivsQ.computeDerivs(vars.q, state_vars);

                    vars.sigmaQ = ( vars.sigmaK.array().colwise() *  ( derivsLogQ.firstPartialsMap["w"] * ( value_vars.kappa * value_vars.chi - state_vars.omega ) )  + ( vars.sigmaXMap["Z"].array().colwise() *  derivsLogQ.firstPartialsMap["Z"]  ) +  ( vars.sigmaXMap["V"].array().colwise() *  derivsLogQ.firstPartialsMap["V"]  ) + ( vars.sigmaXMap["H"].array().colwise() *  derivsLogQ.firstPartialsMap["H"]  ) ).array().colwise() / ( 1.0 - (value_vars.kappa * value_vars.chi - state_vars.omega ) * derivsLogQ.firstPartialsMap["w"].array() );

                    vars.sigmaR = vars.sigmaQ + vars.sigmaK;

                    vars.normR2 = vars.sigmaR.rowwise().norm().array().square();

                    vars.sigmaXMap["w"] = (vars.sigmaR).array().colwise() * ( (value_vars.kappa * value_vars.chi - state_vars.omega) );

                    // Step b(ii)
                    if ((! (parameters.hhCap == 0)) && (parameters.a_h > 0 && parameters.chiUnderline > 0.000000001) ){

                        if (j == 0) {
                            std::cout<<"Households are allowed to hold capital - need to solve for kappa. Inner tolerance: "<<parameters.innerTol<<std::endl;
                        }

                        /* Households are allowed to hold capital; solve for it */
                        sigmaRsigmaXDerivs = sigmaRsigmaXDerivs * 0.0;

                        for (int s = 0; s < parameters.nShocks; s++) {
                            for (int n = 0; n < state_vars.N; n++) {
                                sigmaRsigmaXDerivs = sigmaRsigmaXDerivs + vars.sigmaR.col(s).array() * ( vars.sigmaXMap[state_vars.num2State[n]].col(s).array() * ( (parameters.gamma_h - 1.0) * derivsXiH.firstPartialsMap[state_vars.num2State[n]] - (parameters.gamma_e - 1.0) * derivsXiE.firstPartialsMap[state_vars.num2State[n]] ) );
                            }
                        }

                        H = state_vars.omega * parameters.gamma_h * (1.0 - parameters.chiUnderline * value_vars.kappa) * vars.normR2 + state_vars.omega * parameters.gamma_h * (1.0 - value_vars.kappa) / parameters.chiUnderline * state_vars.H - (1.0 - state_vars.omega) * parameters.gamma_e * parameters.chiUnderline * value_vars.kappa * (vars.normR2 + state_vars.H) + state_vars.omega * (1.0 - state_vars.omega) * (parameters.a_e - parameters.a_h) / (parameters.chiUnderline * vars.q) + state_vars.omega * (1.0 - state_vars.omega) * sigmaRsigmaXDerivs;

                        H = (H <= (1.0 - value_vars.kappa) ).cast<double>() * H + (H > (1.0 - value_vars.kappa) ).cast<double>() * (1.0 - value_vars.kappa);

                        value_vars.kappa = value_vars.kappa + H * parameters.dtInner;
                    }


                    // Step b(iii)

                    if ( (parameters.chiUnderline  >= 1) || (parameters.equityIss == 1) ) {

                        /* If experts are not allowed to issue equity or if the constraint binds always, set chi to be the constraint everywhere */
                        /* When the constraint is less than 1 and when it binds always, homogenenous risk aversion required */

                        if (j == 0 ) {
                            std::cout<<"Experts are either not allowed to isuse equity or the constraint binds always, setting chi to constraint everywhere"<<std::endl;
                        }
                        value_vars.chi = Eigen::MatrixXd::Constant(state_vars.S, 1, parameters.chiUnderline);

                    } else {

                        /* This part will only be executed if parameters.equityIss is 2 and parameters.chiUnderline < 1 */

                        vars.q           = ( (1.0 - value_vars.kappa) * parameters.a_h + value_vars.kappa * parameters.a_e + 1/parameters.phi ) / ( (1.0 -state_vars.omega) * pow(parameters.delta_h, 1/parameters.rho_h) * (value_vars.xi_h.exp()).pow(1-1/parameters.rho_h) + state_vars.omega * pow(parameters.delta_e,1/parameters.rho_e) * value_vars.xi_e.exp().pow(1 - 1/parameters.rho_e) + 1/parameters.phi );
                        vars.logQ        = vars.q.log();
                        vars.I           = vars.logQ / parameters.phi;


                        qTilde           = ( (1.0 - 1.0) * parameters.a_h + 1.0 * parameters.a_e + 1/parameters.phi ) / ( (1.0 -state_vars.omega) * pow(parameters.delta_h, 1/parameters.rho_h) * (value_vars.xi_h.exp()).pow(1-1/parameters.rho_h) + state_vars.omega * pow(parameters.delta_e,1/parameters.rho_e) * value_vars.xi_e.exp().pow(1 - 1/parameters.rho_e) + 1/parameters.phi );

                        logQTilde        = qTilde.log();

                        derivsQTilde.computeDerivs(qTilde, state_vars);
                        derivsLogQTilde.computeDerivs(logQTilde, state_vars);

                        vars.Dx.setZero(); // clear matrix Dx
                        //update Dx
                        for (int s = 0; s < parameters.nShocks; s++ ) {
                            vars.Dx.col(s) = vars.sigmaK.col(s);
                            for (int n = 1; n < state_vars.N; n++ ) {
                                vars.Dx.col(s) = vars.Dx.col(s).array() + vars.sigmaXMap[state_vars.num2State[n]].col(s).array() * derivsLogQTilde.firstPartialsMap[state_vars.num2State[n]].array();
                            }
                        }

                        //update D_zeta_omega
                        vars.DzetaOmega = vars.Dx.rowwise().norm().array().square() * ( (parameters.gamma_h - 1.0) * derivsXiH.firstPartialsMap["w"].array() - (parameters.gamma_e - 1.0) * derivsXiE.firstPartialsMap["w"].array() );
                        vars.DzetaOmega = vars.DzetaOmega * (state_vars.omega * (1.0 - state_vars.omega));
                        //update d_zeta_xtilde
                        vars.DzetaX = vars.DzetaX * 0.0; // clear vector DzetaX

                        for (int s = 0; s < parameters.nShocks; s++ ) {
                            for (int n = 1; n < state_vars.N; n++ ) {
                                vars.DzetaX = vars.DzetaX + vars.Dx.col(s).array() * ( vars.sigmaXMap[state_vars.num2State[n]].col(s).array() * ( (parameters.gamma_h - 1.0) * derivsXiH.firstPartialsMap[state_vars.num2State[n]].array() - (parameters.gamma_e - 1.0) * derivsXiE.firstPartialsMap[state_vars.num2State[n]].array() ) );
                            }

                        }
                        vars.DzetaX = vars.DzetaX.colwise() * (state_vars.omega * (1.0 - state_vars.omega));

                        value_vars.chi = - ( (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H * derivsLogQTilde.firstPartialsMap["w"].array().pow(2.0) * (value_vars.chi - state_vars.omega).pow(3.0) + (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H *  derivsLogQTilde.firstPartialsMap["w"].array() * (state_vars.omega * derivsLogQTilde.firstPartialsMap["w"].array() - 2.0) * (value_vars.chi - state_vars.omega).pow(2.0)  + state_vars.omega * (1 - state_vars.omega) * (parameters.gamma_e - parameters.gamma_h) * vars.Dx.rowwise().norm().array().square() + state_vars.omega * (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H - vars.DzetaX ) / (  ((1 - state_vars.omega) * parameters.gamma_e + state_vars.omega * parameters.gamma_h ) * vars.Dx.rowwise().norm().array().square() + (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H * (1.0 - 2.0 * state_vars.omega * derivsLogQTilde.firstPartialsMap["w"].array() ) + derivsLogQTilde.firstPartialsMap["w"].array() * vars.DzetaX - vars.DzetaOmega ) + state_vars.omega;

                        value_vars.chi = (value_vars.chi <= (parameters.chiUnderline) ).cast<double>() * parameters.chiUnderline + (value_vars.chi > (parameters.chiUnderline) ).cast<double>() * (value_vars.chi);

                        value_vars.chi = (value_vars.chi > (1.0) ).cast<double>() * 1.0 + (value_vars.chi < (1.0) ).cast<double>() * (value_vars.chi);

                        sigmaRsigmaXDerivs = sigmaRsigmaXDerivs * 0.0;

                        for (int s = 0; s < parameters.nShocks; s++) {
                            for (int n = 0; n < state_vars.N; n++) {
                                sigmaRsigmaXDerivs = sigmaRsigmaXDerivs + vars.sigmaR.col(s).array() * ( vars.sigmaXMap[state_vars.num2State[n]].col(s).array() * ( (parameters.gamma_h - 1.0) * derivsXiH.firstPartialsMap[state_vars.num2State[n]] - (parameters.gamma_e - 1.0) * derivsXiE.firstPartialsMap[state_vars.num2State[n]] ) );
                            }
                        }


                        if (  (((  ((1 - state_vars.omega) * parameters.gamma_e + state_vars.omega * parameters.gamma_h ) * vars.Dx.rowwise().norm().array().square() + (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H * (1.0 - 2.0 * state_vars.omega * derivsLogQTilde.firstPartialsMap["w"].array() ) + derivsLogQTilde.firstPartialsMap["w"].array() * vars.DzetaX - vars.DzetaOmega )).minCoeff() < 0) &&  (((  ((1 - state_vars.omega) * parameters.gamma_e + state_vars.omega * parameters.gamma_h ) * vars.Dx.rowwise().norm().array().square() + (1.0 - state_vars.omega) * parameters.gamma_e * state_vars.H * (1.0 - 2.0 * state_vars.omega * derivsLogQTilde.firstPartialsMap["w"].array() ) + derivsLogQTilde.firstPartialsMap["w"].array() * vars.DzetaX - vars.DzetaOmega )).maxCoeff() > 0) ) {

                            std::cout<<"***********************************************************************************************************************"<<std::endl;
                            std::cout<<"WARNING: ZERO IS IN THE DENOMINATOR FOR CHI (EQUITY ISSUANCE CONSTRAINT). THIS HAPPENS BECAUSE ONE OF THE FOLLOWING: \n (1) THE RISK AVERSIONS FOR EXPERTS AND HOUSEHOLDS MAY BE TOO FAR APART. \n (2) INVERSE OF EIS IS TOO FAR AWAY FROM ONE."<<std::endl;
                            std::cout<<"***********************************************************************************************************************"<<std::endl;


                        }
                        value_vars.chi = (value_vars.chi <= (parameters.chiUnderline) ).cast<double>() * parameters.chiUnderline + (value_vars.chi > (parameters.chiUnderline) ).cast<double>() * (value_vars.chi);

                        // value_vars.chi = (value_vars.chi >= (1.0) ).cast<double>() * 1.0 + (value_vars.chi < (1.0) ).cast<double>() * (value_vars.chi);

                    }

                    // Step b(iv)
                    sigmaRsigmaXDerivs = sigmaRsigmaXDerivs * 0.0;

                    for (int s = 0; s < parameters.nShocks; s++) {
                        for (int n = 0; n < state_vars.N; n++) {
                            sigmaRsigmaXDerivs = sigmaRsigmaXDerivs + vars.sigmaR.col(s).array() * ( vars.sigmaXMap[state_vars.num2State[n]].col(s).array() * ( (parameters.gamma_h - 1.0) * derivsXiH.firstPartialsMap[state_vars.num2State[n]] - (parameters.gamma_e - 1.0) * derivsXiE.firstPartialsMap[state_vars.num2State[n]] ) );
                        }
                    }

                    vars.deltaE = parameters.gamma_e * (value_vars.chi * value_vars.kappa) / state_vars.omega * (vars.normR2 + state_vars.H) - parameters.gamma_h * (1.0 - value_vars.chi * value_vars.kappa) / (1.0 - state_vars.omega) * vars.normR2 - sigmaRsigmaXDerivs;

                    vars.deltaH = parameters.chiUnderline * vars.deltaE - (parameters.a_e - parameters.a_h) / vars.q;

                    // Step b(v)
                    vars.q           = ( (1.0 - value_vars.kappa) * parameters.a_h + value_vars.kappa * parameters.a_e + 1/parameters.phi ) / ( (1.0 -state_vars.omega) * pow(parameters.delta_h, 1/parameters.rho_h) * (value_vars.xi_h.exp()).pow(1-1/parameters.rho_h) + state_vars.omega * pow(parameters.delta_e,1/parameters.rho_e) * value_vars.xi_e.exp().pow(1 - 1/parameters.rho_e) + 1/parameters.phi );
                    vars.logQ        = vars.q.log();
                    vars.I        = vars.logQ / parameters.phi;

                    /********************************************/
                    /******* Step c: Compute error         ******/
                    /********************************************/

                    innerError = (value_vars.kappa - value_vars.kappa_old).abs().maxCoeff() + (value_vars.chi - value_vars.chi_old).abs().maxCoeff();
                } else {

                    /* Either tolerance met or the max number of iterations emt; break out of while loop */

                    break;
                }

                /* Tolerance not met; continue while loop */

                j = j + 1;
            }

            t4 = high_resolution_clock::now();
            auto durationInner = duration_cast<microseconds>( t4 - t3 ).count();

            /*********************************************************/
            /* End of inner loop                                     */
            /*********************************************************/

            std::cout << "Inner loop is completed; time elapsed: "<<durationInner / 1000000.0 <<"; took "<<j<<" iterations."<<std::endl;

            /*************************************************************/
            /* Updating outdated equilibrium quantities after inner loop */
            /*************************************************************/

            vars.I        = vars.logQ / parameters.phi;
            vars.muK = 0.04*state_vars.Z + vars.I - parameters.alpha_K - 0.5 * vars.sigmaK.rowwise().sum().array().square();
            derivsLogQ.computeDerivs(vars.logQ, state_vars);
            derivsQ.computeDerivs(vars.q, state_vars); std::cout<<"Finished computing derivs"<<std::endl;
            vars.updateSigmaPi(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, parameters);
            vars.updateMuAndR(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, parameters);
            vars.aBar = value_vars.kappa * parameters.a_e + (1.0 - value_vars.kappa) * parameters.a_h;
            vars.logABar = vars.aBar.array().log().matrix().array();
            derivsLogABar.computeDerivs(vars.logABar, state_vars);

            std::cout<<"Finished updating equilibrium quantities. Next step: update matrices."<<std::endl;

            /*********************************************************/
            /* Step 5: Solving PDEs for households and experts */
            /*********************************************************/

            matrix_vars.updateMatrixVars(state_vars, value_vars, vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, parameters);
            matrix_vars.updateMatrix(state_vars, parameters);
            std::cout<<"Finished updating matrix."<<std::endl;
            matrix_vars.updateKnowns(value_vars, state_vars, parameters);
            std::cout<<"Finished updating the known vector."<<std::endl;



            t5 = high_resolution_clock::now();
            if ( parameters.method.compare("1") == 0 ) {
                /* Implement explicit scheme */

                std::cout<<"Solving systems using explicit scheme."<<std::endl;
                value_vars.xi_e =  (matrix_vars.Le * matrix_vars.Ue).array();
                value_vars.xi_e = value_vars.xi_e + (matrix_vars.Fe * parameters.dt).array();
                value_vars.xi_h =  (matrix_vars.Lh * matrix_vars.Uh).array();
                value_vars.xi_h = value_vars.xi_h + (matrix_vars.Fh * parameters.dt).array();

            } else if ( parameters.method.compare("2") == 0 ) {

                /* Implement CG parameters.method */
                std::cout<<"Solving systems using the implicit scheme via CG..."<<std::endl;
                std::cout << "Running conjugate gradient solver using " << nbThreads() << " threads." << std::endl;
                XiEVector = value_vars.xi_e.matrix(); XiHVector = value_vars.xi_h.matrix();

                cgE.setTolerance( parameters.tol / 10.0 * parameters.dt * parameters.CGscale);
                cgH.setTolerance( parameters.tol / 10.0 * parameters.dt * parameters.CGscale);
                std::cout<<"CG error tolerance: "<<parameters.tol / 10.0 * parameters.dt * parameters.CGscale<<std::endl;

                if (parameters.precondFreq < 0) {

                    // Not using preconditioners; use the CG function provided
                    // in Eigen
                    cgE.compute(matrix_vars.Le);
                    XiEVector = cgE.solveWithGuess(matrix_vars.Ue, XiEVector);
                    value_vars.xi_e = XiEVector;

                    cgH.compute(matrix_vars.Lh);
                    XiHVector = cgH.solveWithGuess(matrix_vars.Uh, XiHVector);
                    value_vars.xi_h = XiHVector;

                    std::cout<<"Conjugate gradient (not preconditioned) took "<< cgE.iterations() << " (experts) and "<< cgH.iterations() << " (households) iterations."<<std::endl;
                    cgEIters.push_back(cgE.iterations()); cgHIters.push_back(cgH.iterations());

                } else {

                    // Using incomplete cholesky to precondition;
                    // Using a modified function from Eigen

                    if (i % parameters.precondFreq == 0) {
                        std::cout<<"Factorizing preconditioners. Note that preconditioenrs are factorized every "<<parameters.precondFreq<<" iteration(s)."<<std::endl;
                        matrix_vars.LeNormed = matrix_vars.Le.transpose() * matrix_vars.Le;
                        matrix_vars.LhNormed = matrix_vars.Lh.transpose() * matrix_vars.Lh;
                        icholE.compute(matrix_vars.LeNormed);
                        icholH.compute(matrix_vars.LhNormed);
                    }

                    matrix_vars.solveWithCGICholE(value_vars, state_vars, icholE,
                        parameters.tol / 10.0 * parameters.dt * parameters.CGscale, 6000);


                    matrix_vars.solveWithCGICholH(value_vars, state_vars, icholH,
                        parameters.tol / 10.0 * parameters.dt * parameters.CGscale, 6000);

                    std::cout<<"Conjugate gradient (preconditioned) took "<< matrix_vars.cgEIters << " (experts) and "<< matrix_vars.cgHIters << " (households) iterations."<<std::endl;
                    cgEIters.push_back(matrix_vars.cgEIters); cgHIters.push_back(matrix_vars.cgHIters);

                }




            }
            t6 = high_resolution_clock::now();

            /*******************************************************************************/
            /* Finished solving linear system or computed matrix-vector products           */
            /*******************************************************************************/

            std::cout<<"Solved linear systems."<<std::endl;

            /**********************************/
            /* Export data during iterations  */
            /**********************************/

            /* Note: value fn solutions at iteration i are solutions to the matrix constructed using equilibrium objects at iteration i, updated using the solutions from iteration (i-1) */

            if ( (i % parameters.exportFreq == 0) && (i > 0) ) {
                //Compute expert leverage
                value_vars.leverageExperts = value_vars.kappa * value_vars.chi / state_vars.omega;
                vars.updateRest(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters);
                std::cout<<"\nBefore exporting data, updated expert leverage and the rest"<<std::endl;

                //Export data
                exportInformation(timeItersVec, timeItersLinSysVec, eErrorsVec, hErrorsVec, cgEIters, cgHIters, parameters);
                exportData(value_vars, vars, derivsXiE, derivsXiH, derivsQ, derivsLogQ, std::to_string(i), state_vars, parameters);

            }


        } else {

            /*************************************************/
            /* Section where tolerance met                   */
            /*************************************************/
            std::cout<<std::endl;
            std::cout<<"========================================================="<<std::endl;
            std::cout<<"END OF ITERATIONS: TOLERANCE MET"<<std::endl;
            std::cout<<"========================================================="<<std::endl;

            // Print out information
            std::cout<<"Error for xi_e: "<<eErrorsVec[i]<<"\n";
            std::cout<<"Error for xi_h: "<<hErrorsVec[i]<<"\n";
            std::cout<<"Tolerance met at iteration "<<i<<std::endl;

            // Update expert and the rest
            std::cout<<"Updating expert leverage and the rest."<<std::endl;
            value_vars.leverageExperts = value_vars.kappa * value_vars.chi / state_vars.omega;
            vars.updateRest(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters);

            // Update chi and kappa if a_e = a_h and Vtilde = 0
            if ( (abs(parameters.a_e - parameters.a_h) < 0.000001) && (state_vars.H.maxCoeff() < 0.00001) ) {
                value_vars.kappa = value_vars.kappa*value_vars.chi;
                value_vars.chi   = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
            }
            std::cout<<"Exporting model solution: value functions, chi, and kappa..."<<std::endl;
            saveMarketVector(value_vars.xi_e, parameters.folderName + "xi_e_" + "final" + ".dat");
            saveMarketVector(value_vars.xi_h, parameters.folderName + "xi_h_" + "final" + ".dat");
            saveMarketVector(value_vars.kappa,parameters.folderName +   "kappa_" +  "final" + ".dat");
            saveMarketVector(value_vars.chi,parameters.folderName +   "chi_" +  "final" + ".dat");
            // Remove first element of errors vectors
            if (!eErrorsVec.empty()) {
                eErrorsVec.erase (eErrorsVec.begin());
                hErrorsVec.erase (hErrorsVec.begin());
            }
            exportInformation(timeItersVec, timeItersLinSysVec, eErrorsVec, hErrorsVec, cgEIters, cgHIters, parameters);
            /*******************************************/
            /* Tolerance met. End iterations           */
            /*******************************************/

            // return status
            return 1;

        }
        t2 = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>( t2 - t1 ).count();
        auto durationLinSys = duration_cast<microseconds>( t6 - t5 ).count();

        /*******************************************/
        /* Print out timing information            */
        /*******************************************/

        std::cout << "Iteration "<<i<<" is completed; time elapsed: "<<duration / 1000000.0 <<"; time spent on solving linear systems (or computing matrix vector product in explicit scheme): "<< durationLinSys / 1000000.0 <<std::endl;

        timeItersVec.push_back(duration / 1000000.0); timeItersLinSysVec.push_back(durationLinSys / 1000000.0);

    }

    /**************************************************/
    /* Only reach this section when the program maxed */
    /* out or the user terminated by ctrl+c (i.e.)    */
    /* terminationCode == 3                           */
    /**************************************************/

    if (terminationCode == -3) {
        /*************************************************/
        /* Section where user terminated program         */
        /*************************************************/

        std::cout<<std::endl;
        std::cout<<"========================================================="<<std::endl;
        std::cout<<"END OF ITERATIONS: USER TERMINATION"<<std::endl;
        std::cout<<"========================================================="<<std::endl;
        // Print out information
        if (!eErrorsVec.empty()) {
            std::cout<<"Error for xi_e: "<<eErrorsVec.back()<<"\n";
            std::cout<<"Error for xi_h: "<<hErrorsVec.back()<<"\n";
            eErrorsVec.erase (eErrorsVec.begin());
            hErrorsVec.erase (hErrorsVec.begin());
        }

        // Update expert and the rest
        std::cout<<"Updating expert leverage and the rest."<<std::endl;
        value_vars.leverageExperts = value_vars.kappa * value_vars.chi / state_vars.omega;
        vars.updateRest(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters);
        if ( (abs(parameters.a_e - parameters.a_h) < 0.000001) && (state_vars.H.maxCoeff() < 0.00001) ) {
            value_vars.kappa = value_vars.chi;
            value_vars.chi   = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
        }
        std::cout<<"Exporting model solution: value functions, chi, and kappa..."<<std::endl;
        saveMarketVector(value_vars.xi_e, parameters.folderName + "xi_e_" + "final" + ".dat");
        saveMarketVector(value_vars.xi_h, parameters.folderName + "xi_h_" + "final" + ".dat");
        saveMarketVector(value_vars.kappa,parameters.folderName +   "kappa_" +  "final" + ".dat");
        saveMarketVector(value_vars.chi,parameters.folderName +   "chi_" +  "final" + ".dat");
        return -3;

    } else {
        /*************************************************/
        /* Section where tolerance not met               */
        /*************************************************/
        std::cout<<std::endl;
        std::cout<<"========================================================="<<std::endl;
        std::cout<<"END OF ITERATIONS: TOLERANCE NOT MET"<<std::endl;
        std::cout<<"========================================================="<<std::endl;

        // Print out information
        std::cout<<"Error for xi_e: "<<eErrorsVec[parameters.maxIters-1]<<"\n";
        std::cout<<"Error for xi_h: "<<hErrorsVec[parameters.maxIters-1]<<"\n";
        std::cout<<"Tolerance not met and max iterations reached: "<<parameters.maxIters<<std::endl;
        // Remove first element of errors vectors
        if (!eErrorsVec.empty()) {
            eErrorsVec.erase (eErrorsVec.begin());
            hErrorsVec.erase (hErrorsVec.begin());
        }
        // Update expert and the rest
        std::cout<<"Updating expert leverage and the rest."<<std::endl;
        value_vars.leverageExperts = value_vars.kappa * value_vars.chi / state_vars.omega;
        vars.updateRest(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters);
        if ( (abs(parameters.a_e - parameters.a_h) < 0.000001) && (state_vars.H.maxCoeff() < 0.00001) ) {
            value_vars.kappa = value_vars.chi;
            value_vars.chi   = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
        }
        // Exporting model solution: value functions, chi, and kappa
        std::cout<<"Exporting model solution: value functions, chi, and kappa..."<<std::endl;
        saveMarketVector(value_vars.xi_e, parameters.folderName + "xi_e_" + "final" + ".dat");
        saveMarketVector(value_vars.xi_h, parameters.folderName + "xi_h_" + "final" + ".dat");
        saveMarketVector(value_vars.kappa,parameters.folderName +   "kappa_" +  "final" + ".dat");
        saveMarketVector(value_vars.chi,parameters.folderName +   "chi_" +  "final" + ".dat");

        /***********************************************************************/
        /* Tolerance not met and max iterations met. End program               */
        /***********************************************************************/

        return 0;
    }
}
