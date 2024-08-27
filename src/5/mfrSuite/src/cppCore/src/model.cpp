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
#include "model.h"
#include "functions.h"

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// This portion includes pybind11 and creating the namespace py
#include <pybind11/pybind11.h>
namespace py = pybind11;
// ---------------------------------------------------------------
// ---------------------------------------------------------------

model::model(int numSds, double sigma_K_norm, double sigma_Z_norm, double sigma_V_norm,
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
             Eigen::ArrayXd kappaGuessInput, double CGscale, int precondFreq) {

    /*********************************/
    /* This function initializes the */
    /* model class                   */
    /*********************************/


    /* Read in volatility parameters */

    parameters.numSds              = numSds;
    parameters.sigma_K_norm        = sigma_K_norm;
    parameters.sigma_Z_norm        = sigma_Z_norm;
    parameters.sigma_V_norm        = sigma_V_norm;
    parameters.sigma_H_norm        = sigma_H_norm;

    /* Read in parameters for the state variables */

    if (logW == -1) {
        parameters.useLogW = false;
    } else if (logW == 1) {
        parameters.useLogW = true;
    }

    if (parameters.useLogW) {
        parameters.omegaMin = log(wMin);
        parameters.omegaMax = log(wMax);
    } else {
        parameters.omegaMin = wMin;
        parameters.omegaMax = wMax;
    }

    parameters.nDims    = nDims;
    parameters.nOmega   = nWealth;
    parameters.nZ       = nZ;
    parameters.nV       = nV;
    parameters.nH       = nH;
    parameters.nShocks  = nShocks;

    if (verbatim == -1) {
        parameters.verbatim = false;
    } else if (verbatim == 1) {
        parameters.verbatim = true;
    }

    /* Iteration parameters */

    parameters.folderName    = folderName + "/";
    parameters.run           = folderName;
    parameters.method        = std::to_string(method);
    parameters.dt            = dt;
    parameters.dtInner       = dtInner;
    parameters.maxIters      = maxIters;
    parameters.maxItersInner = maxItersInner;
    parameters.tol           = tol;
    parameters.innerTol      = innerTol;
    parameters.equityIss     = equityIss;
    parameters.hhCap         = hhCap;
    parameters.preLoad       = preLoad;
    parameters.CGscale       = CGscale;
    parameters.precondFreq   = precondFreq;

    /* Pardiso parameters */

    parameters.iparm_2        = iparm_2;
    parameters.iparm_3        = iparm_3;
    parameters.iparm_28       = iparm_28;
    parameters.iparm_31       = iparm_31;

    /* Model parameters */

    // OLG parameters
    parameters.lambda_d            = lambda_d;
    parameters.nu_newborn          = nu_newborn;

    // Persistence parameters
    parameters.lambda_Z           = lambda_Z;
    parameters.lambda_V           = lambda_V;
    parameters.lambda_H           = lambda_H;

    // Means
    parameters.Z_bar              = Z_bar;
    parameters.V_bar              = V_bar;
    parameters.H_bar              = H_bar;

    // Rate of time preferences
    parameters.delta_e             = delta_e;
    parameters.delta_h             = delta_h;

    // Productivity Parameters
    parameters.a_e               = a_e;
    parameters.a_h               = a_h;

    // Inverses of EIS
    parameters.rho_e               = rho_e;
    parameters.rho_h               = rho_h;

    // Adjustment cost and depreciation
    parameters.phi                   = phi;
    parameters.alpha_K               = alpha_K;

    // Risk aversion
    parameters.gamma_e             = gamma_e;
    parameters.gamma_h             = gamma_h;

    // Equity issuance constraint
    parameters.chiUnderline        = chiUnderline;

    // Read in correlation parameters
    parameters.cov11   = cov11; parameters.cov12   = cov12; parameters.cov13   = cov13; parameters.cov14   = cov14;
    parameters.cov21   = cov21; parameters.cov22   = cov22; parameters.cov23   = cov23; parameters.cov24   = cov24;
    parameters.cov31   = cov31; parameters.cov32   = cov32; parameters.cov33   = cov33; parameters.cov34   = cov34;
    parameters.cov41   = cov41; parameters.cov42   = cov42; parameters.cov43   = cov43; parameters.cov44   = cov44;

    // Read in export frequency
    parameters.exportFreq = exportFreq;

    // Read in guesses
    xiEGuess   = xiEGuessInput;
    xiHGuess   = xiHGuessInput;
    chiGuess     = chiGuessInput;
    kappaGuess   = kappaGuessInput;

    /**********************************************************************/
    /********************** END OF INITIALIZATION *************************/
    /**********************************************************************/



}

void model::reset(int numSds, double sigma_K_norm, double sigma_Z_norm, double sigma_V_norm,
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
                  Eigen::ArrayXd kappaGuessInput, double CGscale, int precondFreq) {

    /*********************************/
    /* This function initializes the */
    /* model class                   */
    /*********************************/


    /* Read in volatility parameters */

    parameters.numSds              = numSds;
    parameters.sigma_K_norm        = sigma_K_norm;
    parameters.sigma_Z_norm        = sigma_Z_norm;
    parameters.sigma_V_norm        = sigma_V_norm;
    parameters.sigma_H_norm        = sigma_H_norm;

    /* Read in parameters for the state variables */

    if (logW == -1) {
        parameters.useLogW = false;
    } else if (logW == 1) {
        parameters.useLogW = true;
    }

    if (parameters.useLogW) {
        parameters.omegaMin = log(wMin);
        parameters.omegaMax = log(wMax);
    } else {
        parameters.omegaMin = wMin;
        parameters.omegaMax = wMax;
    }

    parameters.nDims    = nDims;
    parameters.nOmega   = nWealth;
    parameters.nZ       = nZ;
    parameters.nV       = nV;
    parameters.nH       = nH;
    parameters.nShocks  = nShocks;

    if (verbatim == -1) {
        parameters.verbatim = false;
    } else if (verbatim == 1) {
        parameters.verbatim = true;
    }

    /* Iteration parameters */

    parameters.folderName    = folderName + "/";
    parameters.run           = folderName;
    parameters.method        = std::to_string(method);
    parameters.dt            = dt;
    parameters.dtInner       = dtInner;
    parameters.maxIters      = maxIters;
    parameters.maxItersInner = maxItersInner;
    parameters.tol           = tol;
    parameters.innerTol      = innerTol;
    parameters.equityIss     = equityIss;
    parameters.hhCap         = hhCap;
    parameters.preLoad       = preLoad;
    parameters.CGscale       = CGscale;
    parameters.precondFreq   = precondFreq;

    /* Pardiso parameters */

    parameters.iparm_2        = iparm_2;
    parameters.iparm_3        = iparm_3;
    parameters.iparm_28       = iparm_28;
    parameters.iparm_31       = iparm_31;

    /* Model parameters */

    // OLG parameters
    parameters.lambda_d            = lambda_d;
    parameters.nu_newborn          = nu_newborn;

    // Persistence parameters
    parameters.lambda_Z        = lambda_Z;
    parameters.lambda_V        = lambda_V;
    parameters.lambda_H        = lambda_H;

    // Means
    parameters.Z_bar             = Z_bar;
    parameters.V_bar             = V_bar;
    parameters.H_bar             = H_bar;

    // Rate of time preferences
    parameters.delta_e             = delta_e;
    parameters.delta_h             = delta_h;

    // Productivity Parameters
    parameters.a_e               = a_e;
    parameters.a_h               = a_h;

    // Inverses of EIS
    parameters.rho_e               = rho_e;
    parameters.rho_h               = rho_h;

    // Adjustment cost and depreciation
    parameters.phi                   = phi;
    parameters.alpha_K               = alpha_K;

    // Risk aversion
    parameters.gamma_e             = gamma_e;
    parameters.gamma_h             = gamma_h;

    // Equity issuance constraint
    parameters.chiUnderline        = chiUnderline;

    // Read in correlation parameters
    parameters.cov11   = cov11; parameters.cov12   = cov12; parameters.cov13   = cov13; parameters.cov14   = cov14;
    parameters.cov21   = cov21; parameters.cov22   = cov22; parameters.cov23   = cov23; parameters.cov24   = cov24;
    parameters.cov31   = cov31; parameters.cov32   = cov32; parameters.cov33   = cov33; parameters.cov34   = cov34;
    parameters.cov41   = cov41; parameters.cov42   = cov42; parameters.cov43   = cov43; parameters.cov44   = cov44;

    // Read in export frequency
    parameters.exportFreq = exportFreq;

    // Read in guesses
    xiEGuess   = xiEGuessInput;
    xiHGuess   = xiHGuessInput;
    chiGuess     = chiGuessInput;
    kappaGuess   = kappaGuessInput;

    /**********************************************************************/
    /********************** END OF MODEL RESET    *************************/
    /**********************************************************************/



}

int model::organizeData() {
    // This function is called when the model is solved to organize data for exports.

    derivsXiE_first.resize(state_vars.S, state_vars.N);
    derivsXiH_first.resize(state_vars.S, state_vars.N);
    derivsQ_first.resize(state_vars.S, state_vars.N);
    derivsLogQ_first.resize(state_vars.S, state_vars.N);
    derivsXiE_second.resize(state_vars.S, state_vars.N);
    derivsXiH_second.resize(state_vars.S, state_vars.N);
    derivsQ_second.resize(state_vars.S, state_vars.N);
    derivsLogQ_second.resize(state_vars.S, state_vars.N);
    muX.resize(state_vars.S, state_vars.N);
    sigmaX.resize(state_vars.S, state_vars.N * parameters.nShocks);

    int ch = choose(state_vars.N, 2); // Total number of cross partials is (N choose 2)
    derivsXiE_cross.resize(state_vars.S, ch);
    derivsXiH_cross.resize(state_vars.S, ch);
    derivsQ_cross.resize(state_vars.S, ch);
    derivsLogQ_cross.resize(state_vars.S, ch);

    if (state_vars.N > 1) {
        for (int n = 0; n < ch; n++) {
            derivsXiE_cross.col(n)    = derivsXiE.crossPartialsMap[n];
            derivsXiH_cross.col(n)    = derivsXiH.crossPartialsMap[n];
            derivsQ_cross.col(n)        = derivsQ.crossPartialsMap[n];
            derivsLogQ_cross.col(n)     = derivsLogQ.crossPartialsMap[n];

        }
    }


    for (int n = 0; n < state_vars.N; n++) {
        derivsXiE_first.col(n) = derivsXiE.firstPartialsMap[state_vars.num2State[n]];
        derivsXiH_first.col(n) = derivsXiH.firstPartialsMap[state_vars.num2State[n]];
        derivsQ_first.col(n)     = derivsQ.firstPartialsMap[state_vars.num2State[n]];
        derivsLogQ_first.col(n)  = derivsLogQ.firstPartialsMap[state_vars.num2State[n]];

        derivsXiE_second.col(n) = derivsXiE.secondPartialsMap[state_vars.num2State[n]];
        derivsXiH_second.col(n) = derivsXiH.secondPartialsMap[state_vars.num2State[n]];
        derivsQ_second.col(n)     = derivsQ.secondPartialsMap[state_vars.num2State[n]];
        derivsLogQ_second.col(n)  = derivsLogQ.secondPartialsMap[state_vars.num2State[n]];

        muX.col(n) = vars.muXMap[state_vars.num2State[n]];

        for (int s = 0; s < parameters.nShocks; s++) {

            sigmaX.col(n * parameters.nShocks + s) = vars.sigmaXMap[state_vars.num2State[n]].col(s);
        }
    }
    return 0;

}
void model::smoothDataCPP(Eigen::ArrayXd smoothedQ, Eigen::ArrayXd smoothedKappa) {
    vars.q = smoothedQ; value_vars.kappa = smoothedKappa;
    vars.logQ = vars.q.log();

    derivsLogQ.computeDerivs(vars.logQ, state_vars);
    derivsQ.computeDerivs(vars.q, state_vars);
    derivsKappa.computeDerivs(value_vars.kappa, state_vars);
    vars.updateSigmaPi(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, parameters);
    vars.updateMuAndR(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, parameters);
    vars.updateRest(state_vars, value_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters);
    value_vars.leverageExperts = value_vars.kappa * value_vars.chi / state_vars.omega;


}
int model::solveModel() {
    // Export parameters
    parameters.save_output();
    // Export data to log.txt instead of printing it out

    std::ofstream out(parameters.folderName + "log.txt");
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

    // Export parameters again
    parameters.save_output();
    // Compute the upper and lower limits of the state space and prepare for initialization

    Eigen::ArrayXd upper;
    Eigen::ArrayXd lower;
    Eigen::ArrayXd sizes;

    /* Upper and lower boundaries of the exogenous state variables */

    // min/max for s
    double shape = 2 * parameters.lambda_V * parameters.V_bar / (pow(parameters.sigma_V_norm,2));
    double rate = 2 * parameters.lambda_V / (pow(parameters.sigma_V_norm,2));
    parameters.vMin = 0.00001;
    parameters.vMax = parameters.V_bar + parameters.numSds * sqrt( shape / pow(rate, 2));

    // min/max for g
    double zVar = pow(parameters.V_bar * parameters.sigma_Z_norm, 2) / (2 * parameters.lambda_Z);
    parameters.zMin = parameters.Z_bar - parameters.numSds * sqrt( zVar );
    parameters.zMax = parameters.Z_bar + parameters.numSds * sqrt( zVar );

    // min/max for H
    double shape_H = 2 * parameters.lambda_H * parameters.H_bar / (pow(parameters.sigma_H_norm,2));
    double rate_H = 2 * parameters.lambda_H / (pow(parameters.sigma_H_norm,2));
    parameters.hMin = 0.00001;
    parameters.hMax = parameters.H_bar + parameters.numSds * sqrt(shape_H / pow(rate_H,2));


    /* Input parameters into Eigen::Arrays for intialization */

    if (parameters.nDims == 3) {

        // This part determines which variable is a state variable by usign the variance.
        // For example, if sigma_H_norm is zero, then H is not a state varaible.
        if (parameters.sigma_H_norm < 0.0000001) {
            /* Input parameters into Eigen::Arrays for intialization */
            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.zMax, parameters.vMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.zMin, parameters.vMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nZ, parameters.nV;
        } else if (parameters.sigma_Z_norm < 0.0000001) {
            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.vMax, parameters.hMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.vMin, parameters.hMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nV, parameters.nH;
        } else if (parameters.sigma_V_norm < 0.0000001) {
            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.zMax, parameters.hMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.zMin, parameters.hMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nZ, parameters.nH;
        }


    } else if (parameters.nDims == 2) {


        // This part picks the state variable with positive variance and makes it a state variable.
        // Note: The Python interface will prompt an error if nDims = 2 and there are two exogenous
        //       state variables with positive variance.

        if (parameters.sigma_Z_norm > 0.00000001) {

            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.zMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.zMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nZ;

        } else if ( parameters.sigma_V_norm > 0.00000001 ) {

            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.vMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.vMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nV;

        } else if ( parameters.sigma_H_norm > 0.00000001 ) {

            upper.resize(parameters.nDims); upper << parameters.omegaMax, parameters.hMax;
            lower.resize(parameters.nDims); lower << parameters.omegaMin, parameters.hMin;
            sizes.resize(parameters.nDims); sizes << parameters.nOmega, parameters.nH;

        }


    } else if (parameters.nDims == 1) {

        /* Input parameters into Eigen::Arrays for intialization */
        upper.resize(parameters.nDims); upper << parameters.omegaMax;
        lower.resize(parameters.nDims); lower << parameters.omegaMin;
        sizes.resize(parameters.nDims); sizes << parameters.nOmega;

    }

    //*************************************************//
    /* Initializing state variables, guesses, etc.    */
    //*************************************************//
    // Initialize state variables
    state_vars = stateVars(upper, lower, sizes, parameters);
    // Fill in default values for chi and kappa

    if (kappaGuess.maxCoeff() < 0) {
        kappaGuess = state_vars.omega;
    }

    if (chiGuess.maxCoeff() < 0) {
        chiGuess   =  Eigen::MatrixXd::Constant(state_vars.S, 1, parameters.chiUnderline);
    }


    value_vars = valueVars (state_vars, xiEGuess, xiHGuess, kappaGuess, chiGuess);

    // Initialize equilibrium quantities

    Eigen::ArrayXd qGuess; qGuess = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.2);
    vars = Vars(state_vars, qGuess, parameters);

    // Initialize matrices
    matrix_vars = matrixVars();
    std::cout<<"Finished creating empty matrices"<<std::endl;
    matrix_vars = matrixVars(state_vars, parameters);

    // Initialize derivatives
    derivsXiE    = derivs();       derivsXiE    = derivs(state_vars, parameters);
    derivsXiH    = derivs();       derivsXiH    = derivs(state_vars, parameters);
    derivsLogQ     = derivs();       derivsLogQ     = derivs(state_vars, parameters);
    derivsQ        = derivs();       derivsQ        = derivs(state_vars, parameters);
    derivsKappa    = derivs();       derivsKappa    = derivs(state_vars, parameters);
    derivsLogABar  = derivs();       derivsLogABar  = derivs(state_vars, parameters);

    int status = -2; // -2: unsolved; -1: error; 0: tolerance not met after max num of iterations; 1: toleration met


    timeItersVec.clear(); timeItersLinSysVec.clear();
    eErrorsVec.clear();   hErrorsVec.clear();
    cgEIters.clear();     cgHIters.clear();

    parameters.save_output();
    status = iterFunc(state_vars, value_vars, vars, matrix_vars, derivsXiE, derivsXiH, derivsLogQ, derivsQ, derivsKappa, derivsLogABar, parameters,
                      timeItersVec, timeItersLinSysVec, eErrorsVec, hErrorsVec, cgEIters, cgHIters);
    model::organizeData();

    std::cout.rdbuf(coutbuf); // Redirect cout

    // Export parameters again
    parameters.save_output();
    return status;
}

void model::dumpData() {

    /* This function exports the numerical results */
    /* Should be caleld after model is solved. */
    exportInformation(timeItersVec, timeItersLinSysVec, eErrorsVec, hErrorsVec, cgEIters, cgHIters, parameters);
    exportData(value_vars, vars, derivsXiE, derivsXiH,  derivsQ, derivsLogQ, "final", state_vars, parameters);
    exportPDE(matrix_vars, value_vars, derivsXiE, derivsXiH, parameters, state_vars, "final");
}

/*************************************/
/* Using pybind11 to interface       */
/* with python                       */
/*************************************/

PYBIND11_MODULE(modelSolnCore, m) {
    m.doc() = R"pbdoc(
    MFM Suite HKT Model C++ Core
    )pbdoc";

    py::class_<model>(m, "model")
    .def(py::init<int, double, double, double,
                       double, int, double, double,
                       int, int, int, int, int, int,
                       int, string, string, int,
                       double, double, int, int,
                       double, double, int, int, int,
                       int, int, int, double, double,
                       double, double, double, double,
                       double, double, double, double,
                       double, double, double, double, double,
                       double, double, double, double,
                       double, double, double, double,
                       double, double, double, double,
                       double, double, double, double,
         double, double, double, double, int, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, double, int>())
    .def("solveModel", &model::solveModel)
    .def("dumpData", &model::dumpData)
    .def("reset", &model::reset)
    .def("xiE", &model::getXi_e)
    .def("xiH", &model::getXi_h)
    .def("chatE", &model::getChatE)
    .def("chatH", &model::getChatH)
    .def("CeOverCh", &model::CeOverCh)
    .def("CoverI", &model::getCoverI)
    .def("IoverK", &model::getIoverK)
    .def("CoverK", &model::getCoverK)
    .def("IoverY", &model::getIoverY)
    .def("CoverY", &model::getCoverY)
    .def("kappa", &model::getKappa)
    .def("chi", &model::getChi)
    .def("betaE", &model::getBetaE)
    .def("betaH", &model::getBetaH)
    .def("q", &model::getQ)
    .def("r", &model::getR)
    .def("deltaE", &model::getDeltaE)
    .def("deltaH", &model::getDeltaH)
    .def("I", &model::getI)
    .def("muQ", &model::getMuQ)
    .def("muK", &model::getMuK)
    .def("muY", &model::getMuY)
    .def("muPhi", &model::getMuPhi)
    .def("leverage", &model::getLeverage)
    .def("muC", &model::getMuC)
    .def("muCe", &model::getMuCe)
    .def("muCh", &model::getMuCh)
    .def("muSe", &model::getMuSe)
    .def("muSh", &model::getMuSh)
    .def("piE", &model::getPiE)
    .def("piH", &model::getPiH)
    .def("piETilde", &model::getPiETilde)
    .def("piHTilde", &model::getPiHTilde)
    .def("muRe", &model::getMuRe)
    .def("muRh", &model::getMuRh)
    .def("sigmaQ", &model::getSigmaQ)
    .def("sigmaR", &model::getSigmaR)
    .def("sigmaK", &model::getSigmaK)
    .def("sigmaY", &model::getSigmaY)
    .def("sigmaPhi", &model::getSigmaPhi)
    .def("sigmaC", &model::getSigmaC)
    .def("sigmaCe", &model::getSigmaCe)
    .def("sigmaCh", &model::getSigmaCh)
    .def("sigmaSe", &model::getSigmaSe)
    .def("sigmaSh", &model::getSigmaSh)
    .def("W", &model::getW)
    .def("logW", &model::getLogW)
    .def("Z", &model::getZ)
    .def("V", &model::getV)
    .def("H", &model::getH)
    .def_readwrite("timeOuterloop", &model::timeItersVec)
    .def_readwrite("timeLinSys", &model::timeItersLinSysVec)
    .def_readwrite("errorsE", &model::eErrorsVec)
    .def_readwrite("errorsH", &model::hErrorsVec)
    .def_readwrite("CGEIters", &model::cgEIters)
    .def_readwrite("CGHIters", &model::cgHIters)
    .def("derivsXiE_first", &model::getderivsXiE_first)
    .def("derivsXiH_first", &model::getderivsXiH_first)
    .def("derivsQ_first", &model::getDerivsQ_first)
    .def("derivsLogQ_first", &model::getDerivsLogQ_first)
    .def("derivsXiE_second", &model::getderivsXiE_second)
    .def("derivsXiH_second", &model::getderivsXiH_second)
    .def("derivsQ_second", &model::getDerivsQ_second)
    .def("derivsLogQ_second", &model::getDerivsLogQ_second)
    .def("derivsXiE_cross", &model::getderivsXiE_cross)
    .def("derivsXiH_cross", &model::getderivsXiH_cross)
    .def("derivsQ_cross", &model::getDerivsQ_cross)
    .def("derivsLogQ_cross", &model::getDerivsLogQ_cross)
    .def("muX", &model::getMuX)
    .def("sigmaX", &model::getSigmaX)
    .def("smoothDataCPP", &model::smoothDataCPP)


    ;


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
