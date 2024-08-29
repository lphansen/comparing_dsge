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


#include "Vars.h"

Vars::Vars() {

}
Vars::Vars(stateVars & state_vars, Eigen::ArrayXd qGuess, Parameters & parameters) {


    //Initialize variables related to q
    q.resize(state_vars.S); q_old.resize(state_vars.S); qStar.resize(state_vars.S); oneOmegaQ.resize(state_vars.S); logQ.resize(state_vars.S); omegaQ.resize(state_vars.S); I.resize(state_vars.S);
    q = qGuess; q_old = q;

    sigmaK.resize(state_vars.S, parameters.nShocks); sigmaQ.resize(state_vars.S, parameters.nShocks); sigmaR.resize(state_vars.S, parameters.nShocks);
    std::cout<<"Finished resizing sigmaK, sigmaQ, sigmaR"<<std::endl;
    sigmaK.setZero(); sigmaQ.setZero(); sigmaR.setZero();
    std::cout<<"Finished setting sigmaK, sigmaQ, sigmaR to zero"<<std::endl;


    normR2.resize(state_vars.S);
    Pi.resize(state_vars.S, parameters.nShocks); Pi.setZero();
    PiE.resize(state_vars.S, parameters.nShocks); PiE.setZero();
    deltaE.resize(state_vars.S); deltaEStar.resize(state_vars.S); deltaH.resize(state_vars.S); deltaE_last.resize(state_vars.S);
    muK.resize(state_vars.S);

    trace.resize(state_vars.S);
    r.resize(state_vars.S);
    muQ.resize(state_vars.S);
    muX.resize(state_vars.S, 4); muX.setZero();
    muRe.resize(state_vars.S);
    muRh.resize(state_vars.S);
    cHat_e.resize(state_vars.S);
    cHat_h.resize(state_vars.S);
    CeOverCh.resize(state_vars.S);
    beta_e.resize(state_vars.S); betaEDeltaE.resize(state_vars.S);
    beta_h.resize(state_vars.S); betaHDeltaH.resize(state_vars.S);

    Dx.resize(state_vars.S, parameters.nShocks); Dx.setZero();
    DzetaOmega.resize(state_vars.S); DzetaOmega = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0);
    DzetaX.resize(state_vars.S); DzetaX = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0);

    traceE.resize(state_vars.S); traceH.resize(state_vars.S); traceKappa.resize(state_vars.S);
    traceQ.resize(state_vars.S);
    muY.resize(state_vars.S); muY.setZero(); sigmaY.resize(state_vars.S, parameters.nShocks); sigmaY.setZero();
    IoverK.resize(state_vars.S); CoverI.resize(state_vars.S);
    muLogA.resize(state_vars.S); sigmaLogA.resize(state_vars.S, parameters.nShocks); sigmaLogA.setZero();
    muCe.resize(state_vars.S); sigmaCe.resize(state_vars.S, parameters.nShocks); sigmaCe.setZero();
    muCh.resize(state_vars.S); sigmaCh.resize(state_vars.S, parameters.nShocks); sigmaCh.setZero();
    muC.resize(state_vars.S); sigmaC.resize(state_vars.S, parameters.nShocks); sigmaC.setZero();
    muSe.resize(state_vars.S); sigmaSe.resize(state_vars.S, parameters.nShocks); sigmaSe.setZero();
    muSh.resize(state_vars.S); sigmaSh.resize(state_vars.S, parameters.nShocks); sigmaSh.setZero();
    muPhi.resize(state_vars.S); sigmaPhi.resize(state_vars.S, parameters.nShocks); sigmaPhi.setZero();
    CeOverC.resize(state_vars.S); ChOverC.resize(state_vars.S);
    idenMat.resize(state_vars.N, state_vars.N); derivs_temp.resize(state_vars.N, 1); sigmaX_temp.resize(state_vars.N, state_vars.N);


}

void Vars::updateSigmaPi(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters) {




    sigmaQ =  ( sigmaK.array().colwise() *  ( derivsLogQ.firstPartialsMap["w"] * ( value_vars.kappa * value_vars.chi - state_vars.omega ) )  + ( sigmaXMap["Z"].array().colwise() *  derivsLogQ.firstPartialsMap["Z"]  ) +  ( sigmaXMap["V"].array().colwise() *  derivsLogQ.firstPartialsMap["V"]  ) + ( sigmaXMap["H"].array().colwise() *  derivsLogQ.firstPartialsMap["H"]  ) ).array().colwise() / ( 1.0 - (value_vars.kappa * value_vars.chi - state_vars.omega ) * derivsLogQ.firstPartialsMap["w"]  );

    sigmaR = sigmaQ + sigmaK;

    normR2 = sigmaR.rowwise().norm().array().pow(2);


    sigmaXMap["w"] = (sigmaR).array().colwise() * ( (value_vars.kappa * value_vars.chi - state_vars.omega) );

    Pi =  (sigmaR).array().colwise() * (  parameters.gamma_h * (1.0 - value_vars.chi * value_vars.kappa) / (1.0 - state_vars.omega)  ) + (parameters.gamma_h - 1.0) * ( sigmaXMap["w"].array().colwise() * derivsXiH.firstPartialsMap["w"] + sigmaXMap["Z"].array().colwise() * derivsXiH.firstPartialsMap["Z"] + sigmaXMap["V"].array().colwise() * derivsXiH.firstPartialsMap["V"] + sigmaXMap["H"].array().colwise() * derivsXiH.firstPartialsMap["H"] );


    PiE = (sigmaR).array().colwise() * (parameters.gamma_e * value_vars.chi * value_vars.kappa / state_vars.omega );
    PiE = PiE.array()  + (parameters.gamma_e - 1) * ( sigmaXMap["w"].array().colwise() * derivsXiE.firstPartialsMap["w"] + sigmaXMap["Z"].array().colwise() * derivsXiE.firstPartialsMap["Z"] + sigmaXMap["V"].array().colwise() * derivsXiE.firstPartialsMap["V"] + sigmaXMap["H"].array().colwise() * derivsXiE.firstPartialsMap["H"] );


};

void Vars::updateMuAndR(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters) {




    muXMap["w"]        = state_vars.omega * (1.0 - state_vars.omega) * ( pow(parameters.delta_h, 1/parameters.rho_h) * value_vars.xi_h.exp().pow(1-1/parameters.rho_h) - pow(parameters.delta_e, 1/parameters.rho_e) * value_vars.xi_e.exp().pow(1-1/parameters.rho_e) + beta_e * deltaE - beta_h * deltaH ) + (value_vars.chi * value_vars.kappa - state_vars.omega) * ( sigmaR.cwiseProduct(Pi - sigmaR).rowwise().sum().array() ) + parameters.lambda_d * ( parameters.nu_newborn - state_vars.omega);
    // To compute the trace term, first handle the second partials; after that, handle the cross partials.
    //   Second partials
    trace = derivsQ.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct( sigmaXMap["w"] ).rowwise().sum().array() ) + derivsQ.secondPartialsMap["Z"] * ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsQ.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array() ) + derivsQ.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() ) ;
    //    Cross partials

    k             = choose(state_vars.N, 2) - 1;

    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            trace = trace + 2.0 * derivsQ.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            k = k - 1;

        }
    }


    muQ = 1.0 / q * (muXMap["w"] * derivsQ.firstPartialsMap["w"] + muXMap["Z"] * derivsQ.firstPartialsMap["Z"] + muXMap["V"] * derivsQ.firstPartialsMap["V"] + muXMap["H"] * derivsQ.firstPartialsMap["H"] + 0.5 * trace);


    r = muQ + muK + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array() - sigmaR.cwiseProduct(Pi).rowwise().sum().array() - (1.0 - state_vars.omega) * (beta_h * deltaH - pow(parameters.delta_h, 1/parameters.rho_h) * value_vars.xi_h.exp().pow(1-1/parameters.rho_h) ) - state_vars.omega * (beta_e * deltaE - pow(parameters.delta_e, 1/parameters.rho_e) * value_vars.xi_e.exp().pow(1-1/parameters.rho_e) );

    muRe = (parameters.a_e - 1.0/parameters.phi * ( exp(parameters.phi * I) - 1.0 ) ) / q + I - parameters.alpha_K + 0.04*state_vars.Z + muQ + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array();
    muRh = (parameters.a_h - 1.0/parameters.phi * ( exp(parameters.phi * I) - 1.0 ) ) / q + I - parameters.alpha_K + 0.04*state_vars.Z + muQ + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array();


};


void Vars::updateDeltaEtAl(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters) {

    deltaE = pow(parameters.chiUnderline, -1.0) * (muRe - r - Pi.cwiseProduct(sigmaR).rowwise().sum().array() );
    deltaE = ( deltaE < 0 ).cast<double>() * 0 + ( deltaE >= 0 ).cast<double>() * deltaE;  //impose lower bound of 0 on detalE
    deltaH = (muRh - r - Pi.cwiseProduct(sigmaR).rowwise().sum().array() );

    cHat_e = pow(parameters.delta_e, 1/parameters.rho_e) * value_vars.xi_e.exp().pow(1-1/parameters.rho_e);
    cHat_h = pow(parameters.delta_h, 1/parameters.rho_h) * value_vars.xi_h.exp().pow(1-1/parameters.rho_h);

};


void Vars::updateDerivs(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters) {

    //compute derivs
    derivsXiE.computeDerivs(value_vars.xi_e, state_vars);
    derivsXiH.computeDerivs(value_vars.xi_h, state_vars);
    derivsLogQ.computeDerivs(logQ, state_vars);
    derivsQ.computeDerivs(q, state_vars);


};

void Vars::updateRest(stateVars & state_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, derivs & derivsKappa, derivs & derivsLogABar, Parameters & parameters) {

    derivsKappa.computeDerivs(value_vars.kappa, state_vars);

    /********** Trace Terms ************/

    // First, handle second partials
    traceE = derivsXiE.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() ) + derivsXiE.secondPartialsMap["Z"]* ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsXiE.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array()) + derivsXiE.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() );

    traceH = derivsXiH.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() ) + derivsXiH.secondPartialsMap["Z"] * ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsXiH.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array() ) + derivsXiH.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() );

    traceKappa = derivsKappa.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() ) + derivsKappa.secondPartialsMap["Z"] * ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsKappa.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array()) + derivsKappa.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() );

    traceLogA = derivsLogABar.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() ) + derivsLogABar.secondPartialsMap["Z"] * ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsLogABar.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array()) + derivsLogABar.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() );

    traceQ = derivsQ.secondPartialsMap["w"] * ( sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() ) + derivsQ.secondPartialsMap["Z"] * ( sigmaXMap["Z"].cwiseProduct(sigmaXMap["Z"]).rowwise().sum().array() ) + derivsQ.secondPartialsMap["V"] * ( sigmaXMap["V"].cwiseProduct(sigmaXMap["V"]).rowwise().sum().array()) + derivsQ.secondPartialsMap["H"] * ( sigmaXMap["H"].cwiseProduct(sigmaXMap["H"]).rowwise().sum().array() );

    // Second, handle cross partials
    k             = choose(state_vars.N, 2) - 1;

    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            traceE = traceE + 2.0 * derivsXiE.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            traceH = traceH + 2.0 * derivsXiH.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            traceKappa = traceKappa + 2.0 * derivsKappa.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            traceLogA = traceLogA + 2.0 * derivsLogABar.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            traceQ = traceQ + 2.0 * derivsQ.crossPartialsMap[k].array() * ( sigmaXMap[state_vars.num2State[n]].cwiseProduct(sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );

            k = k - 1;

        }
    }


    /* Individual consumption */
    muCe = (1.0 - 1.0 / parameters.rho_e) * ( muXMap["w"] * derivsXiE.firstPartialsMap["w"] + muXMap["Z"] * derivsXiE.firstPartialsMap["Z"] + muXMap["V"] * derivsXiE.firstPartialsMap["V"] + muXMap["H"] * derivsXiE.firstPartialsMap["H"]  + 0.5 * traceE) + muQ - 0.5 * sigmaQ.cwiseProduct(sigmaQ).rowwise().sum().array() + muK - 0.5 * sigmaK.cwiseProduct(sigmaK).rowwise().sum().array() + muXMap["w"]  / state_vars.omega - 0.5 * sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() / (state_vars.omega.pow(2));

    muCh = (1.0 - 1.0 / parameters.rho_h) * ( muXMap["w"] * derivsXiH.firstPartialsMap["w"] + muXMap["Z"] * derivsXiH.firstPartialsMap["Z"] + muXMap["V"] * derivsXiH.firstPartialsMap["V"] + muXMap["H"] * derivsXiH.firstPartialsMap["H"] + 0.5 * traceH) + muQ - 0.5 * sigmaQ.cwiseProduct(sigmaQ).rowwise().sum().array() + muK - 0.5 * sigmaK.cwiseProduct(sigmaK).rowwise().sum().array() - muXMap["w"] / (1.0 - state_vars.omega) - 0.5 * sigmaXMap["w"].cwiseProduct(sigmaXMap["w"]).rowwise().sum().array() / ( (1.0 - state_vars.omega).pow(2));


    sigmaCe = sigmaQ.array() + sigmaK.array() + sigmaXMap["w"].array().colwise() / state_vars.omega + (1.0 - 1.0 / parameters.rho_e) * ( sigmaXMap["w"].array().colwise() * derivsXiE.firstPartialsMap["w"] + sigmaXMap["Z"].array().colwise() * derivsXiE.firstPartialsMap["Z"] + sigmaXMap["V"].array().colwise() * derivsXiE.firstPartialsMap["V"] + sigmaXMap["H"].array().colwise() * derivsXiE.firstPartialsMap["H"] );

    sigmaCh = sigmaQ.array() + sigmaK.array() - sigmaXMap["w"].array().colwise() / (1.0 - state_vars.omega) + (1.0 - 1.0 / parameters.rho_h) * ( sigmaXMap["w"].array().colwise() * derivsXiH.firstPartialsMap["w"] + sigmaXMap["Z"].array().colwise() * derivsXiH.firstPartialsMap["Z"] + sigmaXMap["V"].array().colwise() * derivsXiH.firstPartialsMap["V"] + sigmaXMap["H"].array().colwise() * derivsXiH.firstPartialsMap["H"] );

    CeOverC = state_vars.omega * pow(parameters.delta_e, 1.0 / parameters.rho_e) * value_vars.xi_e.exp().pow(1.0 - 1.0 / parameters.rho_e) / ( state_vars.omega * pow(parameters.delta_e, 1.0 / parameters.rho_e) * value_vars.xi_e.exp().pow(1.0 - 1.0 / parameters.rho_e) + (1.0 - state_vars.omega) * pow(parameters.delta_h, 1.0 / parameters.rho_h) * value_vars.xi_h.exp().pow(1.0 - 1.0 / parameters.rho_h) );
    ChOverC = 1.0 - CeOverC;

    /* Aggregate consumption */

    sigmaC = sigmaCe.array().colwise() * CeOverC + sigmaCh.array().colwise() * ChOverC;

    muC = CeOverC * muCe + ChOverC * muCh + 0.5 * CeOverC * sigmaCe.cwiseProduct(sigmaCe).rowwise().sum().array() + 0.5 * ChOverC * sigmaCh.cwiseProduct(sigmaCh).rowwise().sum().array() - 0.5 * sigmaC.cwiseProduct(sigmaC).rowwise().sum().array() ;

    /* Output */

    muY = (parameters.a_e - parameters.a_h) / aBar * (muXMap["w"] * derivsKappa.firstPartialsMap["w"] + muXMap["Z"] * derivsKappa.firstPartialsMap["Z"] +
    muXMap["V"] * derivsKappa.firstPartialsMap["V"] + muXMap["H"] * derivsKappa.firstPartialsMap["H"] ) +
    0.5 * ( (parameters.a_e - parameters.a_h) / aBar * traceKappa + ( (parameters.a_e - parameters.a_h) / aBar ).pow(2)
     * (( sigmaXMap["w"].array().colwise() * derivsKappa.firstPartialsMap["w"] +
    sigmaXMap["Z"].array().colwise() * derivsKappa.firstPartialsMap["Z"] +
    sigmaXMap["V"].array().colwise() * derivsKappa.firstPartialsMap["V"] +
    sigmaXMap["H"].array().colwise() * derivsKappa.firstPartialsMap["H"] ) ).matrix().rowwise().norm().array().pow(2) ) + 0.04*state_vars.Z + I - parameters.alpha_K + ( sigmaXMap["w"].cwiseProduct(sigmaK).rowwise().sum().array() * derivsLogABar.firstPartialsMap["w"] + sigmaXMap["Z"].cwiseProduct(sigmaK).rowwise().sum().array() * derivsLogABar.firstPartialsMap["Z"] + sigmaXMap["V"].cwiseProduct( sigmaK ).rowwise().sum().array() * derivsLogABar.firstPartialsMap["V"] + sigmaXMap["H"].cwiseProduct( sigmaK ).rowwise().sum().array() * derivsKappa.firstPartialsMap["H"] );


    sigmaY = ( sigmaK.array() + ( sigmaXMap["w"].array().colwise() * derivsKappa.firstPartialsMap["w"] +
    sigmaXMap["Z"].array().colwise() * derivsKappa.firstPartialsMap["Z"] +
    sigmaXMap["V"].array().colwise() * derivsKappa.firstPartialsMap["V"] +
    sigmaXMap["H"].array().colwise() * derivsKappa.firstPartialsMap["H"] ).array().colwise()  / aBar * (parameters.a_e - parameters.a_h) );

    /* Investment Rate */

    muPhi = (1.0 / (q - 1.0)) * ( muXMap["w"] * derivsQ.firstPartialsMap["w"] +
    muXMap["Z"] * derivsQ.firstPartialsMap["Z"] + muXMap["V"] * derivsQ.firstPartialsMap["V"] + muXMap["H"] * derivsQ.firstPartialsMap["H"]) +
    0.5 * ( (1.0 / (q - 1.0)) * traceQ + (1.0 / (q - 1.0).pow(2)) * ( ( sigmaXMap["w"].array().colwise() * derivsLogABar.firstPartialsMap["w"] +
    sigmaXMap["Z"].array().colwise() * derivsQ.firstPartialsMap["Z"] +
    sigmaXMap["V"].array().colwise() * derivsQ.firstPartialsMap["V"] +
    sigmaXMap["H"].array().colwise() * derivsQ.firstPartialsMap["H"] ) ).matrix().rowwise().norm().array().pow(2) ) +
    muK;

    sigmaPhi =   (  ( sigmaXMap["w"].array().colwise() * derivsQ.firstPartialsMap["w"] +
    sigmaXMap["Z"].array().colwise() * derivsQ.firstPartialsMap["Z"] +
    sigmaXMap["V"].array().colwise() * derivsQ.firstPartialsMap["V"] +
    sigmaXMap["H"].array().colwise() * derivsQ.firstPartialsMap["H"] ) ).array().colwise()  / (q - 1.0) + sigmaK.array();

    /* SDFs */
    sigmaSh = -1.0 * Pi;
    muSh = -r - 0.5 * sigmaSh.cwiseProduct(sigmaSh).rowwise().sum().array();
    sigmaSe = -1.0 * PiE;
    muSe = -r - 0.5 * sigmaSe.cwiseProduct(sigmaSe).rowwise().sum().array();

    /* Consumptio-wealth ratios */

    cHat_e = pow(parameters.delta_e, 1/parameters.rho_e) * value_vars.xi_e.exp().pow(1-1/parameters.rho_e);
    cHat_h = pow(parameters.delta_h, 1/parameters.rho_h) * value_vars.xi_h.exp().pow(1-1/parameters.rho_h);

    CeOverCh = cHat_e * state_vars.omega / (cHat_h * (1.0 - state_vars.omega));
    /* Investment-capital and Consumption-investment ratios */
    IoverK = (q - 1.0) / parameters.phi;
    CoverI = parameters.phi * q/(q - 1.0) * ( (1.0 - state_vars.omega) * cHat_e + state_vars.omega * cHat_h);
    CoverK = parameters.a_e * value_vars.kappa + parameters.a_h * (1.0 - value_vars.kappa) - IoverK;
    IoverY = IoverK / ( parameters.a_e * value_vars.kappa + parameters.a_h * (1.0 - value_vars.kappa) );
    CoverY = 1.0 - IoverY;

    /* Idio risk prices */
    piETilde = parameters.gamma_e * value_vars.chi * value_vars.kappa / state_vars.omega * state_vars.sqrtH;
    piHTilde = parameters.gamma_h * (1.0 - value_vars.kappa) / (1.0 - state_vars.omega) * state_vars.sqrtH;
};
