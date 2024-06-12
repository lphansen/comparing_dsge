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


#include "valueVars.h"
valueVars::valueVars () {

}
valueVars::valueVars (stateVars & state_vars, Eigen::ArrayXd xiEguess, Eigen::ArrayXd xiHguess, Eigen::ArrayXd kappaGuess, Eigen::ArrayXd chiGuess) {

    /* Function to initialize class valueVars */

    xi_e.resize(state_vars.S); xi_h.resize(state_vars.S); xi_e_old.resize(state_vars.S); xi_h_old.resize(state_vars.S);
    leverageExperts.resize(state_vars.S);
    xi_e = xiEguess;
    xi_h = xiHguess;
    xi_e_old = xiEguess;
    xi_h_old = xiHguess;
    kappa = kappaGuess;
    kappa_old = kappaGuess;
    chi = chiGuess;
    chi_old = chiGuess;



}
