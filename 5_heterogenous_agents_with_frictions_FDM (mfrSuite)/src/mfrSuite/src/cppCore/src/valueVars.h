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
/* This file contains the header file of the value and    */
/* policy functions: zetas, chi, and kappa                */
/**********************************************************/
/**********************************************************/

/*********************************************************/
/* Include header files                                  */
/*********************************************************/

#ifndef valueVars_h
#define valueVars_h
// must include files in this order
#include "common.h"
#include "stateVars.h"
#include "Parameters.h"




class valueVars {

public:
    Eigen::ArrayXd xi_e; Eigen::ArrayXd xi_h;
    Eigen::ArrayXd xi_e_old; Eigen::ArrayXd xi_h_old;
    Eigen::ArrayXd kappa; Eigen::ArrayXd chi;
    Eigen::ArrayXd kappa_old; Eigen::ArrayXd chi_old;
    Eigen::ArrayXd kappaStar; Eigen::ArrayXd deltaEStar;
    Eigen::ArrayXd leverageExperts;

    valueVars ();
    valueVars (stateVars &, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd);

};



#endif /* valueVars_h */
