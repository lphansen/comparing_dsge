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
/* This file contains the header file of the class        */
/* of the derivatives    */
/**********************************************************/
/**********************************************************/


/*********************************************************/
/* Include header files                                  */
/*********************************************************/


#ifndef derivs_h
#define derivs_h

#include <stdio.h>
#include "stateVars.h"
#include "Parameters.h"
#include "common.h"
//extern "C" {

class derivs {

public:
    Eigen::VectorXd varToDiff;
    //first partials
    Eigen::MatrixXd firstPartials;
    std::map <string, Eigen::ArrayXd> firstPartialsMap;

    //second partials
    Eigen::MatrixXd secondPartials;
    std::map <string, Eigen::ArrayXd> secondPartialsMap;

    //cross partials
    Eigen::MatrixXd crossPartials;
    std::map <int, Eigen::ArrayXd> crossPartialsMap;

    // Sparse matrices that represent the linear operators of
    // differentiation
    std::map <string, SpMat> firstPartialsLinearOps;

    //second partials
    std::map <string, SpMat> secondPartialsLinearOps;

    //cross partials
    std::map <int, SpMat> crossPartialsLinearOps;

    //Vectors of triplets that will be used to constrauct linear operators
    std::vector<T> firstPartialList; std::vector<T> secondPartialList;
    std::vector<T> crossPartialList;

    derivs ();
    derivs (stateVars &, Parameters &);
    void computeDerivs(Eigen::ArrayXd f, stateVars & state_vars);
    void computeDerivsOld(Eigen::Ref<Eigen::ArrayXd> f, stateVars & state_vars);
    void computeDerivsValueFn(Eigen::Ref<Eigen::ArrayXd> f, stateVars & state_vars, Eigen::MatrixXd & firstCoefsE, Eigen::MatrixXd & firstCoefsH, string functionName);
    void computeCrossDer(stateVars & state_vars, int stateNum, int stateNum_sub, Eigen::Ref<Eigen::ArrayXd>  f, Eigen::Ref<Eigen::ArrayXd> dfdx);
    void constructLinearOps (stateVars & state_vars);

    int k;
    /*****************************************************/
    /* Other Parameters                                  */
    /*****************************************************/
    bool useLogW;
    int x_star = 0; int y_star = 0;
    int x_step = 0; int y_step = 0;
    int i_star = 0;




};

//}

#endif /* derivs_h */
