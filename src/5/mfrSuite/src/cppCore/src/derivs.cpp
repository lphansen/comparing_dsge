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


#include "derivs.h"
#include "Parameters.h"



// Compute cross partial derivatives given stateVars
void derivs::computeCrossDer(stateVars & state_vars, int stateNum, int stateNum_sub, Eigen::Ref<Eigen::ArrayXd>  f, Eigen::Ref<Eigen::ArrayXd> dfdx) {

    // Approximation technique: use the standard formula outlined here: https://en.wikipedia.org/wiki/Finite_difference

    // If at the boundary, use the point next to it
    int x_step = int(state_vars.increVec(stateNum));
    int y_step = int(state_vars.increVec(stateNum_sub));
    int x_star = 0; int y_star = 0;
    int i_star = 0;

    for (int i = 0; i < state_vars.S; i++) {

        //Check location of the first variable
        if ( abs(state_vars.stateMat(i, stateNum) - state_vars.upperLims(stateNum)) < state_vars.dVec(stateNum)/2 ) {
            //at the upper boundary, use the point below it
            i_star = i - int(state_vars.increVec(stateNum));
        } else if ( abs(state_vars.stateMat(i, stateNum) - state_vars.lowerLims(stateNum)) < state_vars.dVec(stateNum)/2 )  {
            //at the lower boundary, use the point above it
            i_star = i + int(state_vars.increVec(stateNum));
        } else {
            i_star = i;
        }

        //Check location of the second variable
        if ( abs(state_vars.stateMat(i, stateNum_sub) - state_vars.upperLims(stateNum_sub)) < state_vars.dVec(stateNum_sub)/2 ) {
            //at the upper boundary, use the point below it
            i_star = i_star - int(state_vars.increVec(stateNum_sub));
        } else if ( abs(state_vars.stateMat(i, stateNum_sub) - state_vars.lowerLims(stateNum_sub)) < state_vars.dVec(stateNum_sub)/2 )  {
            //at the lower boundary, use the point above it
            i_star = i_star + int(state_vars.increVec(stateNum_sub));
        } else {
            i_star = i_star;
        }

        dfdx(i) = (  f(i_star + x_step + y_step )
                   - f(i_star + x_step - y_step )
                   - f(i_star - x_step + y_step )
                   + f(i_star - x_step - y_step )  ) / (4.0 * state_vars.dVec(stateNum) * state_vars.dVec(stateNum_sub) );
    }

}


derivs::derivs (stateVars & state_vars, Parameters & parameters) {
    useLogW = parameters.useLogW;
    varToDiff.resize(state_vars.S, 1);
    this->constructLinearOps(state_vars);
    k = choose(state_vars.N, 2) - 1;

    firstPartialsMap["w"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    firstPartialsMap["Z"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    firstPartialsMap["V"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    firstPartialsMap["H"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();

    secondPartialsMap["w"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    secondPartialsMap["Z"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    secondPartialsMap["V"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
    secondPartialsMap["H"]  = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();


    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            crossPartialsMap[k] = Eigen::MatrixXd::Constant(state_vars.S, 1, 0.0).array();
            k = k - 1;

        }
    }

        int ch = choose(state_vars.N, 2);
}

derivs::derivs(){
}

void  derivs::computeDerivsOld (Eigen::Ref<Eigen::ArrayXd>  f, stateVars & state_vars) {

    //  Numerically compute derivatives via finite diff approx.
    //    - First partials:  use central diff when not at the boundary and use forward/backward diff when at the boundary
    //    - Second partials: apply the first partial technique on the first partials again when not at the boundary;
    //                       when at the boundary and the point next to the boundary,
    //                       use the point next to the boundary and apply the standard formula.
    //    - Cross partials:  for now apply the standard formula.
    //  Process: iterate over n (iterant on state variables) and then i (iterant on grid points).
    //           For each i, check whether it's at the boundary and compute partials accordingly.

    k = choose(state_vars.N, 2) - 1;

    for (int n = (state_vars.N - 1); n >= 0; --n) {
        // point at the upper boundary
        //for(int i = 0; i < state_vars.upperBdryCt[n]; i++) {
        for(int i = 0; i < state_vars.upperBdryCt[n]; i++) {

            // First partial: backward diff
            firstPartialsMap[state_vars.num2State[n]](state_vars.upperBdryPts[n][i])  =
            (f(state_vars.upperBdryPts[n][i]) - f(state_vars.upperBdryPts[n][i] - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);

            // Second partial: apply standard formula with point next to boundary
            secondPartialsMap[state_vars.num2State[n]](state_vars.upperBdryPts[n][i]) =
            (f(state_vars.upperBdryPts[n][i]) - 2 * f(state_vars.upperBdryPts[n][i] - int(state_vars.increVec(n)) ) + f(state_vars.upperBdryPts[n][i] - 2 * int(state_vars.increVec(n)) ) ) / (pow(state_vars.dVec(n),2));

        }

        // point at the lower boundary
        for(int i = 0; i < state_vars.lowerBdryCt[n]; i++) {
            // First partial: forward diff
            firstPartialsMap[state_vars.num2State[n]](state_vars.lowerBdryPts[n][i])  =
            (f(state_vars.lowerBdryPts[n][i] + int(state_vars.increVec(n)) ) - f(state_vars.lowerBdryPts[n][i]) ) / state_vars.dVec(n);

            // Second partial: apply standard formula with point next to boundary
            secondPartialsMap[state_vars.num2State[n]](state_vars.lowerBdryPts[n][i]) =
            (f(state_vars.lowerBdryPts[n][i] + 2 * int(state_vars.increVec(n)) )  - 2 * f(state_vars.lowerBdryPts[n][i] + int(state_vars.increVec(n)) ) + f(state_vars.lowerBdryPts[n][i]) ) / (pow(state_vars.dVec(n),2));

        }

        // point next to the upper boundary
        for(int i = 0; i < state_vars.adjUpperBdryCt[n]; i++) {
            // First partial: central diff
            firstPartialsMap[state_vars.num2State[n]](state_vars.adjUpperBdryPts[n][i])  =
            (f(state_vars.adjUpperBdryPts[n][i] + int(state_vars.increVec(n))) - f(state_vars.adjUpperBdryPts[n][i] - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));

            // Second partial: apply standard formula at the correct point
            secondPartialsMap[state_vars.num2State[n]](state_vars.adjUpperBdryPts[n][i]) =
            (f(state_vars.adjUpperBdryPts[n][i] + int(state_vars.increVec(n))) - 2 * f(state_vars.adjUpperBdryPts[n][i]) + f(state_vars.adjUpperBdryPts[n][i] - int(state_vars.increVec(n))) ) / (pow(state_vars.dVec(n),2));

        }

        // point next to the upper boundary
        for(int i = 0; i < state_vars.adjLowerBdryCt[n]; i++) {
            // First partial: central diff
            firstPartialsMap[state_vars.num2State[n]](state_vars.adjLowerBdryPts[n][i])  =
            (f(state_vars.adjLowerBdryPts[n][i] + int(state_vars.increVec(n))) - f(state_vars.adjLowerBdryPts[n][i] - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));

            // Second partial: apply standard formula at the correct point
            secondPartialsMap[state_vars.num2State[n]](state_vars.adjLowerBdryPts[n][i]) =
            (f(state_vars.adjLowerBdryPts[n][i] + int(state_vars.increVec(n))) - 2 * f(state_vars.adjLowerBdryPts[n][i]) + f(state_vars.adjLowerBdryPts[n][i] - int(state_vars.increVec(n))) ) / (pow(state_vars.dVec(n),2));

        }

        // central points
        for(int i = 0; i < state_vars.centralCt[n]; i++) {
            // First partial: central diff
            firstPartialsMap[state_vars.num2State[n]](state_vars.centralPts[n][i])  =
            (f(state_vars.centralPts[n][i] + int(state_vars.increVec(n))) - f(state_vars.centralPts[n][i] - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));

            // Second partial: apply the first derivative operator twice
            secondPartialsMap[state_vars.num2State[n]](state_vars.centralPts[n][i]) =
            (f(state_vars.centralPts[n][i] + int( 2 * state_vars.increVec(n))) - 2 * f(state_vars.centralPts[n][i]) + f(state_vars.centralPts[n][i] - int(2 * state_vars.increVec(n))) ) / (pow(2.0 * state_vars.dVec(n),2));

        }


        /****************************************************************/
        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            computeCrossDer(state_vars, n, n_sub, f, crossPartialsMap[k] );
            k = k - 1;

        }
    }

    /****************************************************************/
    /* Change the derivative since we are changing from w to log(w) */
    if (useLogW) {
        firstPartialsMap[state_vars.num2State[0]] = ( secondPartialsMap[state_vars.num2State[0]].array() - firstPartialsMap[state_vars.num2State[0]].array()) * (-2.0 * state_vars.logW).exp();
        firstPartialsMap[state_vars.num2State[0]] = firstPartialsMap[state_vars.num2State[0]] * (-1.0 * state_vars.logW).exp();
    }



    /* Cross partials are stored as

     crossPartials.col(5): 32; crossPartials.col(4): 31; crossPartials.col(3): 30; crossPartials.col(2): 21; crossPartials.col(1): 20; crossPartials.col(0): 10

     */

}

void  derivs::computeDerivs (Eigen::ArrayXd f, stateVars & state_vars) {

    //  Numerically compute derivatives via finite diff approx.
    //    - First partials:  use central diff when not at the boundary and use forward/backward diff when at the boundary
    //    - Second partials: apply the first partial technique on the first partials again when not at the boundary;
    //                       when at the boundary and the point next to the boundary,
    //                       use the point next to the boundary and apply the standard formula.
    //    - Cross partials:  for now apply the standard formula.
    //  Process: iterate over n (iterant on state variables) and then i (iterant on grid points).
    //           For each i, check whether it's at the boundary and compute partials accordingly.
    //varToDiff = f;
    varToDiff.conservativeResize(state_vars.S, 1);
    k = choose(state_vars.N, 2) - 1;

    for (int n = (state_vars.N - 1); n >= 0; --n) {
        varToDiff = firstPartialsLinearOps[state_vars.num2State[n]]   * f.matrix();
        firstPartialsMap[state_vars.num2State[n]]  = (varToDiff).array().transpose();
        varToDiff = secondPartialsLinearOps[state_vars.num2State[n]]  * f.matrix();
        secondPartialsMap[state_vars.num2State[n]] = (varToDiff).array().transpose();

        /****************************************************************/
        for (int n_sub = n-1; n_sub >=0; --n_sub) {
            varToDiff = crossPartialsLinearOps[k] * f.matrix();
            crossPartialsMap[k] = (varToDiff).array().transpose();
            k = k - 1;

        }
    }

    /****************************************************************/
    /* Change the derivative since we are changing from w to log(w) */
    if (useLogW) {
        firstPartialsMap[state_vars.num2State[0]] = ( secondPartialsMap[state_vars.num2State[0]].array() - firstPartialsMap[state_vars.num2State[0]].array()) * (-2.0 * state_vars.logW).exp();
        firstPartialsMap[state_vars.num2State[0]] = firstPartialsMap[state_vars.num2State[0]] * (-1.0 * state_vars.logW).exp();
    }



    /* Cross partials are stored as

     crossPartials.col(5): 32; crossPartials.col(4): 31; crossPartials.col(3): 30; crossPartials.col(2): 21; crossPartials.col(1): 20; crossPartials.col(0): 10

     */

}

void  derivs::computeDerivsValueFn (Eigen::Ref<Eigen::ArrayXd>  f, stateVars & state_vars, Eigen::MatrixXd & firstCoefsE, Eigen::MatrixXd & firstCoefsH, string functionName) {

    // *** This function is deprecated; saved here for testing purposes *** //
    //  Numerically compute derivatives via finite diff approx.
    //    - First partials:  use central diff when not at the boundary and use forward/backward diff when at the boundary
    //    - Second partials: apply the first partial technique on the first partials again when not at the boundary;
    //                       when at the boundary and the point next to the boundary,
    //                       use the point next to the boundary and apply the standard formula.
    //    - Cross partials:  for now apply the standard formula.
    //  Process: iterate over n (iterant on state variables) and then i (iterant on grid points).
    //           For each i, check whether it's at the boundary and compute partials accordingly.

    k = choose(state_vars.N, 2) - 1;

    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int i = 0; i < state_vars.S; i++) {
            if ( abs(state_vars.stateMat(i, n) - state_vars.upperLims(n)) < state_vars.dVec(n)/2 ) {
                // point at the upper boundary

                // First partial: backward diff
                firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);

                // Second partial: apply standard formula with point next to boundary
                secondPartialsMap[state_vars.num2State[n]](i) = (f(i) - 2 * f(i - int(state_vars.increVec(n)) ) + f(i - 2 * int(state_vars.increVec(n)) ) ) / (pow(state_vars.dVec(n),2));

            } else if ( abs(state_vars.stateMat(i, n) - state_vars.lowerLims(n)) < state_vars.dVec(n)/2 ) {
                // point at the lower boundary

                // First partial: forward diff
                firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);

                // Second partial: apply standard formula with point next to boundary
                secondPartialsMap[state_vars.num2State[n]](i) = (f(i + 2 * int(state_vars.increVec(n)) )  - 2 * f(i + int(state_vars.increVec(n)) ) + f(i) ) / (pow(state_vars.dVec(n),2));

            } else if ( abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) * 1.5) {
                // point next to the upper boundary

                // First partial: central diff
                // firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n))) - f(i - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));

                // First partial: upwinding style
                if (functionName.compare("zetaE") == 0) {
                    if (firstCoefsE(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else if (functionName.compare("zetaH") == 0) {
                    if (firstCoefsH(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else {
                    firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n))) - f(i - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));
                }

                // Second partial: apply standard formula at the correct point
                secondPartialsMap[state_vars.num2State[n]](i) = (f(i + int(state_vars.increVec(n))) - 2 * f(i) + f(i - int(state_vars.increVec(n))) ) / (pow(state_vars.dVec(n),2));

            } else if ( abs(state_vars.stateMat(i, n) - state_vars.lowerLims(n)) < state_vars.dVec(n) * 1.5 ) {
                // point next to the upper boundary

                // First partial: central diff

                // First partial: upwinding style
                if (functionName.compare("zetaE") == 0) {
                    if (firstCoefsE(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else if (functionName.compare("zetaH") == 0) {
                    if (firstCoefsH(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else {
                    firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n))) - f(i - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));
                }


                // Second partial: apply standard formula at the correct point
                secondPartialsMap[state_vars.num2State[n]](i) = (f(i + int(state_vars.increVec(n))) - 2 * f(i) + f(i - int(state_vars.increVec(n))) ) / (pow(state_vars.dVec(n),2));

            } else {
                // point not at the boundary

                // First partial: central diff

                // First partial: upwinding style
                if (functionName.compare("zetaE") == 0) {
                    if (firstCoefsE(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else if (functionName.compare("zetaH") == 0) {
                    if (firstCoefsH(i,n) >= 0) {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n)) ) - f(i) ) / state_vars.dVec(n);
                    } else {
                        firstPartialsMap[state_vars.num2State[n]](i)  = (f(i) - f(i - int(state_vars.increVec(n)) ) ) / state_vars.dVec(n);
                    }

                } else {
                    firstPartialsMap[state_vars.num2State[n]](i)  = (f(i + int(state_vars.increVec(n))) - f(i - int(state_vars.increVec(n))) ) / (2 * state_vars.dVec(n));
                }

                // Second partial: apply the first derivative operator twice
                secondPartialsMap[state_vars.num2State[n]](i) = (f(i + int( 2 * state_vars.increVec(n))) - 2 * f(i) + f(i - int(2 * state_vars.increVec(n))) ) / (pow(2.0 * state_vars.dVec(n),2));
            }
        }

        /****************************************************************/
        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            computeCrossDer(state_vars, n, n_sub, f, crossPartialsMap[k] );
            k = k - 1;

        }
    }

    /****************************************************************/
    /* Change the derivative since we are changing from w to log(w) */
    if (useLogW) {
        firstPartialsMap[state_vars.num2State[0]] = ( secondPartialsMap[state_vars.num2State[0]].array() - firstPartialsMap[state_vars.num2State[0]].array()) * (-2.0 * state_vars.logW).exp();
        firstPartialsMap[state_vars.num2State[0]] = firstPartialsMap[state_vars.num2State[0]] * (-1.0 * state_vars.logW).exp();
    }



    /* Cross partials are stored as

     crossPartials.col(5): 32; crossPartials.col(4): 31; crossPartials.col(3): 30; crossPartials.col(2): 21; crossPartials.col(1): 20; crossPartials.col(0): 10

     */

}

void derivs::constructLinearOps (stateVars & state_vars) {

  // This function constructs the linear operators needed for differentiation
  k = choose(state_vars.N, 2) - 1;

  for (int n = (state_vars.N - 1); n >=0; --n ) {

        firstPartialList.clear(); secondPartialList.clear(); crossPartialList.clear();

        for(int i = 0; i < state_vars.upperBdryCt[n]; i++) {

          // First derivatives
          firstPartialList.push_back(T(state_vars.upperBdryPts[n][i], state_vars.upperBdryPts[n][i], 1.0 / state_vars.dVec(n) ));
          firstPartialList.push_back(T(state_vars.upperBdryPts[n][i], state_vars.upperBdryPts[n][i] - state_vars.increVec(n), -1.0 / state_vars.dVec(n) ));

          // Second derivatives
          secondPartialList.push_back(T(state_vars.upperBdryPts[n][i], state_vars.upperBdryPts[n][i], 1.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.upperBdryPts[n][i], state_vars.upperBdryPts[n][i] - state_vars.increVec(n), -2.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.upperBdryPts[n][i], state_vars.upperBdryPts[n][i] - 2 * state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));

        }

        // point at the lower boundary
        for(int i = 0; i < state_vars.lowerBdryCt[n]; i++) {

          // First derivatives
          firstPartialList.push_back(T(state_vars.lowerBdryPts[n][i], state_vars.lowerBdryPts[n][i], -1.0 / state_vars.dVec(n) ));
          firstPartialList.push_back(T(state_vars.lowerBdryPts[n][i], state_vars.lowerBdryPts[n][i] + state_vars.increVec(n), 1.0 / state_vars.dVec(n) ));

          // Second derivatives
          secondPartialList.push_back(T(state_vars.lowerBdryPts[n][i], state_vars.lowerBdryPts[n][i], 1.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.lowerBdryPts[n][i], state_vars.lowerBdryPts[n][i] + state_vars.increVec(n), -2.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.lowerBdryPts[n][i], state_vars.lowerBdryPts[n][i] + 2 * state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));

        }

        // point next to the upper boundary
        for(int i = 0; i < state_vars.adjUpperBdryCt[n]; i++) {
            // First partial: central diff
            firstPartialList.push_back(T(state_vars.adjUpperBdryPts[n][i], state_vars.adjUpperBdryPts[n][i] + state_vars.increVec(n), 0.5 / state_vars.dVec(n) ));
            firstPartialList.push_back(T(state_vars.adjUpperBdryPts[n][i], state_vars.adjUpperBdryPts[n][i] - state_vars.increVec(n), -0.5 / state_vars.dVec(n) ));

            // Second derivatives
            secondPartialList.push_back(T(state_vars.adjUpperBdryPts[n][i], state_vars.adjUpperBdryPts[n][i], - 2.0 / pow(state_vars.dVec(n),2) ));
            secondPartialList.push_back(T(state_vars.adjUpperBdryPts[n][i], state_vars.adjUpperBdryPts[n][i] - state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));
            secondPartialList.push_back(T(state_vars.adjUpperBdryPts[n][i], state_vars.adjUpperBdryPts[n][i] + state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));

        }

        // point next to the upper boundary
        for(int i = 0; i < state_vars.adjLowerBdryCt[n]; i++) {
          // First partial: central diff
          firstPartialList.push_back(T(state_vars.adjLowerBdryPts[n][i], state_vars.adjLowerBdryPts[n][i] + state_vars.increVec(n), 0.5 / state_vars.dVec(n) ));
          firstPartialList.push_back(T(state_vars.adjLowerBdryPts[n][i], state_vars.adjLowerBdryPts[n][i] - state_vars.increVec(n), -0.5 / state_vars.dVec(n) ));

          // Second derivatives
          secondPartialList.push_back(T(state_vars.adjLowerBdryPts[n][i], state_vars.adjLowerBdryPts[n][i], - 2.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.adjLowerBdryPts[n][i], state_vars.adjLowerBdryPts[n][i] - state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));
          secondPartialList.push_back(T(state_vars.adjLowerBdryPts[n][i], state_vars.adjLowerBdryPts[n][i] + state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));

        }

        // central points
        for(int i = 0; i < state_vars.centralCt[n]; i++) {
            // First partial: central diff
            firstPartialList.push_back(T(state_vars.centralPts[n][i], state_vars.centralPts[n][i] + state_vars.increVec(n), 0.5 / state_vars.dVec(n) ));
            firstPartialList.push_back(T(state_vars.centralPts[n][i], state_vars.centralPts[n][i] - state_vars.increVec(n), -0.5 / state_vars.dVec(n) ));

            // Second derivatives
            secondPartialList.push_back(T(state_vars.centralPts[n][i], state_vars.centralPts[n][i], - 2.0 / pow(state_vars.dVec(n),2) ));
            secondPartialList.push_back(T(state_vars.centralPts[n][i], state_vars.centralPts[n][i] - 2 * state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));
            secondPartialList.push_back(T(state_vars.centralPts[n][i], state_vars.centralPts[n][i] + 2 * state_vars.increVec(n), 1.0 / pow(state_vars.dVec(n),2) ));

        }
        firstPartialsLinearOps[state_vars.num2State[n]].resize(state_vars.S,state_vars.S);
        secondPartialsLinearOps[state_vars.num2State[n]].resize(state_vars.S,state_vars.S);

        firstPartialsLinearOps[state_vars.num2State[n]].setFromTriplets(firstPartialList.begin(), firstPartialList.end());
        secondPartialsLinearOps[state_vars.num2State[n]].setFromTriplets(secondPartialList.begin(), secondPartialList.end());
        firstPartialsLinearOps[state_vars.num2State[n]].makeCompressed();
        secondPartialsLinearOps[state_vars.num2State[n]].makeCompressed();

        /****************************************************************/
        for (int n_sub = n-1; n_sub >=0; --n_sub) {

          // If at the boundary, use the point next to it
          int x_step = int(state_vars.increVec(n));
          int y_step = int(state_vars.increVec(n_sub));
          int x_star = 0; int y_star = 0;
          int i_star = 0;

          for (int i = 0; i < state_vars.S; i++) {

              //Check location of the first variable
              if ( abs(state_vars.stateMat(i, n) - state_vars.upperLims(n)) < state_vars.dVec(n)/2 ) {
                  //at the upper boundary, use the point below it
                  i_star = i - int(state_vars.increVec(n));
              } else if ( abs(state_vars.stateMat(i, n) - state_vars.lowerLims(n)) < state_vars.dVec(n)/2 )  {
                  //at the lower boundary, use the point above it
                  i_star = i + int(state_vars.increVec(n));
              } else {
                  i_star = i;
              }

              //Check location of the second variable
              if ( abs(state_vars.stateMat(i, n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub)/2 ) {
                  //at the upper boundary, use the point below it
                  i_star = i_star - int(state_vars.increVec(n_sub));
              } else if ( abs(state_vars.stateMat(i, n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub)/2 )  {
                  //at the lower boundary, use the point above it
                  i_star = i_star + int(state_vars.increVec(n_sub));
              } else {
                  i_star = i_star;
              }

              crossPartialList.push_back(T(i, i_star + x_step + y_step, (4.0  * state_vars.dVec(n) * state_vars.dVec(n_sub) ) ));
              crossPartialList.push_back(T(i, i_star + x_step - y_step, (-4.0 * state_vars.dVec(n) * state_vars.dVec(n_sub) ) ));
              crossPartialList.push_back(T(i, i_star - x_step + y_step, (-4.0 * state_vars.dVec(n) * state_vars.dVec(n_sub) ) ));
              crossPartialList.push_back(T(i, i_star - x_step - y_step, (4.0  * state_vars.dVec(n) * state_vars.dVec(n_sub) ) ));

          }

          crossPartialsLinearOps[k].resize(state_vars.S, state_vars.S);
          crossPartialsLinearOps[k].setFromTriplets(crossPartialList.begin(), crossPartialList.end());
          crossPartialsLinearOps[k].makeCompressed();
          k = k - 1;

        }
  }
}
