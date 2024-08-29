//
//  matrixVars.cpp
//
//
//  Created by Joseph Huang on 8/6/18.
//
//

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "matrixVars.h"
#include "common.h"


matrixVars::matrixVars() {

}
matrixVars::matrixVars(stateVars & state_vars, Parameters & parameters) {

    /* Initialize the arrays needed */

    Fe.resize(state_vars.S); firstCoefsE.resize(state_vars.S, state_vars.N); secondCoefsE.resize(state_vars.S, state_vars.N);
    Fh.resize(state_vars.S); firstCoefsH.resize(state_vars.S, state_vars.N); secondCoefsH.resize(state_vars.S, state_vars.N);

    sigmaX_temp.resize(4,4); derivs_temp.resize(4, 1); idenMat.resize(4, 4); tempResult.resize(1,1);

    Le.resize(state_vars.S,state_vars.S); Lh.resize(state_vars.S,state_vars.S);
    LeNoTransp.resize(state_vars.S,state_vars.S); LhNoTransp.resize(state_vars.S,state_vars.S);

    eList.reserve(7 * state_vars.S); hList.reserve(7 * state_vars.S);
    Ue.resize(state_vars.S); Uh.resize(state_vars.S);

    I.resize(state_vars.S,state_vars.S); I.setIdentity();

    rowNorms.resize(state_vars.S);
    //find indices

}

void matrixVars::updateMatrixVars(stateVars & state_vars, valueVars & value_vars, Vars & vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, Parameters & parameters) {

    ///update terms for households pde

    if (parameters.rho_h == 1.0) {
        Fh = (-value_vars.xi_h + log(parameters.delta_h)) * parameters.delta_h - parameters.delta_h;
    } else if (parameters.rho_h != 1.0) {
        Fh = ( parameters.rho_h / (1-parameters.rho_h) * pow(parameters.delta_h, 1/parameters.rho_h) * value_vars.xi_h.exp().pow((1- 1/parameters.rho_h)) - parameters.delta_h / (1 - parameters.rho_h) ) ;
    }

    Fh = Fh.array() + vars.r + (  vars.Pi.rowwise().norm().array().square() + (parameters.gamma_h * vars.beta_h * state_vars.H.sqrt() ).square() ) / (2*parameters.gamma_h);
    for (int s = 0; s < parameters.nShocks; ++s ) {

        // This takes care of the squared term (1-parameters.gamma_h)/parameters.gamma_h * the squared norm.
        Fh = Fh.array() + 0.5 * (1.0 - parameters.gamma_h) / parameters.gamma_h * ( vars.sigmaXMap["w"].col(s).array() * derivsXiH.firstPartialsMap["w"].array() + vars.sigmaXMap["Z"].col(s).array() * derivsXiH.firstPartialsMap["Z"].array() + vars.sigmaXMap["V"].col(s).array() * derivsXiH.firstPartialsMap["V"].array() + vars.sigmaXMap["H"].col(s).array() * derivsXiH.firstPartialsMap["H"].array() ).square();
    }


    // This takes care of the cross partials.
    k = choose(state_vars.N, 2) - 1;
    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            Fh = Fh.array() + derivsXiH.crossPartialsMap[k] * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            k = k - 1;
        }

    }


    for (int n = 0; n < state_vars.N; ++n ) {

        if (n == 0) {
            firstCoefsH.col(n) = vars.muXMap[state_vars.num2State[n]] + (1 - parameters.gamma_h) / parameters.gamma_h * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.Pi).rowwise().sum().array() );

            if (parameters.useLogW) {
                firstCoefsH.col(n) = firstCoefsH.col(n).array() * ( (-1.0 * state_vars.logW).exp()  ).array();
            }

            secondCoefsH.col(n) = 0.5 *  ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n]]).rowwise().sum().array()  ).array() ;

            if (parameters.useLogW) {
                secondCoefsH.col(n) = secondCoefsH.col(n).array() * ( (-2.0 * state_vars.logW).exp() ).array()  ;
                firstCoefsH.col(n)  = firstCoefsH.col(n).array() - secondCoefsH.col(n).array();
            }

        } else {
            firstCoefsH.col(n) = vars.muXMap[state_vars.num2State[n]] + (1 - parameters.gamma_h) / parameters.gamma_h * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.Pi).rowwise().sum().array() );
            secondCoefsH.col(n) = 0.5 *  ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n]]).rowwise().sum().array()  ).array()  ;
        }

    }


    ///update terms for experts pde

    if (parameters.rho_e == 1.0) {
        Fe = (-value_vars.xi_e + log(parameters.delta_e)) * parameters.delta_e - parameters.delta_e;
    } else if (parameters.rho_e != 1.0)  {
        Fe = parameters.rho_e / (1 - parameters.rho_e) * pow(parameters.delta_e, 1/parameters.rho_e) * ( value_vars.xi_e).exp().pow((1- 1/parameters.rho_e)) - parameters.delta_e / (1 - parameters.rho_e);
    }

    Fe = Fe.array() + vars.r  + (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() ).square() / (2 * parameters.gamma_e * (vars.normR2 + state_vars.H));


    // This takes care of the cross partials.
    k = choose(state_vars.N, 2) - 1;
    for (int n = (state_vars.N - 1); n >= 0; --n) {

        for (int n_sub = n-1; n_sub >=0; --n_sub) {

            Fe = Fe.array() + derivsXiE.crossPartialsMap[k] * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n_sub]]).rowwise().sum().array() );
            k = k - 1;
        }
    }

    //take care of the quadratic term for Fe

    for (int s = 0; s < parameters.nShocks; s++) {
        for (int s_sub = 0; s_sub < parameters.nShocks; s_sub++) {
            Fe = Fe.array() + ( vars.sigmaXMap["w"].col(s).array() * derivsXiE.firstPartialsMap["w"] + vars.sigmaXMap["Z"].col(s).array() * derivsXiE.firstPartialsMap["Z"] + vars.sigmaXMap["V"].col(s).array() * derivsXiE.firstPartialsMap["V"] + vars.sigmaXMap["H"].col(s).array() * derivsXiE.firstPartialsMap["H"] ) * (vars.sigmaR.col(s).array() * vars.sigmaR.col(s_sub).array() * (1.0 - parameters.gamma_e) / (vars.normR2 + state_vars.H) + (parameters.gamma_e) * (s == s_sub)) * ( vars.sigmaXMap["w"].col(s_sub).array() * derivsXiE.firstPartialsMap["w"] + vars.sigmaXMap["Z"].col(s_sub).array() * derivsXiE.firstPartialsMap["Z"] + vars.sigmaXMap["V"].col(s_sub).array() * derivsXiE.firstPartialsMap["V"] + vars.sigmaXMap["H"].col(s_sub).array() * derivsXiE.firstPartialsMap["H"]) * (1.0 - parameters.gamma_e) / parameters.gamma_e * 0.5;


        }

    }


    for (int n = 0; n < state_vars.N; ++n ) {
        if (n == 0) {
            firstCoefsE.col(n) = vars.muXMap[state_vars.num2State[n]] + (1 - parameters.gamma_e) / parameters.gamma_e * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaR).rowwise().sum().array() ) * (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array()   ) / (vars.normR2 + state_vars.H);

            if (parameters.useLogW) {
                firstCoefsE.col(n) = firstCoefsE.col(n).array() * ( (-1.0 * state_vars.logW).exp()  ).array();
            }

            secondCoefsE.col(n) = 0.5 *  ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n]]).rowwise().sum().array()  );

            if (parameters.useLogW) {
                secondCoefsE.col(n) = secondCoefsE.col(n).array() * ( (-2.0 * state_vars.logW).exp() ).array()  ;
                firstCoefsE.col(n) = firstCoefsE.col(n).array() - secondCoefsE.col(n).array();

            }
        } else {
            firstCoefsE.col(n) = vars.muXMap[state_vars.num2State[n]] + (1 - parameters.gamma_e) / parameters.gamma_e * ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaR).rowwise().sum().array() ) * (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array()  ) / (vars.normR2 + state_vars.H);
            secondCoefsE.col(n) = 0.5 *  ( vars.sigmaXMap[state_vars.num2State[n]].cwiseProduct(vars.sigmaXMap[state_vars.num2State[n]]).rowwise().sum().array()  ) ;
        }
    }

}

void matrixVars::updateMatrix(stateVars & state_vars, Parameters & parameters) {
    atBoundIndicators.resize(state_vars.N);
    eList.clear(); hList.clear();

    firstCoefE = 0.0;
    secondCoefE = 0.0;
    firstCoefH = 0.0;
    secondCoefH = 0.0;
    //construct matrix
    for (int i = 0; i < state_vars.S; ++i) {



        eList.push_back(T(i,i, 1.0 ));
        hList.push_back(T(i,i, 1.0 ));

        //check boundaries

        for (int n = (state_vars.N - 1); n >=0; --n ) {
            atBoundIndicators(n) = -1.0;
            firstCoefE = firstCoefsE(i,n);
            secondCoefE = secondCoefsE(i,n);
            firstCoefH = firstCoefsH(i,n);
            secondCoefH = secondCoefsH(i,n);

            //check whether it's at upper or lower boundary
            if ( abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n)/2 ) {  //upper boundary

                atBoundIndicators(n) = 1.0;
                eList.push_back(T(i, i, - parameters.dt * ( firstCoefE/state_vars.dVec(n) + secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                eList.push_back(T(i, i - state_vars.increVec(n), - parameters.dt * ( - firstCoefE/state_vars.dVec(n) - 2 * secondCoefE / pow(state_vars.dVec(n), 2) ) ));
                eList.push_back(T(i, i - 2*state_vars.increVec(n), - parameters.dt * ( secondCoefE / pow(state_vars.dVec(n), 2) ) ) );

                hList.push_back(T(i, i, - parameters.dt * ( firstCoefH/state_vars.dVec(n) + secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                hList.push_back(T(i, i - state_vars.increVec(n), - parameters.dt * ( - firstCoefH/state_vars.dVec(n) - 2 * secondCoefH / pow(state_vars.dVec(n), 2) ) ));
                hList.push_back(T(i, i - 2*state_vars.increVec(n), - parameters.dt * ( secondCoefH / pow(state_vars.dVec(n), 2) ) ) );

            } else if ( abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n)/2 ) { //lower boundary

                atBoundIndicators(n) = 1.0;
                eList.push_back(T(i, i, - parameters.dt * ( - firstCoefE/state_vars.dVec(n) + secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                eList.push_back(T(i, i + state_vars.increVec(n), - parameters.dt * ( firstCoefE/state_vars.dVec(n) - 2 * secondCoefE / pow(state_vars.dVec(n), 2) ) ));
                eList.push_back(T(i, i + 2*state_vars.increVec(n), - parameters.dt * ( secondCoefE / pow(state_vars.dVec(n), 2) ) ) );

                hList.push_back(T(i, i, - parameters.dt * ( - firstCoefH/state_vars.dVec(n) + secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                hList.push_back(T(i, i + state_vars.increVec(n), - parameters.dt * ( firstCoefH/state_vars.dVec(n) - 2 * secondCoefH / pow(state_vars.dVec(n), 2) ) ));
                hList.push_back(T(i, i + 2*state_vars.increVec(n), - parameters.dt * ( secondCoefH / pow(state_vars.dVec(n), 2) ) ) );


            }



        }


        for (int n = (state_vars.N - 1); n >= 0; --n) {
            //level and time deriv


            //add elements to the vector of triplets for matrix construction
            if (atBoundIndicators(n) < 0) {

                firstCoefE = firstCoefsE(i,n);
                secondCoefE = secondCoefsE(i,n);
                firstCoefH = firstCoefsH(i,n);
                secondCoefH = secondCoefsH(i,n);

                //first derivative


                if (firstCoefE != 0.0) {

                    eList.push_back(T(i,i, - parameters.dt * ( -firstCoefE * ( firstCoefE > 0) + firstCoefE * ( firstCoefE < 0) ) / state_vars.dVec(n)  ) );
                    eList.push_back(T(i,i + state_vars.increVec(n), - parameters.dt * firstCoefE * ( firstCoefE > 0) / state_vars.dVec(n) ));
                    eList.push_back(T(i,i - state_vars.increVec(n), - parameters.dt *  - firstCoefE * ( firstCoefE < 0) / state_vars.dVec(n) ));
                }

                if (firstCoefH != 0.0) {
                    hList.push_back(T(i,i, - parameters.dt * ( -firstCoefH * ( firstCoefH > 0) + firstCoefH * ( firstCoefH < 0) ) / state_vars.dVec(n)  ) );
                    hList.push_back(T(i,i + state_vars.increVec(n), - parameters.dt * firstCoefH * ( firstCoefH > 0) / state_vars.dVec(n) ));
                    hList.push_back(T(i,i - state_vars.increVec(n), - parameters.dt *  - firstCoefH * ( firstCoefH < 0) / state_vars.dVec(n) ));

                }

                //second derivative

                if (secondCoefE != 0.0) {
                    eList.push_back(T(i, i, - parameters.dt * -2 * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                    eList.push_back(T(i, i + state_vars.increVec(n), - parameters.dt * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                    eList.push_back(T(i, i - state_vars.increVec(n), - parameters.dt * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                }

                if (secondCoefH != 0.0) {
                    hList.push_back(T(i, i, - parameters.dt * -2 * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                    hList.push_back(T(i, i + state_vars.increVec(n), - parameters.dt * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                    hList.push_back(T(i, i - state_vars.increVec(n), - parameters.dt * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                }


            }

        }


    }

    //form matrices
    Le.setFromTriplets(eList.begin(), eList.end());
    Lh.setFromTriplets(hList.begin(), hList.end());

    if ( parameters.method.compare("1") == 0 )  {
        Le = -1.0 * (Le - I) + I; Lh = -1.0 * (Lh - I) + I;
    }

    //compress
    Le.makeCompressed(); Lh.makeCompressed();

    /* Reserved for PARDISO; not active right now
    if ( parameters.method.compare("1") == 0 )  {

        //fill a, ia, ja
        jae.clear(); jah.clear(); ae.clear(); ah.clear(); iae.clear(); iah.clear();


        int* outerPtr_e = Le.outerIndexPtr(); int* outerPtr_h = Lh.outerIndexPtr();
        for (int i = 0; i < state_vars.S + 1; ++i) {
            iae.push_back(outerPtr_e[i] + 1);
        }
        for (int i = 0; i < state_vars.S + 1; ++i) {
            iah.push_back(outerPtr_h[i] + 1);
        }


        int* innerPtr_e = Le.innerIndexPtr(); int* innerPtr_h = Lh.innerIndexPtr();
        for (int i = 0; i < Le.nonZeros(); ++i) {
            jae.push_back(innerPtr_e[i] + 1);
        }
        for (int i = 0; i < Lh.nonZeros(); ++i) {
            jah.push_back(innerPtr_h[i] + 1);
        }

        double* valuePtr_e = Le.valuePtr(); double* valuePtr_h = Lh.valuePtr();
        for (int i = 0; i < Le.nonZeros(); ++i) {
            ae.push_back(valuePtr_e[i]);
        }
        for (int i = 0; i < Lh.nonZeros(); ++i) {
            ah.push_back(valuePtr_h[i]);
        }

    }
    */

}

void matrixVars::updateKnowns(valueVars & value_vars, stateVars & state_vars, Parameters & parameters) {

    if ( parameters.method.compare("1") == 0 ) {
        Ue = value_vars.xi_e; Uh = value_vars.xi_h;
    } else if ( (parameters.method.compare("2") == 0) ) {
        Ue = Fe.array() * parameters.dt + value_vars.xi_e; Uh = Fh.array() * parameters.dt + value_vars.xi_h;
    }

}

void matrixVars::solveWithCGICholE(valueVars & value_vars, stateVars & state_vars,
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> & ichol, double tol, int maxIters) {


    // This function solves the linear system with CG using incomplete cholesky as preconditioner


    // Initialization

    x = value_vars.xi_e_old.matrix();
    residual =  (Ue - Le * x);
    residual =  Le.transpose() * residual;
    p = ichol.solve(residual);

    absNew = Eigen::numext::real(residual.dot(p));  // the square of the absolute value of r scaled by invM
    Eigen::Index i = 0;
    rhsNorm2 = (Le.transpose() * Ue).squaredNorm();
    threshold = tol * tol * absNew;
    residualNorm2 = residual.squaredNorm();

    while(i < maxIters)
    {
      tmp.noalias() = (Le * p);                    // the bottleneck of the algorithm
      tmp           = Le.transpose() * tmp;
      alpha = absNew / p.dot(tmp);         // the amount we travel on dir

      x += alpha * p;                             // update solution


      if (i % 50 == 0) {
          residual =  (Ue - Le * x);
          residual =  Le.transpose() * residual;      //residual = Le.transpose() * residual;
      } else {
          residual = residual - alpha * tmp;
      }

      z = ichol.solve(residual);                // approximately solve for "A z = residual"

      absOld = absNew;
      absNew = Eigen::numext::real(residual.dot(z));     // update the absolute value of r
      beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      if( sqrt(Eigen::numext::real(residual.dot(residual))) < tol) {
          break;
      }
      i++;
    }
    residualNorm2 = residual.squaredNorm();
    cgEIters = i + 1;
    cgErrorE = sqrt(residualNorm2 / rhsNorm2);

    value_vars.xi_e = x;


}


void matrixVars::solveWithCGICholH(valueVars & value_vars, stateVars & state_vars,
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> & ichol, double tol, int maxIters) {

        // This function solves the linear system with CG using incomplete cholesky as preconditioner

        // Initialization
        x = value_vars.xi_h_old.matrix();
        residual = (Uh - Lh * x);
        residual = Lh.transpose()  * residual ;
        p = ichol.solve(residual);
        absNew = Eigen::numext::real(residual.dot(p));  // the square of the absolute value of r scaled by invM
        Eigen::Index i = 0;
        rhsNorm2 = Uh.squaredNorm();
        threshold = tol * tol * absNew;
        residualNorm2 = residual.squaredNorm();

        while(i < maxIters)
        {

          tmp.noalias() =  (Lh * p);                    // the bottleneck of the algorithm
          tmp           = Lh.transpose() * tmp;

          alpha = absNew / p.dot(tmp);         // the amount we travel on dir
          x += alpha * p;                             // update solution

          if (i % 50 == 0) {
              residual =  (Uh - Lh * x);
              residual =  Lh.transpose() * residual;      //residual = Le.transpose() * residual;
          } else {
              residual = residual - alpha * tmp;
          }
          z = ichol.solve(residual);                // approximately solve for "A z = residual"

          absOld = absNew;
          absNew = Eigen::numext::real(residual.dot(z));     // update the absolute value of r
          beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
          p = z + beta * p;                           // update search direction
          if( sqrt(Eigen::numext::real(residual.dot(residual))) < tol) {
              break;
          }
          i++;
        }

        residualNorm2 = residual.squaredNorm();
        cgHIters = i + 1;
        cgErrorH = sqrt(residualNorm2 / rhsNorm2);

        value_vars.xi_h = x;
}


void matrixVars::solveWithKacz(valueVars & value_vars, stateVars & state_vars,
    double tol, int maxIters) {

    int     r;     // Placeholder for row index
    double  z;     // The scalar to which the fraction evaluates
    int i = 0;     // Iterator of the Kaczmarc
    double  error; // error
    x = value_vars.xi_e_old.matrix(); // initial guess

    // This method attempts to solve the linear system through the Kaczmarc method
    while(i < maxIters)
    {
        error = 0.0;
        // Randomly select a row
        r = rand() % state_vars.S;

        // Compute the fraction
        z = (Ue(r) - Le.row(r).dot(x) ) / Le.row(r).squaredNorm();

        // Update the soluiton vector
        for (SpMat::InnerIterator it(Le,r); it; ++it)
        {
            x(it.index()) = value_vars.xi_e(it.index()) + z * it.value();
            error = error + abs(z * it.value());
        }

        if ( error < tol) {
            break;
        }
        i++;
    }

}
