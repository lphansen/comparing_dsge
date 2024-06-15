//
//  modules.h
//  metaModel
//
//  Created by Joseph Huang on 10/9/17.
//  Copyright (C) 2017, Joseph Huang.
//

#ifndef modules_h
#define modules_h
#define nDims 4
#define nOmega 200
#define nG 40
#define nS 40
#define nVarSig 10

//used to identify MPI's master process
#define MASTER 0

#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "diagnostics.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseQR>
typedef Eigen::SparseMatrix<double, Eigen::RowMajor > SpMat;
typedef Eigen::Triplet<double> T;
#include <typeinfo>

#ifdef USE_MPI
#include <mpi.h>
#endif

//Model parameters
//double lambda_g = 2/exp(2.5);
//double lambda_s = 2/exp(4.5);

double nu_newborn = 0.01;
double lambda_d = 0.02;

double lambda_g = 0.252;
double lambda_s = 0.156;
double lambda_varsigma = 1.38;
double varsigma_bar = 0.25;
double g_bar = 0.0;
double s_bar  = 1.0;
double rho_e = 5.0/100;
double rho_h = 5.0/100;
double a_e = 0.14;
double a_h =  -9999999999999999;
double psi_e = 1.5;
double psi_h = 1.5;
double phi = 3;
double gamma_e = 2.0;
double gamma_h = 2.0;
double sigma_A_norm = 0.027;
double sigma_g_norm = 0.0141;
double sigma_s_norm = 0.132;
double sigma_varsigma_norm = 0.17;
double nu = 0.0;
double chiUnderline = 0.25;


double sigma_vI = 0.0;
double chi = 1;
double deltaD = 1;
double delta = 6.0/100;

//End of model parameters

//Iteration parameters
string run = "example3DequityIss2hhCap0_psi1.0_gamma22_chiUnderline0.25_dt0.05";
string folderName = "results/" + run +"/";

double dt = 0.05;
int maxIters = 10000;
double tol = 0.00001;
double eps = 0.02;
//End of iteration parameters

//Parameters for state variables
int numSds = 5;

//min/max for omega
double omegaMin = 0.01; double omegaMax = 1 - omegaMin;

//min/max for s
double shape = 2 * lambda_s * s_bar / (pow(sigma_s_norm,2));
double rate = 2 * lambda_s / (pow(sigma_s_norm,2));
double sMin = 0.00001;
double sMax = s_bar + numSds * sqrt( shape / pow(rate, 2));

//min/max for g
double gVar = pow(s_bar * sigma_g_norm, 2) / (2 * lambda_g);
double gMin = g_bar - numSds * sqrt( gVar );
double gMax = g_bar + numSds * sqrt( gVar );

//min/max for varsigma
double shape_varsig = 2 * lambda_varsigma * varsigma_bar / (pow(sigma_varsigma_norm,2));
double rate_varsig = 2 * lambda_varsigma / (pow(sigma_varsigma_norm,2));
double varSigMin = 0.00001;
double varSigMax = varsigma_bar + numSds * sqrt(shape_varsig / pow(rate_varsig,2));



//double omega[state_vars.S];
//double g[state_vars.S];
//double s[state_vars.S];

//End of Parameters for state variables

//Coefficients for unitary EIS



//

//* Pardiso protoype and parameters *//

extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                             double *, int    *,    int *, int *,   int *, int *,
                             int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);

// End of Pardiso parameters

//Classes to store variables
class stateVars {
    
public:
    Eigen::MatrixXd stateMat; //matrix to store state variables
    Eigen::ArrayXd increVec; //vector to record steps
    Eigen::ArrayXd dVec; //vector to record steps
    int N; // num of dimensions
    int S; // number of rows for the grid
    Eigen::ArrayXd upperLims;
    Eigen::ArrayXd lowerLims;

    stateVars (Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd); //constructors with arrays of upper/lower bounds and gridsizes
    
    Eigen::ArrayXd omega; Eigen::ArrayXd g; Eigen::ArrayXd s; Eigen::ArrayXd varSig;
    
    Eigen::MatrixXd covMat;
    Eigen::VectorXd sigma_A;
    Eigen::VectorXd sigma_g;
    Eigen::VectorXd sigma_s;
    Eigen::VectorXd sigma_varsigma;

    //variable used in the MPI version to control printouts
    int my_rank;
    
    
    
};


stateVars::stateVars (Eigen::ArrayXd upper, Eigen::ArrayXd lower, Eigen::ArrayXd gridSizes) {
    
    upperLims = upper;
    lowerLims = lower;
    N = upper.size();
    S = gridSizes.prod();
    stateMat.resize(S,N);
    dVec.resize(N);
    increVec.resize(N);
    increVec(0) = 1;
    omega.resize(S); g.resize(S); s.resize(S); varSig.resize(S);

    #ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    #else
        my_rank = 0;
    #endif
    
    //fill in the state object; similar to the ndgrid function in MATLAB
    
    for (int n = 0; n < N; ++n) {
        
        if (n != 0) {
            increVec(n) = gridSizes(n - 1) * increVec(n - 1);
        }
        dVec(n) = (upper(n) - lower(n)) / (gridSizes(n) - 1);
        
        for (int i = 0; i < S; ++i) {
            stateMat(i,n) = lower(n) + dVec(n) * ( int(i /  increVec(n) ) % int( gridSizes(n) ) );
        }
        
    }
    if (N == 1) {
        omega = stateMat.col(0); g = Eigen::MatrixXd::Constant(S, 1, g_bar); s = Eigen::MatrixXd::Constant(S, 1, s_bar); varSig = Eigen::MatrixXd::Constant(S, 1, 0.0);
    } else if (N == 2) {
        omega = stateMat.col(0); g = stateMat.col(1); s = Eigen::MatrixXd::Constant(S, 1, s_bar); varSig = Eigen::MatrixXd::Constant(S, 1, 0.0);
    } else if (N == 3) {
        omega = stateMat.col(0); g = stateMat.col(1); s = stateMat.col(2); varSig = Eigen::MatrixXd::Constant(S, 1, 0.0);
    } else if (N == 4) {
        omega = stateMat.col(0); g = stateMat.col(1); s = stateMat.col(2); varSig = stateMat.col(3);
    }
    
    covMat.resize(N, N);
    sigma_A.resize(N); sigma_g.resize(N); sigma_s.resize(N); sigma_varsigma.resize(N);
    
    if (N == 4) {
        covMat << 1,0,0,0,0,1,-0.5, 0,0, 0, sqrt (1 - pow(0.5, 2)), 0, 0,0,0,1 ;
        sigma_A << covMat(0,0) * sigma_A_norm, covMat(1,0) * sigma_A_norm, covMat(2,0) * sigma_A_norm, covMat(3,0) * sigma_A_norm;
        sigma_g << covMat(0,1) * sigma_g_norm, covMat(1,1) * sigma_g_norm, covMat(2,1) * sigma_g_norm, covMat(3,1) * sigma_g_norm;
        sigma_s << covMat(0,2) * sigma_s_norm, covMat(1,2) * sigma_s_norm, covMat(2,2) * sigma_s_norm, covMat(3,2) * sigma_s_norm;
        sigma_varsigma << covMat(0,3) * sigma_varsigma_norm, covMat(1,3) * sigma_varsigma_norm, covMat(2,3) * sigma_varsigma_norm, covMat(3,3) * sigma_varsigma_norm;
        
    } else if (N == 3) {
        covMat << 1,0,0,0,1, 0, 0, 0, 1;//,-0.5,0, 0, sqrt (1 - pow(0.5, 2));
        sigma_A << covMat(0,0) * sigma_A_norm, covMat(1,0) * sigma_A_norm, covMat(2,0) * sigma_A_norm;
        sigma_g << covMat(0,1) * sigma_g_norm, covMat(1,1) * sigma_g_norm, covMat(2,1) * sigma_g_norm;
        sigma_s << covMat(0,2) * sigma_s_norm, covMat(1,2) * sigma_s_norm, covMat(2,2) * sigma_s_norm;
        sigma_varsigma << covMat(0,2) * sigma_varsigma_norm, covMat(1,2) * sigma_varsigma_norm, covMat(2,2) * sigma_varsigma_norm;
    } else if (N == 2) {
        covMat << 1,0,0,1;
        sigma_A << covMat(0,0) * sigma_A_norm, covMat(1,0) * sigma_A_norm;
        sigma_g << covMat(0,1) * sigma_g_norm, covMat(1,1) * sigma_g_norm;
        sigma_s << 0 * sigma_s_norm, 0 * sigma_s_norm;
        sigma_varsigma << 0 * sigma_varsigma_norm, 0 * sigma_varsigma_norm;
    }
    
}


class valueVars {
    
public:
    Eigen::ArrayXd zeta_e; Eigen::ArrayXd zeta_h;
    Eigen::ArrayXd zeta_e_old; Eigen::ArrayXd zeta_h_old;
    Eigen::ArrayXd kappa; Eigen::ArrayXd chi;
    Eigen::ArrayXd kappa_old; Eigen::ArrayXd chi_old;
    Eigen::ArrayXd kappaStar; Eigen::ArrayXd deltaEStar;
    
    valueVars (stateVars &, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd);
    
};

valueVars::valueVars (stateVars & state_vars, Eigen::ArrayXd zetaEguess, Eigen::ArrayXd zetaHguess, Eigen::ArrayXd kappaGuess, Eigen::ArrayXd chiGuess) {
    
    zeta_e.resize(state_vars.S); zeta_h.resize(state_vars.S); zeta_e_old.resize(state_vars.S); zeta_h_old.resize(state_vars.S);
    
    zeta_e = zetaEguess;
    zeta_h = zetaHguess;
    zeta_e_old = zetaEguess;
    zeta_h_old = zetaHguess;
    kappa = kappaGuess;
    kappa_old = kappaGuess;
    chi = chiGuess;
    chi_old = chiGuess;

    
    
}



class derivs {
    
public:
    //first partials
    Eigen::MatrixXd firstPartials;
    
    //second partials
    Eigen::MatrixXd secondPartials;

    
    //cross partials
    Eigen::MatrixXd crossPartials;

    derivs (stateVars &);
    void computeDerivs(Eigen::Ref<Eigen::ArrayXd> f, stateVars & state_vars);

    int my_rank;
    
};


////Derivatives given stateVars
int computeDer(stateVars & state_vars, int stateNum, Eigen::Ref<Eigen::ArrayXd>  f, Eigen::Ref<Eigen::ArrayXd> dfdx) {
    for (int i = 0; i < state_vars.S; i++) {
        if ( abs(state_vars.stateMat(i, stateNum) - state_vars.upperLims(stateNum)) < state_vars.dVec(stateNum)/2 ) {
            dfdx(i) = (f(i) - f(i - int(state_vars.increVec(stateNum)) ) ) / state_vars.dVec(stateNum);
        } else if ( abs(state_vars.stateMat(i, stateNum) - state_vars.lowerLims(stateNum)) < state_vars.dVec(stateNum)/2 ) {
            dfdx(i) = (f(i + int(state_vars.increVec(stateNum)) ) - f(i) ) / state_vars.dVec(stateNum);
        } else {
            dfdx(i) = (f(i + int(state_vars.increVec(stateNum))) - f(i - int(state_vars.increVec(stateNum))) ) / (2 * state_vars.dVec(stateNum));
        }
    }
    
}

////Second Derivatives given stateVars
int computeSecondDer(stateVars & state_vars, int stateNum, Eigen::Ref<Eigen::ArrayXd>  f, Eigen::Ref<Eigen::ArrayXd> dfdx) {
    for (int i = 0; i < state_vars.S; i++) {
        if ( abs(state_vars.stateMat(i, stateNum) - state_vars.upperLims(stateNum)) < state_vars.dVec(stateNum)/2 ) {
            dfdx(i) = (f(i) - 2 * f(i - int(state_vars.increVec(stateNum)) ) + f(i - 2 * int(state_vars.increVec(stateNum)) ) ) / (pow(state_vars.dVec(stateNum),2));
        } else if ( abs(state_vars.stateMat(i, stateNum) - state_vars.lowerLims(stateNum)) < state_vars.dVec(stateNum)/2 ) {
            dfdx(i) = (f(i + 2 * int(state_vars.increVec(stateNum)) )  - 2 * f(i + int(state_vars.increVec(stateNum)) ) + f(i) ) / (pow(state_vars.dVec(stateNum),2));
        } else {
            dfdx(i) = (f(i + int(state_vars.increVec(stateNum))) - 2 * f(i) + f(i - int(state_vars.increVec(stateNum))) ) / (pow(state_vars.dVec(stateNum),2));
        }
    }
    
}


int choose(int n, int k) {
    if (k == 0) {
        return 1;
    } else {
        return ( n * choose(n - 1, k - 1) ) / k;
    }
}

derivs::derivs (stateVars & state_vars) {
    firstPartials.resize(state_vars.S, state_vars.N);
    secondPartials.resize(state_vars.S, state_vars.N);
    int ch = choose(state_vars.N, 2);
    crossPartials.resize(state_vars.S, choose(state_vars.N, 2) );
    #ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    #else
        my_rank = 0;
    #endif
    if(my_rank == MASTER)
        std::cout<<"Finished initializing derivs" << std::endl;
}

void  derivs::computeDerivs (Eigen::Ref<Eigen::ArrayXd>  f, stateVars & state_vars) {
    int k = choose(state_vars.N, 2) - 1;
    //compute first partials
    for (int n = (state_vars.N - 1); n >= 0; --n) {
        computeDer(state_vars, n, f, firstPartials.col(n) );
        computeSecondDer(state_vars, n, f, secondPartials.col(n) );
        
        for (int n_sub = n-1; n_sub >=0; --n_sub) {
            
            computeDer(state_vars, n_sub, firstPartials.col(n), crossPartials.col(k) );
            k = k - 1;

        }
    }


    
}


class Vars {
    
public:
    
    //capital price and investment rate
    Eigen::ArrayXd q; Eigen::ArrayXd q_old; Eigen::ArrayXd qStar;
    Eigen::ArrayXd logQ;
    Eigen::ArrayXd oneOmegaQ;
    Eigen::ArrayXd omegaQ;
    Eigen::ArrayXd iota;
    
    //Volatilities
    Eigen::MatrixXd sigmaK;
    Eigen::MatrixXd sigmaQ;
    Eigen::MatrixXd sigmaR;
    std::vector<Eigen::MatrixXd> sigmaXVec;
    
    
    //Drifts
    Eigen::ArrayXd muK;
    Eigen::ArrayXd muQ;
    Eigen::MatrixXd muX;
    Eigen::ArrayXd muRe;
    Eigen::ArrayXd muRh;
    
    
    Eigen::ArrayXd normR2;
    Eigen::MatrixXd Pi;
    Eigen::ArrayXd deltaE;
    Eigen::ArrayXd deltaEStar;
    Eigen::ArrayXd deltaH;
    Eigen::ArrayXd deltaE_last;
    Eigen::ArrayXd trace;
    Eigen::ArrayXd r;
    Eigen::ArrayXd cHat_e;
    Eigen::ArrayXd cHat_h;
    Eigen::ArrayXd beta_e;
    Eigen::ArrayXd beta_h;
    Eigen::ArrayXd betaEDeltaE;
    Eigen::ArrayXd betaHDeltaH;
    
    Eigen::MatrixXd Dx;
    Eigen::ArrayXd DzetaOmega;
    Eigen::ArrayXd DzetaX;
    
    Eigen::MatrixXd idenMat; Eigen::VectorXd derivs_temp; Eigen::MatrixXd sigmaX_temp;
    

    
    Vars(stateVars & , Eigen::ArrayXd);
    void updateVars(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2,
                    derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
    void updateSigmaPi(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2,
                    derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
    void updateMuAndR(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2,
                       derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
    void updateDeltaEtAl(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2,
                      derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
    void updateDerivs(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2,
                         derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
};

Vars::Vars(stateVars & state_vars, Eigen::ArrayXd qGuess) {
    q.resize(state_vars.S); q_old.resize(state_vars.S); qStar.resize(state_vars.S); oneOmegaQ.resize(state_vars.S); logQ.resize(state_vars.S); omegaQ.resize(state_vars.S); iota.resize(state_vars.S);
    q = qGuess; q_old = q;
    
    sigmaK.resize(state_vars.S, state_vars.N); sigmaQ.resize(state_vars.S, state_vars.N); sigmaR.resize(state_vars.S, state_vars.N);
    
    for (int n = 0; n < state_vars.N; ++n) {
        Eigen::MatrixXd T; T.resize(state_vars.S,state_vars.N);
        sigmaXVec.push_back(T);
    }
    
    
    
    normR2.resize(state_vars.S);
    Pi.resize(state_vars.S,state_vars.N);
    deltaE.resize(state_vars.S); deltaEStar.resize(state_vars.S); deltaH.resize(state_vars.S); deltaE_last.resize(state_vars.S);
    muK.resize(state_vars.S);

    trace.resize(state_vars.S);
    r.resize(state_vars.S);
    muQ.resize(state_vars.S);
    muX.resize(state_vars.S, state_vars.N);
    muRe.resize(state_vars.S);
    muRh.resize(state_vars.S);
    cHat_e.resize(state_vars.S);
    cHat_h.resize(state_vars.S);
    beta_e.resize(state_vars.S); betaEDeltaE.resize(state_vars.S);
    beta_h.resize(state_vars.S); betaHDeltaH.resize(state_vars.S);
    
    Dx.resize(state_vars.S, state_vars.N);
    DzetaOmega.resize(state_vars.S); DzetaX.resize(state_vars.S);
    
    idenMat.resize(state_vars.N, state_vars.N); derivs_temp.resize(state_vars.N, 1); sigmaX_temp.resize(state_vars.N, state_vars.N);

    
}

void Vars::updateSigmaPi(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6) {
    
    if (state_vars.N == 1) {
        sigmaK.noalias() = state_vars.s.sqrt().matrix() * sigma_A_norm;// Eigen::Map<Eigen::VectorXd>(sigma_A_norm, state_vars.N, 1).transpose();
    } else {
        sigmaK = (state_vars.sigma_A.transpose().replicate(state_vars.S, 1)).array().colwise() * state_vars.s.sqrt().array();
    }
    
    for (int n = 0; n < state_vars.N; ++n) {
        
        if (state_vars.N == 1) {
            sigmaQ.col(n) = ( ( ( value_vars.kappa * value_vars.chi  -state_vars.omega) * derivs3.firstPartials.col(n).array() * sigmaK.col(n).array() ) / (1 - ( value_vars.kappa * value_vars.chi -state_vars.omega) * derivs3.firstPartials.col(n).array() ) ).matrix();
        } else if (state_vars.N == 2) {
            sigmaQ.col(n) = ( ( ( value_vars.kappa * value_vars.chi  -state_vars.omega) * derivs3.firstPartials.col(0).array() * sigmaK.col(n).array() + state_vars.s.sqrt() * state_vars.sigma_g(n) * derivs3.firstPartials.col(1).array() ) / (1 - ( value_vars.kappa * value_vars.chi -state_vars.omega) * derivs3.firstPartials.col(0).array() ) ).matrix();
        } else if (state_vars.N == 3) {
            sigmaQ.col(n) = ( ( ( value_vars.kappa * value_vars.chi  -state_vars.omega) * derivs3.firstPartials.col(0).array() * sigmaK.col(n).array() + state_vars.s.sqrt() * state_vars.sigma_g(n) * derivs3.firstPartials.col(1).array() + derivs3.firstPartials.col(2).array() * state_vars.s.sqrt() * state_vars.sigma_s(n) ) / (1 - ( value_vars.kappa * value_vars.chi -state_vars.omega) * derivs3.firstPartials.col(0).array() ) ).matrix();
            
        } else if (state_vars.N == 4) {
            sigmaQ.col(n) = ( ( ( value_vars.kappa * value_vars.chi  -state_vars.omega) * derivs3.firstPartials.col(n).array() * sigmaK.col(n).array() + state_vars.s.sqrt() * state_vars.sigma_g(n) * derivs3.firstPartials.col(n).array() + derivs3.firstPartials.col(n).array() * state_vars.s.sqrt() * state_vars.sigma_s(n) + derivs3.firstPartials.col(n).array() * state_vars.sigma_varsigma(n) ) / (1 - ( value_vars.kappa * value_vars.chi -state_vars.omega) * derivs3.firstPartials.col(n).array() ) ).matrix();
        }
        

        
    }

    sigmaR = sigmaQ + sigmaK;
    normR2 = sigmaR.rowwise().norm().array().pow(2);
  
    for (int n = 0; n < state_vars.N; ++n) {
        
        if (n == 0) {
            sigmaXVec[n] = (sigmaR).array().colwise() * ( (value_vars.kappa * value_vars.chi - state_vars.omega) );
        } else if (n == 1) {
            sigmaXVec[n] = (state_vars.sigma_g.transpose().replicate(state_vars.S, 1)).array().colwise() * state_vars.s.sqrt().array();
        } else if (n == 2) {
            sigmaXVec[n] = (state_vars.sigma_s.transpose().replicate(state_vars.S, 1)).array().colwise() * state_vars.s.sqrt().array();
        } else if (n == 3) {
            sigmaXVec[n] = (state_vars.sigma_varsigma.transpose().replicate(state_vars.S, 1)).array().colwise() * state_vars.s.sqrt().array();
        }
        
        
    }
    Pi =  (sigmaR).array().colwise() * (  gamma_h * (1.0 - value_vars.chi * value_vars.kappa) / (1.0 - state_vars.omega)  );
    
    for (int n = 0; n < state_vars.N; ++n) {
        
        if (state_vars.N == 1) {
            
            Pi.col(n) = Pi.col(n).array() + (gamma_h - 1.0) * (sigmaXVec[0].col(n).array() * derivs2.firstPartials.col(0).array());
                                                                                                              
        } else if (state_vars.N == 2) {
            
            Pi.col(n) = Pi.col(n).array() + (gamma_h - 1.0) * (sigmaXVec[0].col(n).array() * derivs2.firstPartials.col(0).array() + sigmaXVec[1].col(n).array() * derivs2.firstPartials.col(1).array() );

        } else if (state_vars.N == 3) {

            Pi.col(n) = Pi.col(n).array() + (gamma_h - 1.0) * (sigmaXVec[0].col(n).array() * derivs2.firstPartials.col(0).array() + sigmaXVec[1].col(n).array() * derivs2.firstPartials.col(1).array() + sigmaXVec[2].col(n).array() * derivs2.firstPartials.col(2).array() );
            
        } else if (state_vars.N == 4) {

            Pi.col(n) = Pi.col(n).array() + (gamma_h - 1.0) * (sigmaXVec[0].col(n).array() * derivs2.firstPartials.col(0).array() + sigmaXVec[1].col(n).array() * derivs2.firstPartials.col(1).array() + sigmaXVec[2].col(n).array() * derivs2.firstPartials.col(2).array() + sigmaXVec[3].col(n).array() * derivs2.firstPartials.col(3).array() );
            
        }
        
        
    }
    
};

void Vars::updateMuAndR(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6) {


    muK = state_vars.g + iota - delta;
    
    
    
    for (int n = 0; n < state_vars.N; ++n ) {
        
        if (n == 0) {

            muX.col(n) = state_vars.omega * (1.0 - state_vars.omega) * ( pow(rho_h, 1/psi_h) * value_vars.zeta_h.exp().pow(1-1/psi_h) - pow(rho_e, 1/psi_e) * value_vars.zeta_e.exp().pow(1-1/psi_e) + betaEDeltaE - betaHDeltaH ) + (value_vars.chi * value_vars.kappa - state_vars.omega) * ( sigmaR.cwiseProduct(Pi - sigmaR).rowwise().sum().array() ) + lambda_d * ( nu_newborn - state_vars.omega);
            
        } else if (n == 1) {
            
            muX.col(n) = lambda_g * (g_bar - state_vars.g);
            


        } else if (n == 2) {
            muX.col(n) = lambda_s * (s_bar - state_vars.s);
        } else if (n == 3) {
            muX.col(n) = lambda_varsigma * (varsigma_bar - state_vars.varSig);
        }
        

    }

    if (state_vars.N == 1) {
        trace = derivs5.secondPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[0]).rowwise().sum().array() );
    } else if (state_vars.N == 2) {
        trace = derivs5.secondPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[0]).rowwise().sum().array() ) + derivs5.secondPartials.col(1).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[0]).rowwise().sum().array() ) + 2 * derivs5.crossPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[1]).rowwise().sum().array() );
    } else if (state_vars.N == 3) {
        trace = derivs5.secondPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[0]).rowwise().sum().array() ) + derivs5.secondPartials.col(1).array() * ( sigmaXVec[1].cwiseProduct(sigmaXVec[1]).rowwise().sum().array() ) +  derivs5.secondPartials.col(2).array() * ( sigmaXVec[2].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) + 2 * (derivs5.crossPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[1]).rowwise().sum().array() ) + derivs5.crossPartials.col(1).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) + derivs5.crossPartials.col(2).array() * ( sigmaXVec[1].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) );
    } else if (state_vars.N == 4) {
        trace = derivs5.secondPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[0]).rowwise().sum().array() ) + derivs5.secondPartials.col(1).array() * ( sigmaXVec[1].cwiseProduct(sigmaXVec[1]).rowwise().sum().array() ) + derivs5.crossPartials.col(0).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[1]).rowwise().sum().array() ) + derivs5.secondPartials.col(2).array() * ( sigmaXVec[2].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) + derivs5.crossPartials.col(1).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) + derivs5.crossPartials.col(2).array() * ( sigmaXVec[1].cwiseProduct(sigmaXVec[2]).rowwise().sum().array() ) + derivs5.secondPartials.col(3).array() * ( sigmaXVec[3].cwiseProduct(sigmaXVec[3]).rowwise().sum().array() ) + derivs5.crossPartials.col(3).array() * ( sigmaXVec[0].cwiseProduct(sigmaXVec[3]).rowwise().sum().array() ) + derivs5.crossPartials.col(4).array() * ( sigmaXVec[1].cwiseProduct(sigmaXVec[3]).rowwise().sum().array() ) + derivs5.crossPartials.col(5).array() * ( sigmaXVec[2].cwiseProduct(sigmaXVec[3]).rowwise().sum().array() ) ;
    }

    if (state_vars.N == 1) {
        muQ = 1.0 / q  * ( muX.col(0).array() * derivs5.firstPartials.col(0).array() + 0.5 * trace );
    } else if (state_vars.N == 2) {
        muQ = 1.0 / q  * ( muX.col(0).array() * derivs5.firstPartials.col(0).array() + muX.col(1).array() * derivs5.firstPartials.col(1).array() + 0.5 * trace );
    } else if (state_vars.N == 3) {
        muQ = 1.0 / q  * ( muX.col(0).array() * derivs5.firstPartials.col(0).array() + muX.col(1).array() * derivs5.firstPartials.col(1).array() + muX.col(2).array() * derivs5.firstPartials.col(2).array() + 0.5 * trace );
    } else if (state_vars.N == 4) {
        muQ = 1.0 / q  * ( muX.col(0).array() * derivs5.firstPartials.col(0).array() + muX.col(1).array() * derivs5.firstPartials.col(1).array() + muX.col(2).array() * derivs5.firstPartials.col(2).array() + muX.col(3).array() * derivs5.firstPartials.col(3).array() + 0.5 * trace);
    }
    
    r = muQ + muK + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array() - sigmaR.cwiseProduct(Pi).rowwise().sum().array() - (1.0 - state_vars.omega) * (betaHDeltaH - pow(rho_h, 1/psi_h) * value_vars.zeta_h.exp().pow(1-1/psi_h) ) - state_vars.omega * (betaEDeltaE - pow(rho_e, 1/psi_e) * value_vars.zeta_e.exp().pow(1-1/psi_e) );
    
    muRe = (a_e - 1.0/phi * ( exp(phi * iota) - 1.0 ) ) / q + iota - delta + state_vars.g + muQ + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array();
    muRh = (a_h - 1.0/phi * ( exp(phi * iota) - 1.0 ) ) / q + iota - delta + state_vars.g + muQ + sigmaK.cwiseProduct(sigmaQ).rowwise().sum().array();

    
};


void Vars::updateDeltaEtAl(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6) {
    
    deltaE = pow(chiUnderline, -1.0) * (muRe - r - Pi.cwiseProduct(sigmaR).rowwise().sum().array() );
    deltaE = ( deltaE < 0 ).cast<double>() * 0 + ( deltaE >= 0 ).cast<double>() * deltaE;  //impose lower bound of 0 on detalE
    deltaH = (muRh - r - Pi.cwiseProduct(sigmaR).rowwise().sum().array() );
    
    cHat_e = pow(rho_e, 1/psi_e) * value_vars.zeta_e.exp().pow(1-1/psi_e);
    cHat_h = pow(rho_h, 1/psi_h) * value_vars.zeta_h.exp().pow(1-1/psi_h);
    
};

void Vars::updateDerivs(stateVars & state_vars, valueVars & value_vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6) {
    
    //compute derivs
    
    derivs1.computeDerivs(value_vars.zeta_e, state_vars);
    derivs2.computeDerivs(value_vars.zeta_h, state_vars);
    derivs3.computeDerivs(logQ, state_vars);
    derivs4.computeDerivs(oneOmegaQ, state_vars);
    derivs5.computeDerivs(q, state_vars);
    derivs6.computeDerivs(omegaQ, state_vars);


    
};



class matrixVars {

public:
    Eigen::VectorXd Fe;
    Eigen::MatrixXd firstCoefsE; Eigen::MatrixXd secondCoefsE;
    Eigen::VectorXd Fh;
    Eigen::MatrixXd firstCoefsH; Eigen::MatrixXd secondCoefsH;
    Eigen::ArrayXd atBoundIndicators;
    
    Eigen::VectorXd Ue; Eigen::VectorXd Uh;
    
    Eigen::MatrixXd sigmaX_temp; Eigen::VectorXd derivs_temp; Eigen::MatrixXd idenMat; Eigen::MatrixXd tempResult;
    
    std::vector<T> eList; std::vector<T> hList;
    SpMat Le; SpMat Lh;
    
    std::vector<double> a_e; std::vector<int> ia_e; std::vector<int> ja_e;
    std::vector<double> a_h; std::vector<int> ia_h; std::vector<int> ja_h;
    
    matrixVars(stateVars &);
    void updateMatrixVars(stateVars & state_vars, valueVars & value_vars, Vars & vars, derivs & derivs1, derivs & derivs2,
                    derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6);
    void updateMatrix(stateVars & state_vars);

    void updateKnowns(valueVars & value_vars, stateVars & state_vars);

    int my_rank;
    
};

matrixVars::matrixVars(stateVars & state_vars) {
    Fe.resize(state_vars.S); firstCoefsE.resize(state_vars.S, state_vars.N); secondCoefsE.resize(state_vars.S, state_vars.N);
    Fh.resize(state_vars.S); firstCoefsH.resize(state_vars.S, state_vars.N); secondCoefsH.resize(state_vars.S, state_vars.N);
    
    sigmaX_temp.resize(state_vars.N,state_vars.N); derivs_temp.resize(state_vars.N, 1); idenMat.resize(state_vars.N, state_vars.N); tempResult.resize(1,1);
    
    Le.resize(state_vars.S,state_vars.S); Lh.resize(state_vars.S,state_vars.S); eList.reserve(7 * state_vars.S); hList.reserve(7 * state_vars.S);
    Ue.resize(state_vars.S); Uh.resize(state_vars.S);
    
    #ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    #else
        my_rank = 0;
    #endif
    
    //find indices

}

void matrixVars::updateMatrixVars(stateVars & state_vars, valueVars & value_vars, Vars & vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6) {
    
    ///update terms for households pde
    
    if (psi_h == 1.0) {
        Fh = (-value_vars.zeta_h + log(rho_h)) * rho_h - rho_h;
    } else if (psi_h != 1.0) {
        Fh = ( psi_h / (1-psi_h) * pow(rho_h, 1/psi_h) * value_vars.zeta_h.exp().pow((1- 1/psi_h)) - rho_h / (1 - psi_h) ) ;
    }
    
    if(my_rank == MASTER)
        std::cout<<"\nFinished special handling for Fh"<<std::endl;
    if (state_vars.N == 1) {
        Fh = Fh.array() + vars.r + ( vars.Pi.rowwise().norm().array().pow(2) ) / (2*gamma_h)+ 0.5 * (1-gamma_h)/gamma_h * ( ( vars.sigmaXVec[0].col(0).array() * derivs2.firstPartials.col(0).array() * (state_vars.N >= 1) ).pow(2)   );
    }   else if (state_vars.N == 2) {
        Fh = Fh.array() + vars.r + (  vars.Pi.rowwise().norm().array().pow(2)  ) / (2*gamma_h)+ 0.5 * (1-gamma_h)/gamma_h * ( ( vars.sigmaXVec[0].col(0).array() * derivs2.firstPartials.col(0).array() + vars.sigmaXVec[1].col(0).array() * derivs2.firstPartials.col(1).array() ).pow(2) + ( vars.sigmaXVec[0].col(1).array() * derivs2.firstPartials.col(0).array() + vars.sigmaXVec[1].col(1).array() * derivs2.firstPartials.col(1).array() ).pow(2)) + derivs2.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) ;
        
    }
    
    else if (state_vars.N == 3) {
        Fh = Fh.array() + vars.r + (  vars.Pi.rowwise().norm().array().pow(2)  ) / (2*gamma_h)+ 0.5 * (1-gamma_h)/gamma_h * ( ( vars.sigmaXVec[0].col(0).array() * derivs2.firstPartials.col(0).array() + vars.sigmaXVec[1].col(0).array() * derivs2.firstPartials.col(1).array() + vars.sigmaXVec[2].col(0).array() * derivs2.firstPartials.col(2).array() ).pow(2) + ( vars.sigmaXVec[0].col(1).array() * derivs2.firstPartials.col(0).array() + vars.sigmaXVec[1].col(1).array() * derivs2.firstPartials.col(1).array() + vars.sigmaXVec[2].col(1).array() * derivs2.firstPartials.col(2).array() ).pow(2) + ( vars.sigmaXVec[0].col(2).array() * derivs2.firstPartials.col(0).array() + vars.sigmaXVec[1].col(2).array() * derivs2.firstPartials.col(1).array() + vars.sigmaXVec[2].col(2).array() * derivs2.firstPartials.col(2).array() ).pow(2) ) + derivs2.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) + derivs2.crossPartials.col(1).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[2]).rowwise().sum().array()   ) + derivs2.crossPartials.col(2).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array()  );

    } else if (state_vars.N == 4) {
        Fh = Fh.array() + vars.r + ( vars.Pi.rowwise().norm().array().pow(2) ) / (2*gamma_h)+ 0.5 * (1-gamma_h)/gamma_h * ( ( vars.sigmaXVec[0].col(0).array() * derivs2.firstPartials.col(0).array() * (state_vars.N >= 1) + vars.sigmaXVec[1].col(0).array() * derivs2.firstPartials.col(1).array() * (state_vars.N >= 2) + vars.sigmaXVec[2].col(0).array() * derivs2.firstPartials.col(2).array() * (state_vars.N >= 3) + vars.sigmaXVec[3].col(0).array() * derivs2.firstPartials.col(3).array() * (state_vars.N >= 4) ).pow(2) + (state_vars.N >= 2) * ( vars.sigmaXVec[0].col(1).array() * derivs2.firstPartials.col(0).array() * (state_vars.N >= 1) + vars.sigmaXVec[1].col(1).array() * derivs2.firstPartials.col(1).array() * (state_vars.N >= 2) + vars.sigmaXVec[2].col(1).array() * derivs2.firstPartials.col(2).array() * (state_vars.N >= 3) + vars.sigmaXVec[3].col(1).array() * derivs2.firstPartials.col(3).array()* (state_vars.N >= 4) ).pow(2)  + (state_vars.N >= 3) * ( vars.sigmaXVec[0].col(2).array() * derivs2.firstPartials.col(0).array() * (state_vars.N >= 1) + vars.sigmaXVec[1].col(2).array() * derivs2.firstPartials.col(1).array() * (state_vars.N >= 2) + vars.sigmaXVec[2].col(2).array() * derivs2.firstPartials.col(2).array() * (state_vars.N >= 3) + vars.sigmaXVec[3].col(2).array() * derivs2.firstPartials.col(3).array() * (state_vars.N >= 4)  ).pow(2) + (state_vars.N >= 4) * ( vars.sigmaXVec[0].col(3).array() * derivs2.firstPartials.col(0).array() * (state_vars.N >= 1) + vars.sigmaXVec[1].col(3).array() * derivs2.firstPartials.col(1).array() * (state_vars.N >= 2)+ vars.sigmaXVec[2].col(3).array() * derivs2.firstPartials.col(2).array() * (state_vars.N >= 3) + vars.sigmaXVec[3].col(3).array() * derivs2.firstPartials.col(3).array() * (state_vars.N >= 4) ).pow(2) ) + derivs2.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() * (state_vars.N >= 2) ) + derivs2.crossPartials.col(1).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[2]).rowwise().sum().array()  * (state_vars.N >= 3) ) + derivs2.crossPartials.col(2).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array()  * (state_vars.N >= 3) ) + derivs2.crossPartials.col(3).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array() * (state_vars.N >= 4) ) + derivs2.crossPartials.col(4).array() * ( vars.sigmaXVec[1].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array() * (state_vars.N >= 4) ) + derivs2.crossPartials.col(5).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array() * (state_vars.N >= 4)  ) ;
    }

    if(my_rank == MASTER)
        std::cout<<"\nFinished Fh"<<std::endl;
        
    for (int n = 0; n < state_vars.N; ++n ) {
        firstCoefsH.col(n) = vars.muX.col(n).array() + (1 - gamma_h) / gamma_h * ( vars.sigmaXVec[n].cwiseProduct(vars.Pi).rowwise().sum().array() );
        secondCoefsH.col(n) = 0.5 *  ( vars.sigmaXVec[n].cwiseProduct(vars.sigmaXVec[n]).rowwise().sum().array()  );
    }
    

    ///update terms for experts pde
    
    if (psi_e == 1.0) {
        Fe = (-value_vars.zeta_e + log(rho_e)) * rho_e - rho_e;
    } else if (psi_e != 1.0)  {
        Fe = psi_e / (1 - psi_e) * pow(rho_e, 1/psi_e) * ( value_vars.zeta_e).exp().pow((1- 1/psi_e)) - rho_e / (1 - psi_e);
    }
    
    if (state_vars.N == 1) {
        Fe = Fe.array() + vars.r + nu / (1-gamma_e) * ( ( (1-gamma_e) * (value_vars.zeta_h - value_vars.zeta_e)).exp() - 1) + (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() ).pow(2) / (2 * gamma_e * vars.normR2);
        
    } else if (state_vars.N == 2) {
        Fe = Fe.array() + vars.r + nu / (1-gamma_e) * ( ( (1-gamma_e) * (value_vars.zeta_h - value_vars.zeta_e)).exp() - 1) + (vars.deltaE + vars.Pi.col(0).array() * vars.sigmaR.col(0).array() + vars.Pi.col(1).array() * vars.sigmaR.col(1).array() ).pow(2) / (2 * gamma_e * vars.normR2) + derivs1.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) ;
    
    }
    else if (state_vars.N == 3) {
        Fe = Fe.array() + vars.r + nu / (1-gamma_e) * ( ( (1-gamma_e) * (value_vars.zeta_h - value_vars.zeta_e)).exp() - 1) + (vars.deltaE + vars.Pi.col(0).array() * vars.sigmaR.col(0).array() + vars.Pi.col(1).array() * vars.sigmaR.col(1).array() + vars.Pi.col(2).array() * vars.sigmaR.col(2).array() ).pow(2) / (2 * gamma_e * vars.normR2) + derivs1.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) + derivs1.crossPartials.col(1).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[2]).rowwise().sum().array()   ) + derivs1.crossPartials.col(2).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array()  );
    } else if (state_vars.N == 4)  {
        
        Fe = Fe.array() + vars.r + nu / (1-gamma_e) * ( ( (1-gamma_e) * (value_vars.zeta_h - value_vars.zeta_e)).exp() - 1) + (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() ).pow(2) / (2 * gamma_e * vars.normR2) + derivs1.crossPartials.col(0).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) * (state_vars.N >= 2) + derivs1.crossPartials.col(1).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[2]).rowwise().sum().array() ) * (state_vars.N >= 3)  + derivs1.crossPartials.col(2).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[1]).rowwise().sum().array() ) * (state_vars.N >= 3)  + derivs1.crossPartials.col(3).array() * ( vars.sigmaXVec[0].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array()  ) * (state_vars.N >= 4) + derivs1.crossPartials.col(4).array() * ( vars.sigmaXVec[1].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array()  ) * (state_vars.N >= 4) + derivs1.crossPartials.col(5).array() * ( vars.sigmaXVec[2].cwiseProduct(vars.sigmaXVec[3]).rowwise().sum().array()  ) * (state_vars.N >= 4);
        
    }
    
    

    //take care of the quadratic term for Fe
    idenMat = Eigen::VectorXd::Constant(state_vars.N, gamma_e).asDiagonal();

    for(int i = 0; i < state_vars.S; i++) {
        
        if (state_vars.N == 1) {
            
            sigmaX_temp << vars.sigmaXVec[0](i,0);
            
            derivs_temp << derivs1.firstPartials(i,0);
        } else if (state_vars.N == 2) {
            
            sigmaX_temp << vars.sigmaXVec[0](i,0), vars.sigmaXVec[0](i,1), vars.sigmaXVec[1](i,0), vars.sigmaXVec[1](i,1);
            derivs_temp << derivs1.firstPartials(i,0), derivs1.firstPartials(i,1);
        } else if (state_vars.N == 3) {
            
            sigmaX_temp << vars.sigmaXVec[0](i,0), vars.sigmaXVec[0](i,1), vars.sigmaXVec[0](i,2), vars.sigmaXVec[1](i,0), vars.sigmaXVec[1](i,1), vars.sigmaXVec[1](i,2), vars.sigmaXVec[2](i,0), vars.sigmaXVec[2](i,1), vars.sigmaXVec[2](i,2);
            derivs_temp << derivs1.firstPartials(i,0), derivs1.firstPartials(i,1), derivs1.firstPartials(i,2);
            
        } else if (state_vars.N == 4) {
            
            sigmaX_temp << vars.sigmaXVec[0](i,0), vars.sigmaXVec[0](i,1), vars.sigmaXVec[0](i,2), vars.sigmaXVec[0](i,3), vars.sigmaXVec[1](i,0), vars.sigmaXVec[1](i,1), vars.sigmaXVec[1](i,2), vars.sigmaXVec[1](i,3), vars.sigmaXVec[2](i,0), vars.sigmaXVec[2](i,1), vars.sigmaXVec[2](i,2), vars.sigmaXVec[2](i,3), vars.sigmaXVec[3](i,0), vars.sigmaXVec[3](i,1), vars.sigmaXVec[3](i,2), vars.sigmaXVec[3](i,3);
            
            derivs_temp << derivs1.firstPartials(i,0), derivs1.firstPartials(i,1), derivs1.firstPartials(i,2), derivs1.firstPartials(i,3);
            
        }
        
        
        tempResult.noalias() = 0.5 * (1 - gamma_e) / gamma_e *   derivs_temp.transpose() * sigmaX_temp * (idenMat + (1 - gamma_e) * vars.sigmaR.row(i).transpose() * vars.sigmaR.row(i) / vars.normR2(i) ) * sigmaX_temp.transpose() * derivs_temp;

        Fe(i) = Fe(i) + tempResult(0,0);
        
    }
    
    for (int n = 0; n < state_vars.N; ++n ) {
        firstCoefsE.col(n) = vars.muX.col(n).array() + (1 - gamma_e) / gamma_e * ( vars.sigmaXVec[n].cwiseProduct(vars.sigmaR).rowwise().sum().array() ) * (vars.deltaE + vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array()  ) / vars.normR2;
        
        secondCoefsE.col(n) = 0.5 *  ( vars.sigmaXVec[n].cwiseProduct(vars.sigmaXVec[n]).rowwise().sum().array()  );
    }
    
}

void matrixVars::updateMatrix(stateVars & state_vars) {
    atBoundIndicators.resize(state_vars.N); 
    eList.clear(); hList.clear();    
    //construct matrix
    for (int i = 0; i < state_vars.S; ++i) {
        
        

        eList.push_back(T(i,i, 1.0 ));
        hList.push_back(T(i,i, 1.0 ));
        
        //check boundaries
        
        for (int n = (state_vars.N - 1); n >=0; --n ) {
            atBoundIndicators(n) = -1.0;    
            double firstCoefE = firstCoefsE(i,n);
            double secondCoefE = secondCoefsE(i,n);
            double firstCoefH = firstCoefsH(i,n);
            double secondCoefH = secondCoefsH(i,n);
            
            //check whether it's at upper or lower boundary
            if ( abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n)/2 ) {  //upper boundary
    
            	atBoundIndicators(n) = 1.0; 
                eList.push_back(T(i, i, - dt * ( firstCoefE/state_vars.dVec(n) + secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                eList.push_back(T(i, i - state_vars.increVec(n), - dt * ( - firstCoefE/state_vars.dVec(n) - 2 * secondCoefE / pow(state_vars.dVec(n), 2) ) ));
                eList.push_back(T(i, i - 2*state_vars.increVec(n), - dt * ( secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                
                hList.push_back(T(i, i, - dt * ( firstCoefH/state_vars.dVec(n) + secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                hList.push_back(T(i, i - state_vars.increVec(n), - dt * ( - firstCoefH/state_vars.dVec(n) - 2 * secondCoefH / pow(state_vars.dVec(n), 2) ) ));
                hList.push_back(T(i, i - 2*state_vars.increVec(n), - dt * ( secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                
            } else if ( abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n)/2 ) { //lower boundary
                
                atBoundIndicators(n) = 1.0;         
                eList.push_back(T(i, i, - dt * ( - firstCoefE/state_vars.dVec(n) + secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                eList.push_back(T(i, i + state_vars.increVec(n), - dt * ( firstCoefE/state_vars.dVec(n) - 2 * secondCoefE / pow(state_vars.dVec(n), 2) ) ));
                eList.push_back(T(i, i + 2*state_vars.increVec(n), - dt * ( secondCoefE / pow(state_vars.dVec(n), 2) ) ) );
                
                hList.push_back(T(i, i, - dt * ( - firstCoefH/state_vars.dVec(n) + secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                hList.push_back(T(i, i + state_vars.increVec(n), - dt * ( firstCoefH/state_vars.dVec(n) - 2 * secondCoefH / pow(state_vars.dVec(n), 2) ) ));
                hList.push_back(T(i, i + 2*state_vars.increVec(n), - dt * ( secondCoefH / pow(state_vars.dVec(n), 2) ) ) );
                
                
            }
            
            
            
        }
        
        
        for (int n = (state_vars.N - 1); n >= 0; --n) {
            //level and time deriv
            
            
            //add elements to the vector of triplets for matrix construction
            if (atBoundIndicators(n) < 0) {
                
                double firstCoefE = firstCoefsE(i,n);
                double secondCoefE = secondCoefsE(i,n);
                double firstCoefH = firstCoefsH(i,n);
                double secondCoefH = secondCoefsH(i,n);

                //first derivative
                
                
                if (firstCoefE != 0.0) {
                    
                    eList.push_back(T(i,i, - dt * ( -firstCoefE * ( firstCoefE > 0) + firstCoefE * ( firstCoefE < 0) ) / state_vars.dVec(n)  ) );
                    eList.push_back(T(i,i + state_vars.increVec(n), - dt * firstCoefE * ( firstCoefE > 0) / state_vars.dVec(n) ));
                    eList.push_back(T(i,i - state_vars.increVec(n), - dt *  - firstCoefE * ( firstCoefE < 0) / state_vars.dVec(n) ));
                }

                if (firstCoefH != 0.0) {
                    hList.push_back(T(i,i, - dt * ( -firstCoefH * ( firstCoefH > 0) + firstCoefH * ( firstCoefH < 0) ) / state_vars.dVec(n)  ) );
                    hList.push_back(T(i,i + state_vars.increVec(n), - dt * firstCoefH * ( firstCoefH > 0) / state_vars.dVec(n) ));
                    hList.push_back(T(i,i - state_vars.increVec(n), - dt *  - firstCoefH * ( firstCoefH < 0) / state_vars.dVec(n) ));
                    
                }
                
                //second derivative
                
                if (secondCoefE != 0.0) {
                    eList.push_back(T(i, i, - dt * -2 * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                    eList.push_back(T(i, i + state_vars.increVec(n), - dt * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                    eList.push_back(T(i, i - state_vars.increVec(n), - dt * secondCoefE / ( pow(state_vars.dVec(n), 2) ) ));
                }
    
                if (secondCoefH != 0.0) {
                    hList.push_back(T(i, i, - dt * -2 * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                    hList.push_back(T(i, i + state_vars.increVec(n), - dt * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                    hList.push_back(T(i, i - state_vars.increVec(n), - dt * secondCoefH / ( pow(state_vars.dVec(n), 2) ) ));
                }
                
                
            }
            
        }
        
        
    }

    //form matrices
    Le.setFromTriplets(eList.begin(), eList.end());
    Lh.setFromTriplets(hList.begin(), hList.end());
    
    //compress
    Le.makeCompressed(); Lh.makeCompressed();

    //fill a, ia, ja
    ja_e.clear(); ja_h.clear(); a_e.clear(); a_h.clear(); ia_e.clear(); ia_h.clear();
    
    //ja_e.resize(Le.nonZeros()); ja_h.resize(Lh.nonZeros());
    //a_e.resize(Le.nonZeros()); //ia_e.resize(state_vars.S + 1);
    //a_h.resize(Lh.nonZeros()); //ia_h.resize(state_vars.S + 1);
    
    int* outerPtr_e = Le.outerIndexPtr(); int* outerPtr_h = Lh.outerIndexPtr();
    for (int i = 0; i < state_vars.S + 1; ++i) {
        ia_e.push_back(outerPtr_e[i] + 1);
    }
    for (int i = 0; i < state_vars.S + 1; ++i) {
        ia_h.push_back(outerPtr_h[i] + 1);
    }
    

    int* innerPtr_e = Le.innerIndexPtr(); int* innerPtr_h = Lh.innerIndexPtr();
    for (int i = 0; i < Le.nonZeros(); ++i) {
        ja_e.push_back(innerPtr_e[i] + 1);
    }
    for (int i = 0; i < Lh.nonZeros(); ++i) {
        ja_h.push_back(innerPtr_h[i] + 1);
    }

    double* valuePtr_e = Le.valuePtr(); double* valuePtr_h = Lh.valuePtr();
    for (int i = 0; i < Le.nonZeros(); ++i) {
        a_e.push_back(valuePtr_e[i]);
    }
    for (int i = 0; i < Lh.nonZeros(); ++i) {
        a_h.push_back(valuePtr_h[i]);
    }
    
    
}

void matrixVars::updateKnowns(valueVars & value_vars, stateVars & state_vars) {
    Ue = Fe.array() * dt + value_vars.zeta_e; Uh = Fh.array() * dt + value_vars.zeta_h;
    /*
    for (int i = 0; i < state_vars.S; ++i) {
        if ( state_vars.omega(i) == omegaMin) {  //omega_min: forward diff
		Ue(i) = 0.0; Uh(i) = 0.0;

        }
    }    */
}


class errorTerms {
    
public:
    Eigen::ArrayXd eErrors; Eigen::ArrayXd hErrors;
    Eigen::ArrayXd eErrorsRel; Eigen::ArrayXd hErrorsRel;
    
    errorTerms (int maxIt);
    
};

errorTerms::errorTerms (int maxIt) {
    
    eErrors.resize(maxIt); hErrors.resize(maxIt);
    eErrorsRel.resize(maxIt); hErrorsRel.resize(maxIt);
}



void exportData(valueVars & value_vars, Vars & vars,  derivs & derivs1, derivs & derivs2,  int i, int N) {
    
    
    /* Exporting data */
    std::cout<<"Exporting data..."<<std::endl;
    //Risk prices and interest rate
    writeArray(vars.q,folderName + run + "_"   + "q_" + std::to_string(i));
    writeArray(vars.Pi,folderName + run + "_"  + "Pi_" + std::to_string(i));
    writeArray(vars.deltaE,folderName + run + "_"  + "deltaE_" + std::to_string(i));
    writeArray(vars.deltaH,folderName + run + "_" + "deltaH_" + std::to_string(i));
    writeArray(vars.r,folderName + run+ "_"  + "r_" + std::to_string(i));
    std::cout<<"Exporting drifts"<<std::endl;
    //Drifts
    writeArray(vars.muQ,folderName + run + "_"  + "muQ_" + std::to_string(i));
    writeArray(vars.muX,folderName + run + "_" + "muX_" + std::to_string(i));
    writeArray(vars.muK,folderName + run + "_" + "muK_" + std::to_string(i));
    writeArray( vars.muRe,folderName + run + "_" + "muRe_" + std::to_string(i));
    writeArray( vars.muRh,folderName + run + "_" + "muRh_" + std::to_string(i));
    std::cout<<"Exporting vols"<<std::endl;
    //Volatilities
    writeArray( vars.sigmaQ,folderName + run + "_"  + "sigmaQ_" + std::to_string(i));
    writeArray( vars.sigmaR,folderName + run + "_"  + "sigmaR_" + std::to_string(i));
    
    for (int l = 0; l < vars.sigmaXVec.size(); l++) {
        writeArray( vars.sigmaXVec[l] ,folderName + run + "_"  + "sigmaX" + std::to_string(l+1) + "_" + std::to_string(i));
    }
    
    std::cout<<"Exporting value functions"<<std::endl;
    //Value functions
    writeArray( value_vars.zeta_e,folderName  + run  + "_" + "zeta_e_" + std::to_string(i));
    writeArray( value_vars.zeta_h,folderName + run  + "_" + "zeta_h_" + std::to_string(i));
    writeArray( vars.cHat_e,folderName + run + "_" + "cHat_e_" + std::to_string(i));
    writeArray( vars.cHat_h,folderName + run + "_" + "cHat_h_" + std::to_string(i));
    
    //Constraints
    writeArray( value_vars.kappa,folderName + run + "_" + "kappa_" + std::to_string(i));
    writeArray( value_vars.chi,folderName + run + "_" + "chi_" + std::to_string(i));
    writeArray( vars.beta_e,folderName + run + "_" + "betaE_" + std::to_string(i));
    writeArray( vars.beta_h,folderName + run + "_" + "betaH_" + std::to_string(i));
    std::cout<<"Exporting derivs"<<std::endl;
    //Derivatives
    for (int n = 0; n < N; n++) {
        writeArray( derivs1.firstPartials.col(n).array(),folderName + run + "_" + "dzeta_e_dx_" + std::to_string(n) + "_" + std::to_string(i));
        writeArray( derivs2.firstPartials.col(n).array(),folderName + run + "_" + "dzeta_h_dx_" + std::to_string(n) + "_" + std::to_string(i));
        writeArray( derivs1.secondPartials.col(n).array(),folderName + run + "_" + "dzeta_e_dx2_" + std::to_string(n) + "_" + std::to_string(i));
        writeArray( derivs2.secondPartials.col(n).array(),folderName + run + "_" + "dzeta_h_dx2_" + std::to_string(n) + "_" + std::to_string(i));

    }
    int ch = choose(N, 2);
    for (int n = 0; n < ch; n++) {
        writeArray( derivs1.crossPartials.col(n).array(),folderName + run + "_" + "dzeta_e_dxdy_" + std::to_string(n) + "_" + std::to_string(i));
        writeArray( derivs2.crossPartials.col(n).array(),folderName + run + "_" + "dzeta_h_dxdy_" + std::to_string(n) + "_" + std::to_string(i));
    }
    
    std::cout<<"Finished exporting data"<<std::endl;
    
}


//iteration function

int iterFunc(stateVars & state_vars, valueVars & value_vars, Vars & vars, matrixVars & matrix_vars, derivs & derivs1, derivs & derivs2, derivs & derivs3, derivs & derivs4, derivs & derivs5, derivs & derivs6, errorTerms & error_terms, int equityIss, int hhCap) {
    
    /* Explanation on equityIss and hhCap */
    
    /* equityIss = 0 if equity issuance is not allowed; chi = 1 always */
    /*           = 1 if equity issuance is allowed and skin-in-the-game constraint binds always; chi = chiUnderline < 1.0 */
    /*           = 2 if equity issuance is allowed and skin-in-the-game constraint binds occasionally */
    
    /* hhCap     = 0 if households are not allowed to hold capital; kappa = 1 always */
    /*           = 1 if households are allowed to hold capital; */
    
    
    int      n;
    int      nnzE;
    int      nnzH;
    int      mtype = 11;        /* Real unsymmetric matrix */
    int      nrhs = 1;
    void    *pt[64];
    int      iparm[64];
    double   dparm[64];
    int      solver;
    int      maxfct, mnum, phase, error, msglvl;
    int      num_procs;
    char    *var;
    int      i;
    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */
    double innerTol = 0.001; /* Tolerance for q and chi */
    double innerDt = 0.00000001;
    double innerError = 0;
    
    int my_rank = 0;
    
    iparm[0]  = 1;
    iparm[2]  = 28;      /* num of threads */
    //iparm[50] = 1;
    //iparm[51] = 2;       /* num of nodes */
    //iparm[11] = 0;      /* solve transpose  */
    //iparm[31] = 0;      /* select solver */
    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */
    solver = 0;
    n = state_vars.S;
    
   
    #ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    #else
        my_rank = 0;
    #endif


    /* Initialize arrays to store temporary data */
    Eigen::ArrayXd indicatorVec; indicatorVec.resize(state_vars.S, 1);
    double A; double B; double C;
    double epsilon = 0.1; double derivsQwrtOmega1; double derivsQwrtOmega2; double derivsQwrtOmega;
    Eigen::ArrayXd allErrorE; allErrorE.resize(state_vars.S);
    Eigen::ArrayXd allErrorH; allErrorH.resize(state_vars.S);
    Eigen::MatrixXf::Index index;
    Eigen::ArrayXd chiThres; chiThres.resize(maxIters);
    double thresSum = 0;
    /* Initialize Pardiso */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);
    
    /*  Export state variables */
    if(my_rank == MASTER){
        writeArray(state_vars.omega,folderName + run + "_" + "omega");
        writeArray(state_vars.g,folderName + run + "_" + "g");
        writeArray(state_vars.s, folderName + run + "_"  + "s");
        writeArray(state_vars.varSig, folderName + run + "_"  + "varSig");
        std::cout<<"\n finished exporting state vars"<<std::endl;
    }
    /* Start iterations */
    for(int i = 0; i < maxIters; i ++) {
     
        if(my_rank == MASTER)   
            std::cout<<"Run name: "<<run<<std::endl; 
        
        //******* Compute error and store data ************//
        
        allErrorE = (value_vars.zeta_e - value_vars.zeta_e_old).array() / dt;
        allErrorH = (value_vars.zeta_h - value_vars.zeta_h_old).array() / dt;
        allErrorE.abs().maxCoeff(&index); error_terms.eErrors(i) = allErrorE(index);
        allErrorH.abs().maxCoeff(&index); error_terms.hErrors(i) = allErrorH(index);

        error_terms.eErrorsRel(i) = (value_vars.zeta_e.exp() / value_vars.zeta_e_old.exp() - 1).abs().maxCoeff() / dt;
        error_terms.hErrorsRel(i) = (value_vars.zeta_h.exp() / value_vars.zeta_h_old.exp() - 1).abs().maxCoeff() / dt;
        value_vars.zeta_e_old = value_vars.zeta_e; value_vars.zeta_h_old = value_vars.zeta_h;
        value_vars.kappa_old = value_vars.kappa; value_vars.chi_old = value_vars.chi;
        
        /*******************************************/
        /* Export error */
        /*******************************************/
        if(my_rank == MASTER){
            writeArray(error_terms.eErrors, folderName + run + "_"  + "eErrors");
            writeArray(error_terms.hErrors,  folderName + run + "_" +  "hErrors");
            writeArray(error_terms.eErrorsRel, folderName + run + "_" +  "eErrorsRel");
            writeArray(error_terms.hErrorsRel,  folderName + run + "_" +  "hErrorsRel");
	        writeArray(chiThres,folderName + run + "_"   + "chiThres");
            std::cout<<"\nFinished exporting error"<<std::endl;    
        }
        /*********************************************************/
        /* Execute numerical algorithm */
        /*********************************************************/

        if (i == 0  || ( ( abs(error_terms.eErrors(i)) > tol) || ( abs(error_terms.hErrors(i)) > tol) ) ) {
            
            /*********************************************************/
            /* Step 0: Print out information */
            /*********************************************************/
            
            if(my_rank == MASTER){
                if (i == 0) {
                    std::cout<<"\n First iteration...";
                } else {
                    std::cout<<"Error for zeta_e: "<<error_terms.eErrors(i)<<"\n";
                    std::cout<<"Error for zeta_h: "<<error_terms.hErrors(i)<<"\n";
                    std::cout<<"Rel error for zeta_e: "<<error_terms.eErrorsRel(i)<<"\n";
                    std::cout<<"Rel error for zeta_h: "<<error_terms.hErrorsRel(i)<<"\n";
                    std::cout<<"Greatest change for kappa: "<<(value_vars.kappa - value_vars.kappa_old).maxCoeff()<<std::endl;
                    std::cout<<"Greatest change for chi: "<<(value_vars.chi - value_vars.chi_old).maxCoeff()<<std::endl;
                    std::cout<<"Tolerance not met; keep iterating... " << "iteration: " << i+1 << " \n";
                }
            }
            
            

            if (hhCap == 1) {  //Step 1 is executed only in hhCap = 1 where households are allowed to hold capital
                /*********************************************************/
                /* Step 1(a - b): Iterate for q_tilde and kappa */
                /*********************************************************/
                
                vars.updateDerivs(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
                vars.updateSigmaPi(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
                
                for (int j = 0; j < state_vars.S; j++) {
                    
                    if ( abs( state_vars.stateMat(j,0) - state_vars.lowerLims(0) ) < state_vars.dVec(0) / 2 ) {
                        //at omega_min, kappa = omega * (epsilon + chiUnderline ^(-1) )
                        //std::cout<<"omega_min; stateMat: "<<state_vars.stateMat.row(j)<<std::endl;
                        if (state_vars.N == 1) {
                            //In 1D, use L'hopitals rule to compute q, q', kappa, kappa' at omega = 0 (not omega = omega_min)
                            double q_zero = (a_h + 1.0 / phi) / ( pow(rho_h, 1.0/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h) + 1.0 / phi );
                            double kappa_zero_prime = 1.0 / (gamma_e * chiUnderline) * (gamma_h + (a_e - a_h) / (chiUnderline * q_zero * pow(sigma_A_norm,2 ) ));
                            double q_zero_prime = 1.0 / ( pow(rho_h, 1.0/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h) + 1.0 / phi ) *
                                            ((a_e - a_h) * kappa_zero_prime - q_zero * ( (1.0 - 1.0 /psi_h) * pow(rho_h, 1.0/psi_h) * pow(exp(value_vars.zeta_h(j)), - 1/psi_h ) * derivs2.firstPartials(j,0) + pow(rho_e, 1.0/psi_e) * pow(exp(value_vars.zeta_e(j)), 1 - 1/psi_e ) - pow(rho_h, 1.0/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h ) ));
                            value_vars.kappa(j) = state_vars.lowerLims(0) * kappa_zero_prime;
                            vars.q(j) = q_zero + q_zero_prime * state_vars.lowerLims(0);
                            if(my_rank == MASTER)
			                    std::cout<<"q_zero: "<<q_zero<<"; q_zero_prime: " << q_zero_prime << "; kappa_zero_prime: "
                                    << kappa_zero_prime << "; kappa: " << value_vars.kappa(j) << std::endl;
                        }
                        
                        /* FORMULAS THAT RELY ON EPSILON; TO BE MODIFIED
                        value_vars.kappa(j) = state_vars.stateMat(j,0) * (epsilon + 1.0/chiUnderline);
                        vars.q(j) = ( (1.0 - value_vars.kappa(j) ) * a_h + value_vars.kappa(j) * a_e + 1/phi ) / ( (1.0 -state_vars.omega(j)) * pow(rho_h, 1/psi_h) * pow( (exp(value_vars.zeta_h(j))), 1-1/psi_h) + state_vars.omega(j) * pow(rho_e,1/psi_e) * pow(exp(value_vars.zeta_e(j)), 1 - 1/psi_e) + 1/phi );
                        */
                    }
                    
                    if ( abs( state_vars.stateMat(j,0) - state_vars.lowerLims(0) ) < state_vars.dVec(0) / 2 ||   ( abs( state_vars.stateMat(j,0) - state_vars.lowerLims(0) ) > state_vars.dVec(0) / 2  && value_vars.kappa(j - state_vars.increVec(0)) < 1 ) )  {
                        //std::cout<<"not omega_min; previous kappa less than 1 stateMat: "<<state_vars.stateMat.row(j)<<std::endl;

                        if ( abs( state_vars.stateMat(j,0) - state_vars.lowerLims(0) ) > state_vars.dVec(0) / 2 ) {
                            value_vars.kappa(j) = 1.0 / (a_e - a_h) * ( vars.q(j) * ( (1-state_vars.omega(j)) * pow(rho_h, 1/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h) + state_vars.omega(j) * pow(rho_e,1/psi_e) * pow(exp(value_vars.zeta_e(j)), 1 - 1/psi_e) + 1/phi ) - 1.0 / phi - a_h);
                            value_vars.kappa(j) = value_vars.kappa(j) * (value_vars.kappa(j) <= 1) + 1.0 * (value_vars.kappa(j) > 1);
                            
                        }
                        
                        //If not at omega min and kappa is less than 1, compute kappa
                        //std::cout<<"kappa: "<<value_vars.kappa(j)<<std::endl;
                        
                        //First, solve for kappa by using (43)

                        //Compute deltaE, deltaH, A, B, and C
                        vars.deltaH(j) = gamma_h * ( 1.0 - value_vars.kappa(j) ) / (1.0 - state_vars.omega(j) ) * state_vars.varSig(j);
                        vars.deltaE(j) = ((a_e - a_h) / vars.q(j) + vars.deltaH(j)) / chiUnderline;
                        
                        //update Dx
                        if (state_vars.N == 1) {
                            vars.Dx.col(0) = vars.sigmaK.col(0).array();
                        } else if (state_vars.N == 3){
                            for (int n = 0; n < state_vars.N; ++n ) {
                                vars.Dx.col(n) = vars.sigmaK.col(n).array() + vars.sigmaXVec[1].col(n).array() * derivs3.firstPartials.col(1).array() + vars.sigmaXVec[2].col(n).array() * derivs3.firstPartials.col(2).array();
                                
                            }
                            
                        }
                        
                        //Update d_zeta_xtilde
                        if (state_vars.N == 1) {
                            vars.DzetaX = vars.Dx.col(0).array() * 0.0;
                        } else if (state_vars.N == 3){
                            vars.DzetaX = vars.Dx.col(0).array() * vars.sigmaXVec[1].col(0).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(0).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() ) + vars.Dx.col(1).array() * vars.sigmaXVec[1].col(1).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(1).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() ) + vars.Dx.col(2).array() * vars.sigmaXVec[1].col(2).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(2).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() );
                            
                            
                        }
                
                        
                        A = state_vars.omega(j) * ( 1 - state_vars.omega(j) ) * vars.deltaE(j) - ( 1 - state_vars.omega(j) ) * gamma_e * chiUnderline * state_vars.varSig(j) * value_vars.kappa(j);
                        B = state_vars.omega(j) * ( 1 - state_vars.omega(j) ) * vars.DzetaX(j);
                        C = ( state_vars.omega(j)  * gamma_h * (1.0 - chiUnderline * value_vars.kappa(j) ) - (1 - state_vars.omega(j) ) * gamma_e * chiUnderline * value_vars.kappa(j) + state_vars.omega(j) * (1 - state_vars.omega(j) ) * (chiUnderline * value_vars.kappa(j) - state_vars.omega(j) )  * ( (gamma_h - 1.0) * derivs2.firstPartials(j,0) - (gamma_e - 1.0) * derivs1.firstPartials(j,0) ) ) * (vars.Dx.rowwise().norm().array().pow(2))(j);
                        //C = (gamma_h * ( state_vars.omega(j) * (1.0 - chiUnderline * value_vars.kappa(j) ) / ( 1.0 - state_vars.omega(j) ) ) - gamma_e * value_vars.kappa(j) + state_vars.omega(j) * ( (chiUnderline * value_vars.kappa(j) - state_vars.omega(j)) * ( (gamma_h - 1.0) * derivs2.firstPartials(j,0) - (gamma_e - 1.0) * derivs1.firstPartials(j,0) ) ) )  * (vars.Dx.rowwise().norm().array().pow(2))(j);
                        if(my_rank == MASTER)
                            std::cout<<"A: "<<A<<"; B: "<<B<<"; C: "<<C<<std::endl; 
                        if (pow(B,2) - 4 * A * C < 0) {
                            //std::cout<<"Quadratic less than 0"<<std::endl;
                            value_vars.kappa(j) = 1;
                            vars.q(j) = ( (1.0 - value_vars.kappa(j) ) * a_h + value_vars.kappa(j) * a_e + 1/phi ) / ( (1.0 -state_vars.omega(j)) * pow(rho_h, 1/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h) + state_vars.omega(j) * pow(rho_e,1/psi_e) * pow(exp(value_vars.zeta_e(j)), 1 - 1/psi_e) + 1/phi );
                        } else if (pow(B,2) - 4 * A * C >= 0) {
                            //std::cout<<"Quadratic greater than 0"<<std::endl;
                            derivsQwrtOmega1 = ( 1.0 - 1.0/(2*A) *  (-B + sqrt((pow(B,2) - 4 * A * C) ) )  ) * 1.0 / ( chiUnderline*value_vars.kappa(j) - state_vars.omega(j) );
                            derivsQwrtOmega2 = ( 1.0 - 1.0/(2*A) *  (-B - sqrt((pow(B,2) - 4 * A * C) ) )  ) * 1.0 / ( chiUnderline*value_vars.kappa(j) - state_vars.omega(j) );
			                derivsQwrtOmega = derivsQwrtOmega1; //max(derivsQwrtOmega1, derivsQwrtOmega2);
                           
                            if(my_rank == MASTER)
                                std::cout<<"Derivs: "<<derivsQwrtOmega<<std::endl;
                            
                            vars.q(j + state_vars.increVec(0)) = vars.q(j  ) * (1 + derivsQwrtOmega * state_vars.dVec(0) );
                            if(my_rank == MASTER)
                                std::cout<<"q: "<<vars.q(j)<<std::endl;

                        }
                        
                                        
                    } else if ( state_vars.stateMat(j,0) != state_vars.lowerLims(0) && value_vars.kappa(j - state_vars.increVec(0)) >= 1) {
                        //std::cout<<"not omega_min; previous kappa greater than 1 stateMat: "<<state_vars.stateMat.row(j)<<std::endl;
                        //If kappa is greater than 1, set it to 1.
                        value_vars.kappa(j) = 1.0;
                        vars.q(j) = ( (1.0 - value_vars.kappa(j) ) * a_h + value_vars.kappa(j) * a_e + 1/phi ) / ( (1.0 -state_vars.omega(j)) * pow(rho_h, 1/psi_h) * pow(exp(value_vars.zeta_h(j)), 1 - 1/psi_h) + state_vars.omega(j) * pow(rho_e,1/psi_e) * pow(exp(value_vars.zeta_e(j)), 1 - 1/psi_e) + 1/phi );
                    }
                }

            } else {
                
                //kappa is 1 always
                value_vars.kappa = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
                //compute q based on formula (43)
                vars.q = ( (1.0 - value_vars.kappa) * a_h + value_vars.kappa * a_e + 1/phi ) / ( (1.0 -state_vars.omega) * pow(rho_h, 1/psi_h) * (value_vars.zeta_h.exp()).pow(1-1/psi_h) + state_vars.omega * pow(rho_e,1/psi_e) * value_vars.zeta_e.exp().pow(1 - 1/psi_e) + 1/phi );
                
                vars.oneOmegaQ = (1 - state_vars.omega) * vars.q;
                vars.omegaQ = state_vars.omega * vars.q;
                vars.logQ = vars.q.log();
                vars.iota = vars.logQ / phi;
                
                if(my_rank == MASTER)
                    std::cout<<"\nFinished updating q"<<std::endl;
            }

            
            /*********************************************************/
            /* Step 2(a - d): Update chi */
            /*********************************************************/
            
            //update derivs, vols, and risk prices
            vars.updateDerivs(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
            if(my_rank == MASTER)
                std::cout<<"\nFinihsed updating derivs"<<std::endl;
            vars.updateSigmaPi(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
            if(my_rank == MASTER)
                std::cout<<"\nFinihsed updating sigma and Pi"<<std::endl;

            if (state_vars.N == 4 && equityIss == 2) {
                /* In this case, equity issuance is allowed and skin-in-the-game constraint occasionally binds */
                /* With 4 state variables, need to iterate to solve a nonlinear equation */
                
                /* For iteration = 0, chi = omega */
                value_vars.chi = state_vars.omega;
                
                int k = 0;
                //TODO: empty while loop
                while (k == 0 || innerError > innerTol) {
                    
                }
            } else if (state_vars.N < 4 && equityIss == 2) {
                /* In this case, equity issuance is allowed and skin-in-the-game constraint occasionally binds */
                /* With less than 4 state variables, need to solve a linear equation */


                //update Dx
                if (state_vars.N == 1) {
                    vars.Dx.col(0) = vars.sigmaK.col(0).array();
                } else if (state_vars.N == 2) {
                    for (int n = 0; n < state_vars.N; ++n ) {
                        vars.Dx.col(n) = vars.sigmaK.col(n).array() + vars.sigmaXVec[1].col(n).array() * derivs3.firstPartials.col(1).array();
                    }
                    
                } else if (state_vars.N == 3){
                    for (int n = 0; n < state_vars.N; ++n ) {
                        vars.Dx.col(n) = vars.sigmaK.col(n).array() + vars.sigmaXVec[1].col(n).array() * derivs3.firstPartials.col(1).array() + vars.sigmaXVec[2].col(n).array() * derivs3.firstPartials.col(2).array();
                        
                    }
                    
                }
                
                //update D_zeta_omega
                vars.DzetaOmega = vars.Dx.rowwise().norm().array().pow(2) * ( (gamma_h - 1.0) * derivs2.firstPartials.col(0).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(0).array() );
                
                //update d_zeta_xtilde
                if (state_vars.N == 1) {
                    vars.DzetaX = vars.Dx.col(0).array() * 0.0;
                } else if (state_vars.N == 2){
                    vars.DzetaX = vars.Dx.col(0).array() * ( vars.sigmaXVec[1].col(0).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) ) + vars.Dx.col(1).array() * ( vars.sigmaXVec[1].col(1).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) ) ;

                } else if (state_vars.N == 3){
                    vars.DzetaX = vars.Dx.col(0).array() * ( vars.sigmaXVec[1].col(0).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(0).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() ) ) + vars.Dx.col(1).array() * ( vars.sigmaXVec[1].col(1).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(1).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() ) ) + vars.Dx.col(2).array() * ( vars.sigmaXVec[1].col(2).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(1).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(1).array() ) + vars.sigmaXVec[2].col(2).array() * ( (gamma_h - 1.0) * derivs2.firstPartials.col(2).array() - (gamma_e - 1.0) * derivs1.firstPartials.col(2).array() ) );

                    
                }
                
                
                

                
                //compute chi using linear formula
		if (i % 1 == 0) {
            if(my_rank == MASTER)
			    std::cout<<"updating chi at iteration i = "<<i<<std::endl;
            value_vars.chi = (vars.DzetaX - state_vars.omega * (1 - state_vars.omega) * (gamma_e - gamma_h) * vars.Dx.rowwise().norm().array().pow(2) + state_vars.omega * (  ((1 - state_vars.omega) * gamma_e + state_vars.omega * gamma_h ) * vars.Dx.rowwise().norm().array().pow(2) + derivs3.firstPartials.col(0).array() * vars.DzetaX - vars.DzetaOmega) ) / (  ((1 - state_vars.omega) * gamma_e + state_vars.omega * gamma_h ) * vars.Dx.rowwise().norm().array().pow(2) + derivs3.firstPartials.col(0).array() * vars.DzetaX - vars.DzetaOmega );
            value_vars.chi = (value_vars.chi < chiUnderline).cast<double>() * chiUnderline + (value_vars.chi >= chiUnderline).cast<double>() * value_vars.chi;
            value_vars.chi = (value_vars.chi > 1.0).cast<double>() * 1.0 + (value_vars.chi <= 1.0).cast<double>() * value_vars.chi;

		if (state_vars.N == 1 && gamma_e >= 20 ) {
			int omegaStarInd = 0;
			Eigen::MatrixXf::Index index;
			abs( (1.0 - state_vars.omega ) * gamma_e + state_vars.omega * gamma_h -  ( (gamma_h - 1) * derivs2.firstPartials.col(0).array() - (gamma_e - 1) * derivs1.firstPartials.col(0).array() ) ).minCoeff(&index);
			int omegaDoubleStar = 0;
			double chiMax = 1.0;

                        for (int omegaInd = index; omegaInd < index + 0.1 * nOmega; omegaInd++ ) {
                            if ( value_vars.chi(omegaInd + 1) <= (chiMax + 0.000001) ) {
				chiMax = value_vars.chi(omegaInd + 1);
				omegaDoubleStar = omegaInd + 1;
				
			    }
                        }
            if(my_rank == MASTER){
			    std::cout<<"omegaDoubleStar: "<<omegaDoubleStar<<std::endl; 
			    std::cout<<"chi selected segment: "<<value_vars.chi.segment(omegaDoubleStar - 3, 10)<<std::endl;
            }
                        for (int omegaInd = 0; omegaInd < omegaDoubleStar; omegaInd++ ) {
				value_vars.chi(omegaInd) = chiUnderline;
                        }

		} else if( state_vars.N == 1) {
                        int omegaStarInd = 0;
                        for (int omegaInd = 0; omegaInd < nOmega; omegaInd++ ) {
                            if (value_vars.chi(omegaInd) <= (chiUnderline + 0.000000001) ) {
                                omegaStarInd = omegaInd;
                            }
                        }
                        for (int omegaInd = 0; omegaInd < omegaStarInd; omegaInd++ ) {
                            value_vars.chi(omegaInd) = chiUnderline;
                        }

		}				
                if (state_vars.N == 10) {
                        int omegaStarInd = 0;
                        for (int omegaInd = 0; omegaInd < nOmega; omegaInd++ ) {
                            if (value_vars.chi(omegaInd) <= (chiUnderline + 0.000000001) ) {
                                omegaStarInd = omegaInd;
                            }
                        }
                        for (int omegaInd = 0; omegaInd <= omegaStarInd; omegaInd++ ) {
                            value_vars.chi(omegaInd) = chiUnderline;
                        }
                        std::cout<<"chi: "<<value_vars.chi.tail(5)<<std::endl;
			/*
                        int omegaDoubleStar = 0;
	
                        for (int omegaInd = nOmega - 3; omegaInd >= 0; omegaInd = omegaInd - 1) {
                            if (value_vars.chi(omegaInd) > .999 && value_vars.chi(omegaInd + state_vars.increVec(0)) > .99999 && value_vars.chi(omegaInd + 2*state_vars.increVec(0)) > .99999  ) {
                                omegaDoubleStar = omegaInd;
                            }

                        for (int omegaInd = omegaDoubleStar ; omegaInd < nOmega; omegaInd++ ) {
                            value_vars.chi(omegaInd) = 1.0;
                        

                        } */
			
                }  
		
                }

		/*
                //Impose lower bound
		if (i % 1000 != 0 ||  i < 2) {
		    std::cout<<"iteration i = "<<i<<std::endl;
                    for (int chiIter = 0; chiIter < state_vars.S; chiIter++) {
                        value_vars.chi(chiIter) = max(chiUnderline, state_vars.omega(chiIter));
                    }

		} 
 		*/
		/*
                for (int chiIter = 0; chiIter < state_vars.S; chiIter++) {
                	value_vars.chi(chiIter) = max(chiUnderline, state_vars.omega(chiIter));
                }
		*/

                value_vars.chi = (value_vars.chi < chiUnderline).cast<double>() * chiUnderline + (value_vars.chi >= chiUnderline).cast<double>() * value_vars.chi;
                value_vars.chi = (value_vars.chi > 1.0).cast<double>() * 1.0 + (value_vars.chi <= 1.0).cast<double>() * value_vars.chi;

                /*
                for (int chiIter = 0; chiIter < state_vars.S; chiIter++) {
                    value_vars.chi(chiIter) = max(chiUnderline, state_vars.omega(chiIter));
                }*/

                if (state_vars.N == 2) { 
                    for (int gInd = 0; gInd < nG; gInd++) {
                        int omegaStarInd = 0;
                        for (int omegaInd = 0; omegaInd < nOmega; omegaInd++ ) {
                            if (value_vars.chi(gInd * nOmega + omegaInd) <= (chiUnderline + 0.000000001) ) {
                                omegaStarInd = omegaInd; 
                            }
                        }
                        for (int omegaInd = 0; omegaInd < omegaStarInd; omegaInd++ ) {
                            value_vars.chi(gInd * nOmega + omegaInd) = chiUnderline;
                        }
                        
                    }
                }
                
                if (state_vars.N == 3) {
                    for (int sInd = 0; sInd < nS; sInd++) {
                        for (int gInd = 0; gInd <nG; gInd++) {
                            int omegaStarInd = 0;
                            for (int omegaInd = 0; omegaInd < nOmega; omegaInd++ ) {
                                if (value_vars.chi(nOmega * nG * sInd + gInd * nOmega + omegaInd) <= (chiUnderline + 0.000000001) ) {
                                    omegaStarInd = omegaInd;
                                }
                            }
                            for (int omegaInd = 0; omegaInd < omegaStarInd; omegaInd++ ) {
                                value_vars.chi(nOmega * nG * sInd + gInd * nOmega + omegaInd) = chiUnderline;
                            }
                            
                        }

                        
                    }
                }
                
                /*
                for (int chiIter = 0; chiIter < state_vars.S; chiIter++) {
                    
                    if (state_vars.N == 3) {
                        //Look for where constraint binds
                        thresSum = 0.0;
                        for (int t = 0; t < state_vars.S; t = t + nOmega) {
                            for (int s = t; s < s + nOmega; s++) {
                            if (value_vars.chi(s) > chiUnderline) {
                                thresSum = thresSum + state_vars.omega(s);
                                break;
                            }
                        }
                        
                    }
                    chiThres(i) = thresSum / ( (double)(nG * nS) );;
                    std::cout<<"chi threshold: "<<chiThres(i)<<std::endl;
                    
                }*/
    
            } else if (equityIss == 1) {
                /* In this case, equity issuance is allowed and skin-in-the-game constraint always binds */
                /* chi is forced to be chiUnderline < 1 */
                if(my_rank == MASTER)
                    std::cout<<"\nSetting chi"<<std::endl;
                value_vars.chi = Eigen::MatrixXd::Constant(state_vars.S, 1, chiUnderline);
            } else if (equityIss == 0) {
                /* In this case, equity issuance is not allowed  */
                /* chi is forced to be 1 */
                value_vars.chi = Eigen::MatrixXd::Constant(state_vars.S, 1, 1.0);
            }

        if(my_rank == MASTER){
	        std::cout<<"last 5 values of chi: "<<value_vars.chi.tail(5)<<std::endl;

            std::cout<<"\nFinished step 2"<<std::endl;
        }
            /*********************************************************/
            /* Step 3(a - d): Update regimes */
            /*********************************************************/

            
            //update sigma and Pi
            vars.updateDerivs(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
            vars.updateSigmaPi(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);

            //update deltaE and deltaH
            if ( (value_vars.kappa < 1.0).any() ) {
                
            } else {
                if (state_vars.N == 1) {
                    vars.deltaE = ( ( value_vars.chi * value_vars.kappa * (gamma_e * vars.normR2 + state_vars.varSig) ) / state_vars.omega - (vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() + (1.0 - gamma_e) * ( vars.sigmaR.col(0).array() * (vars.sigmaXVec[0].col(0).array() * derivs1.firstPartials.col(0).array() ) ) ) );
                    
                    if(my_rank == MASTER)
                        std::cout<<"\nFinished deltaE"<<std::endl;
                } else if (state_vars.N == 2) {
                    vars.deltaE = ( ( value_vars.chi * value_vars.kappa * (gamma_e * vars.normR2 + state_vars.varSig) ) / state_vars.omega - (vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() + (1.0 - gamma_e) * ( vars.sigmaR.col(0).array() * (vars.sigmaXVec[0].col(0).array() * derivs1.firstPartials.col(0).array() + vars.sigmaXVec[1].col(0).array() * derivs1.firstPartials.col(1).array() ) + vars.sigmaR.col(1).array() * (vars.sigmaXVec[0].col(1).array() * derivs1.firstPartials.col(0).array() + vars.sigmaXVec[1].col(1).array() * derivs1.firstPartials.col(1).array()  ) ) ) );
                    
                    
                } else if (state_vars.N == 3) {
                    vars.deltaE = ( ( value_vars.chi * value_vars.kappa * (gamma_e * vars.normR2 + state_vars.varSig) ) / state_vars.omega - (vars.Pi.cwiseProduct(vars.sigmaR).rowwise().sum().array() + (1.0 - gamma_e) * ( vars.sigmaR.col(0).array() * (vars.sigmaXVec[0].col(0).array() * derivs1.firstPartials.col(0).array() + vars.sigmaXVec[1].col(0).array() * derivs1.firstPartials.col(1).array() + vars.sigmaXVec[2].col(0).array() * derivs1.firstPartials.col(2).array() ) + vars.sigmaR.col(1).array() * (vars.sigmaXVec[0].col(1).array() * derivs1.firstPartials.col(0).array() + vars.sigmaXVec[1].col(1).array() * derivs1.firstPartials.col(1).array() + vars.sigmaXVec[2].col(1).array() * derivs1.firstPartials.col(2).array() ) + vars.sigmaR.col(2).array() * (vars.sigmaXVec[0].col(2).array() * derivs1.firstPartials.col(0).array() + vars.sigmaXVec[1].col(2).array() * derivs1.firstPartials.col(1).array() + vars.sigmaXVec[2].col(2).array() * derivs1.firstPartials.col(2).array() )) ) );
                    
                    
                }
                
                vars.deltaH = vars.deltaE * chiUnderline - (a_e - a_h) / vars.q;
            }
            
            //update betaE*deltaE and betaH * deltaH
            
            vars.betaEDeltaE = value_vars.chi * value_vars.kappa * vars.deltaE / state_vars.omega;
            vars.betaHDeltaH = (1.0 - value_vars.kappa) * vars.deltaH / (1.0 - state_vars.omega);
            
            //update mus
            if(my_rank == MASTER)
                std::cout<<"\nBeginning to update mu and R"<<std::endl;
            vars.updateMuAndR(state_vars, value_vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
            
            
            /*********************************************************/
            /* Step 5: Solving PDEs for households and experts */
            /*********************************************************/
            if(my_rank == MASTER)
                std::cout<<"\nFinished updating mu and R"<<std::endl;
            matrix_vars.updateMatrixVars(state_vars, value_vars, vars, derivs1, derivs2, derivs3, derivs4, derivs5, derivs6);
            if(my_rank == MASTER)
                std::cout<<"\nFinished updating matrix vars"<<std::endl;
            matrix_vars.updateMatrix(state_vars);
            if(my_rank == MASTER)
                std::cout<<"\nFinished updating matrix"<<std::endl;
            matrix_vars.updateKnowns(value_vars, state_vars);
            nnzE = matrix_vars.Le.nonZeros();
            nnzH = matrix_vars.Lh.nonZeros();
           
            #ifdef USE_MPI 
                if(my_rank == MASTER){
                    phase = 13;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &matrix_vars.a_e[0], &matrix_vars.ia_e[0], &matrix_vars.ja_e[0], &idum, &nrhs, iparm, &msglvl, &matrix_vars.Ue(0), &value_vars.zeta_e(0), &error,  dparm);
            
                    phase = -1;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &matrix_vars.ia_e[0], &matrix_vars.ja_e[0], &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error,  dparm);
                    MPI_Recv(&value_vars.zeta_h(0),value_vars.zeta_h.size(),MPI_DOUBLE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    MPI_Send(&value_vars.zeta_e[0],value_vars.zeta_e.size(),MPI_DOUBLE,1,0,MPI_COMM_WORLD);
                    std::cout<<"\n solved linear systems..."<<std::endl;
                } else {        
                    phase = 13;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &matrix_vars.a_h[0], &matrix_vars.ia_h[0], &matrix_vars.ja_h[0], &idum, &nrhs, iparm, &msglvl, &matrix_vars.Uh(0), &value_vars.zeta_h(0), &error,  dparm);
            
                    phase = -1;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &matrix_vars.ia_h[0], &matrix_vars.ja_h[0], &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error,  dparm);
                    MPI_Send(&value_vars.zeta_h[0],value_vars.zeta_h.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD);
                    MPI_Recv(&value_vars.zeta_e[0],value_vars.zeta_e.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
            #else
                    phase = 13;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &matrix_vars.a_e[0], &matrix_vars.ia_e[0], &matrix_vars.ja_e[0], &idum, &nrhs, iparm, &msglvl, &matrix_vars.Ue(0), &value_vars.zeta_e(0), &error,  dparm);
                
                    phase = -1;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &matrix_vars.ia_e[0], &matrix_vars.ja_e[0], &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error,  dparm);

                    phase = 13;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &matrix_vars.a_h[0], &matrix_vars.ia_h[0], &matrix_vars.ja_h[0], &idum, &nrhs, iparm, &msglvl, &matrix_vars.Uh(0), &value_vars.zeta_h(0), &error,  dparm);

                    phase = -1;
                    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &matrix_vars.ia_h[0], &matrix_vars.ja_h[0], &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error,  dparm);

                    std::cout<<"\n solved linear systems..."<<std::endl;
            #endif

        } else {
            if(my_rank == MASTER){
                std::cout<<"Error for zeta_e: "<<error_terms.eErrors(i)<<"\n";
                std::cout<<"Error for zeta_h: "<<error_terms.hErrors(i)<<"\n";
                std::cout<<"Tolerance met at iteration "<<i<<"; releasing data... \n";
            }
            /* Exporting data */
            if(my_rank == MASTER){
                exportData(value_vars, vars, derivs1, derivs2, i, state_vars.N);
                writeArray(allErrorE,folderName + run + "_"   + "allErrorE_" + std::to_string(i));
                writeArray(allErrorH,folderName + run + "_"   + "allErrorH_" + std::to_string(i));            
            }
            return 0;
        }
        
        

	if ( (i % 1000 == 0) && my_rank == MASTER) {
	    std::cout<<"iteration i = "<<i<<std::endl; std::cout<<"last 5 values of chi: "<<value_vars.chi.tail(5)<<std::endl;
        exportData(value_vars, vars, derivs1, derivs2, i, state_vars.N);
	    writeArray(allErrorE,folderName + run + "_"   + "allErrorE_" + std::to_string(i));
        writeArray(allErrorH,folderName + run + "_"   + "allErrorH_" + std::to_string(i));
    }
}

    if(my_rank == MASTER){
        std::cout<<"Error for zeta_e: "<<error_terms.eErrors(i)<<"\n";
        std::cout<<"Error for zeta_h: "<<error_terms.hErrors(i)<<"\n";
        std::cout<<"Tolerance not met and max iterations reached: "<<i<<"; releasing data... \n";
        /* Exporting data */
        exportData(value_vars, vars, derivs1, derivs2, i, state_vars.N);
    }
    return 0;
    
}
#endif /* modules_h */
