//
//  matrixVars.hpp
//
//
//  Created by Joseph Huang on 8/6/18.
//
//

#ifndef matrixVars_h
#define matrixVars_h

#include <stdio.h>
#include "derivs.h"
#include "stateVars.h"
#include "valueVars.h"
#include "Vars.h"
#include "common.h"
#include "Parameters.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> RowVector;

class matrixVars {

public:


    int k;
    Eigen::VectorXd Fe;
    Eigen::MatrixXd firstCoefsE; Eigen::MatrixXd secondCoefsE;
    Eigen::VectorXd Fh;
    Eigen::MatrixXd firstCoefsH; Eigen::MatrixXd secondCoefsH;
    Eigen::ArrayXd atBoundIndicators;

    Eigen::VectorXd Ue; Eigen::VectorXd Uh;

    Eigen::MatrixXd sigmaX_temp; Eigen::VectorXd derivs_temp; Eigen::MatrixXd idenMat; Eigen::MatrixXd tempResult;

    std::vector<T> eList; std::vector<T> hList;
    SpMat Le; SpMat Lh; SpMat I;
    SpMat LeNoTransp; SpMat LhNoTransp;
    SpMat LeNormed; SpMat LhNormed;

    std::vector<double> ae; std::vector<int> iae; std::vector<int> jae;
    std::vector<double> ah; std::vector<int> iah; std::vector<int> jah;

    double firstCoefE;
    double secondCoefE;
    double firstCoefH;
    double secondCoefH;

    // Objects related to CG

    int cgEIters;    int cgHIters;
    double cgErrorE; double cgErrorH;
    Eigen::VectorXd residual;
    Eigen::VectorXd x;

    double rhsNorm2, residualNorm2, threshold;
    Eigen::VectorXd p;
    Eigen::VectorXd z, tmp;
    double absNew, absOld, beta;
    double alpha;

    RowVector rowNorms;

    matrixVars();
    matrixVars(stateVars &, Parameters &);
    void updateMatrixVars(stateVars & state_vars, valueVars & value_vars, Vars & vars, derivs & derivsZetaE, derivs & derivsZetaH,
                          derivs & derivsLogQ, derivs & derivsQv, Parameters & parameters);
    void updateMatrix(stateVars & state_vars, Parameters & parameters);

    void updateKnowns(valueVars & value_vars, stateVars & state_vars, Parameters & parameters);

    void solveWithCGICholE(valueVars & value_vars, stateVars & state_vars, Eigen::IncompleteCholesky<double,
        Eigen::Lower, Eigen::NaturalOrdering<int>> & ichol, double tol, int maxIters);

    void solveWithCGICholH(valueVars & value_vars, stateVars & state_vars, Eigen::IncompleteCholesky<double,
        Eigen::Lower, Eigen::NaturalOrdering<int>> & ichol, double tol, int maxIters);

    void solveWithKacz(valueVars & value_vars, stateVars & state_vars,
            double tol, int maxIters);
};


#endif /* matrixVars_h */
