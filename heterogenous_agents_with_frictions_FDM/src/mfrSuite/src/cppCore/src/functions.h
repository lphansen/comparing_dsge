//
//  functions.hpp
//
//
//  Created by Joseph Huang on 8/6/18.
//
//

#ifndef functions_h
#define functions_h

#include <stdio.h>
#include "derivs.h"
#include "stateVars.h"
#include "valueVars.h"
#include "matrixVars.h"
#include "Vars.h"
#include "common.h"
#include "Parameters.h"

/********************************************************/
/* Export functions                                     */
/********************************************************/


void exportInformation(std::vector<double> & timeItersVec, std::vector<double> & timeItersLinSysVec,
                       std::vector<double> & eErrorsVec, std::vector<double> & hErrorsVec,
                       std::vector<int> & cgEIters, std::vector<int> & cgHIters, Parameters & parameters);

void computeFirstDerUpwind(stateVars & state_vars, valueVars & value_vars, matrixVars & matrix_vars, derivs & derivsXiE, derivs & derivsXiH);

void exportPDE(matrixVars & matrix_vars, valueVars & value_vars, derivs & derivsXiE, derivs & derivsXiH, Parameters & parameters, stateVars & state_vars, string suffix);

void exportData(valueVars & value_vars, Vars & vars,  derivs & derivsXiE, derivs & derivsXiH, derivs & derivsQ, derivs & derivsLogQ, string suffix, stateVars & state_vars, Parameters & parameters);


/********************************************************/
/* Main loop                                            */
/********************************************************/

int iterFunc(stateVars & state_vars, valueVars & value_vars, Vars & vars, matrixVars & matrix_vars, derivs & derivsXiE, derivs & derivsXiH, derivs & derivsLogQ, derivs & derivsQ, derivs & derivsKappa, derivs & derivsLogABar, Parameters & parameters, std::vector<double> & timeItersVec, std::vector<double> & timeItersLinSysVec,
             std::vector<double> & eErrorsVec, std::vector<double> & hErrorsVec,
             std::vector<int> & cgEIters, std::vector<int> & cgHIters);



#endif /* functions_h */
