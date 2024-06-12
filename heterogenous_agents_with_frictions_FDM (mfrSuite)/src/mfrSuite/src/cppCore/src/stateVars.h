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
/* of the state variables, including w, g, s, varsigma    */
/**********************************************************/
/**********************************************************/


/*********************************************************/
/* Include header files                                  */
/*********************************************************/

#ifndef stateVars_h
#define stateVars_h
// must include files in this order
#include "common.h"
#include "Parameters.h"


class stateVars {

public:

    /***********************************************/
    /* Variables                                   */
    /***********************************************/

    Eigen::MatrixXd stateMat; //matrix to store state variables
    Eigen::ArrayXd increVec; //vector to record steps
    Eigen::ArrayXd dVec; //vector to record steps
    int N; // num of dimensions
    int S; // number of rows for the grid
    Eigen::ArrayXd upperLims;
    Eigen::ArrayXd lowerLims;



    Eigen::ArrayXd omega; Eigen::ArrayXd logW;
    Eigen::ArrayXd Z;
    Eigen::ArrayXd V;
    Eigen::ArrayXd H;
    
    Eigen::ArrayXd sqrtV;
    Eigen::ArrayXd sqrtH;

    Eigen::MatrixXd covMat;
    Eigen::VectorXd sigma_K;
    Eigen::VectorXd sigma_Z;
    Eigen::VectorXd sigma_V;
    Eigen::VectorXd sigma_H;

    /***********************************************/
    /* Maps                                        */
    /***********************************************/

    std::map <int, string> num2State; //maps from number to name of state variable

    // Maps that store boundary points, points next to the boundary point, and central points
    std::map <int, std::vector<int>> upperBdryPts;
    std::map <int, std::vector<int>> lowerBdryPts;
    std::map <int, std::vector<int>> adjUpperBdryPts;
    std::map <int, std::vector<int>> adjLowerBdryPts;
    std::map <int, std::vector<int>> centralPts;
    std::map <int, std::vector<int>> nonBdryPts;

    // Maps that store the counts
    std::map <int, int> upperBdryCt;
    std::map <int, int> lowerBdryCt;
    std::map <int, int> adjUpperBdryCt;
    std::map <int, int> adjLowerBdryCt;
    std::map <int, int> centralCt;
    std::map <int, int> nonBdryCt;

    //// Note that nonBdryPts = centralPts + adjUpperBdryPts + adjLowerBdryPts

    /***********************************************/
    /* Methods                                     */
    /***********************************************/
    stateVars ();
    stateVars (Eigen::ArrayXd upper, Eigen::ArrayXd lower, Eigen::ArrayXd gridSizes, Parameters & parameters); //constructors with arrays of upper/lower bounds and gridsizes
};



#endif /* stateVars_h */
