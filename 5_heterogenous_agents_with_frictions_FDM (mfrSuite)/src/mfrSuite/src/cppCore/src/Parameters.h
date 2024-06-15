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


/*****************************************************/
/* This header file contains all the parameters used */
/* in the model solution. It should be included  */
/* everywhere in the program.*/
/*****************************************************/


#ifndef parameters_h
#define parameters_h

#include <string>
using std::string;
class Parameters{

public:
    /*****************************************************/
    /* Program Parameters                                */
    /*****************************************************/

    // Parameters for state variables

    bool useG;     // If true, the second state variable would be growth (g)
    bool useLogW;  // If true, model will be solved on log(w) instead of w
    int nDims;
    int nOmega;
    int nZ;
    int nV;
    int nH;
    int numSds;
    int nShocks;


    // Iteration parameters

    bool verbatim; // true (1): print out detail information on inner loops
    string run;
    string folderName;
    string preLoad; // if "zero", start iterations with the zero guess. If not, load model solution from previous solved models
    string method;  //can be {impl, expl, cg}
    double dt;
    double dtInner;
    int maxIters;
    int maxItersInner;
    double tol;
    double innerTol; /* Tolerance for kappa and chi */
    int equityIss;
    int hhCap;
    int exportFreq; /* export data every *exportFreq* iterations */
    double CGscale; /* scale CG tolerance; error tolerance for each time step would be tol * dt / 10 * CGscale */
    int precondFreq; // frequency at which preconditioner will be recomputed
    
    // Pardiso parameters

    int iparm_2;
    int iparm_3;
    int iparm_28;
    int iparm_31;


    /*****************************************************/
    /* Model Parameters                                  */
    /*****************************************************/

    // OLG parameters
    double nu_newborn;
    double lambda_d;

    // Persistence parameters
    double lambda_Z;
    double lambda_V;
    double lambda_H;

    // Means
    double H_bar;
    double Z_bar;
    double V_bar;

    // Rates of time preferences
    double delta_e;
    double delta_h;

    // Productivity parameters
    double a_e;
    double a_h;

    // Inverses of EIS
    double rho_e;
    double rho_h;

    // Adjustment cost and depreciation
    double phi;
    double alpha_K;

    // Risk aversion
    double gamma_e;
    double gamma_h;

    // Norm of vol
    double sigma_K_norm;
    double sigma_Z_norm;
    double sigma_V_norm;
    double sigma_H_norm;

    // Equity issuance constraint
    double chiUnderline;

    // Upper and lower boundaries of the state variables
    double omegaMin;  double omegaMax;
    double zMin;      double zMax;
    double vMin;      double vMax;
    double hMin;      double hMax;

    // Correlations

    double cov11 = 1.0; double cov12 = 0.0; double cov13 = 0.0; double cov14 = 0.0;
    double cov21 = 0.0; double cov22 = 1.0; double cov23 = 0.0; double cov24 = 0.0;
    double cov31 = 0.0; double cov32 = 0.0; double cov33 = 1.0; double cov34 = 0.0;
    double cov41 = 0.0; double cov42 = 0.0; double cov43 = 0.0; double cov44 = 1.0;

    Parameters (); // construct object
    void save_output(); // function to output parameters


};



#endif /* parameters_h */
