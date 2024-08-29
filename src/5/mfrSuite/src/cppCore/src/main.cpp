//
//  main.cpp
//  metaModel
//
//  Created by Joseph Huang on 10/9/17.
//  Copyright (C) 2017 Joseph Huang.
//

/*********************************************************/
/* Include header files                                  */
/*********************************************************/

// must include files in this order
#include "common.h"
#include "Parameters.h"
#include "functions.h"
#include "model.h"


/*****************************************************/
/* Boost libraries                                   */
/*****************************************************/
#include <boost/filesystem.hpp>
#include <boost/program_options/detail/config_file.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>


// Handling name spaces
using namespace std;
namespace po = boost::program_options;

/*********************************************************/
/* Start of program                                      */
/*********************************************************/


int main(int argc, const char * argv[]) {


    /*****************************************************/
    /* Read command line arguments                       */
    /*****************************************************/

    po::options_description description("Parameters");
    description.add_options()
        ("numSds", po::value<int>(), "numSds")
        ("rho_e", po::value<double>(), "rho_e")
        ("rho_h", po::value<double>(), "rho_h")
        ("gamma_e", po::value<double>(), "gamma_e")
        ("gamma_h", po::value<double>(), "gamma_h")
        ("delta_e", po::value<double>(), "delta_e")
        ("delta_h", po::value<double>(), "delta_h")
        ("lambda_d", po::value<double>(), "lambda_d")
        ("lambda_Z", po::value<double>(), "lambda_Z")
        ("lambda_V", po::value<double>(), "lambda_V")
        ("lambda_Vtilde", po::value<double>(), "lambda_Vtilde")
        ("Z_bar", po::value<double>(), "Z_bar")
        ("V_bar", po::value<double>(), "V_bar")
        ("Vtilde_bar", po::value<double>(), "Vtilde_bar")
        ("phi", po::value<double>(), "phi")
        ("sigma_K_norm", po::value<double>(), "sigma_K_norm")
        ("sigma_Z_norm", po::value<double>(), "sigma_Z_norm")
        ("sigma_V_norm", po::value<double>(), "sigma_V_norm")
        ("tol", po::value<double>(), "tol")
        ("innerTol", po::value<double>(), "innerTol")
        ("sigma_Vtilde_norm", po::value<double>(), "sigma_Vtilde_norm")
        ("nu_newborn", po::value<double>(), "nu_newborn")
        ("chiUnderline", po::value<double>(), "chiUnderline")
        ("equityIss", po::value<int>(), "equityIss")
        ("hhCap", po::value<int>(), "hhCap")
        ("dt", po::value<double>(), "dt")
        ("dtInner", po::value<double>(), "dtInner")
        ("nDims", po::value<int>(), "nDims")
        ("nShocks", po::value<int>(), "nShocks")
        ("wMin", po::value<double>(), "wMin")
        ("wMax", po::value<double>(), "wMax")
        ("nWealth", po::value<int>(), "nWealth")
        ("nZ", po::value<int>(), "nZ")
        ("nV", po::value<int>(), "nV")
        ("nVtilde", po::value<int>(), "nVtilde")
        ("method", po::value<int>(), "method")
        ("maxIters", po::value<int>(), "maxIters")
        ("maxItersInner", po::value<int>(), "maxItersInner")
        ("a_e", po::value<double>(), "a_e")
        ("a_h", po::value<double>(), "a_h")
        ("verbatim", po::value<int>(), "verbatim")
        ("logW", po::value<int>(), "logW")
        ("iparm_2", po::value<int>(), "iparm_2")
        ("iparm_3", po::value<int>(), "iparm_3")
        ("iparm_28", po::value<int>(), "iparm_28")
        ("iparm_31", po::value<int>(), "iparm_31")
        ("alpha_K", po::value<double>(), "alpha_K")
        ("folderName", po::value<std::string>(), "folderName")
        ("preLoad", po::value<std::string>(), "preLoad")
        ("cov11", po::value<double>(), "cov11")
        ("cov12", po::value<double>(), "cov12")
        ("cov13", po::value<double>(), "cov13")
        ("cov14", po::value<double>(), "cov14")
        ("cov21", po::value<double>(), "cov21")
        ("cov22", po::value<double>(), "cov22")
        ("cov23", po::value<double>(), "cov23")
        ("cov24", po::value<double>(), "cov24")
        ("cov31", po::value<double>(), "cov31")
        ("cov32", po::value<double>(), "cov32")
        ("cov33", po::value<double>(), "cov33")
        ("cov34", po::value<double>(), "cov34")
        ("cov41", po::value<double>(), "cov41")
        ("cov42", po::value<double>(), "cov42")
        ("cov43", po::value<double>(), "cov43")
        ("cov44", po::value<double>(), "cov44")
        ("exportFreq", po::value<int>(), "exportFreq")
        ("CGscale", po::value<double>(), "CGscale")
        ("precondFreq", po::value<int>(), "precondFreq")

    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, description), vm);
    po::notify(vm);

    /* Initialize model */
    int S = vm["nWealth"].as<int>();
    if (vm["nZ"].as<int>() > 1) {
            S = S * vm["nZ"].as<int>();
    }
    if (vm["nV"].as<int>() > 1) {
            S = S * vm["nV"].as<int>();
    }
    if (vm["nVtilde"].as<int>() > 1) {
            S = S * vm["nVtilde"].as<int>();
    }

    Eigen::ArrayXd chiGuessInput;      chiGuessInput   =  Eigen::MatrixXd::Constant(S, 1, -1.0);
    Eigen::ArrayXd kappaGuessInput;    kappaGuessInput =  Eigen::MatrixXd::Constant(S, 1, -1.0);
    Eigen::ArrayXd xiEGuessInput;      xiEGuessInput   =  Eigen::MatrixXd::Constant(S, 1, 0.0);
    Eigen::ArrayXd xiHGuessInput;      xiHGuessInput   =  Eigen::MatrixXd::Constant(S, 1, 0.0);

    std::cout<<"Parameters from command line found"<<std::endl;
    std::cout<<"Beginning to read parameters of the model"<<std::endl;
    double rho_e                 = vm["rho_e"].as<double>();
    double rho_h                 = vm["rho_h"].as<double>();
    double a_e                   = vm["a_e"].as<double>();
    double a_h                   = vm["a_h"].as<double>();
    double gamma_e               = vm["gamma_e"].as<double>();
    double gamma_h               = vm["gamma_h"].as<double>();
    double delta_e               = vm["delta_e"].as<double>();
    double delta_h               = vm["delta_h"].as<double>();
    double lambda_d              = vm["lambda_d"].as<double>();
    double nu_newborn            = vm["nu_newborn"].as<double>();
    double chiUnderline          = vm["chiUnderline"].as<double>();
    double phi                   = vm["phi"].as<double>();
    double alpha_K               = vm["alpha_K"].as<double>();

    double lambda_Z              = vm["lambda_Z"].as<double>();
    double lambda_V              = vm["lambda_V"].as<double>();
    double lambda_H              = vm["lambda_Vtilde"].as<double>(); // H and Vtilde interchangeable
    double Z_bar                 = vm["Z_bar"].as<double>();
    double V_bar                 = vm["V_bar"].as<double>();
    double H_bar                 = vm["Vtilde_bar"].as<double>(); // H and Vtilde interchangeable
    double sigma_K_norm          = vm["sigma_K_norm"].as<double>();
    double sigma_Z_norm          = vm["sigma_Z_norm"].as<double>();
    double sigma_V_norm          = vm["sigma_V_norm"].as<double>();
    double sigma_H_norm          = vm["sigma_Vtilde_norm"].as<double>(); // H and Vtilde interchangeable

    std::cout<<"Beginning to read parameters of the grid"<<std::endl;
    int numSds                   = vm["numSds"].as<int>();
    int logW                     = vm["logW"].as<int>();
    double wMin                  = vm["wMin"].as<double>();
    double wMax                  = vm["wMax"].as<double>();
    int nDims                    = vm["nDims"].as<int>();
    int nShocks                  = vm["nShocks"].as<int>();
    int nWealth                  = vm["nWealth"].as<int>();
    int nZ                       = vm["nZ"].as<int>();
    int nV                       = vm["nV"].as<int>();
    int nH                       = vm["nVtilde"].as<int>();      // Note that Vtilde and H are interchangeable
    int verbatim                 = vm["verbatim"].as<int>();

    std::cout<<"Beginning to read parameters of the numerical algorithm"<<std::endl;
    double dt                    = vm["dt"].as<double>();
    double dtInner               = vm["dtInner"].as<double>();
    double tol                   = vm["tol"].as<double>();
    double innerTol              = vm["innerTol"].as<double>();
    int maxIters                 = vm["maxIters"].as<int>();
    int maxItersInner            = vm["maxItersInner"].as<int>();
    int equityIss                = vm["equityIss"].as<int>();
    int hhCap                    = vm["hhCap"].as<int>();
    int exportFreq               = vm["exportFreq"].as<int>();
    int precondFreq              = vm["precondFreq"].as<int>();
    double CGscale               = vm["CGscale"].as<double>();
    int method                   = vm["method"].as<int>();
    string preLoad               = vm["preLoad"].as<std::string>();
    string folderName            = vm["folderName"].as<std::string>();

    std::cout<<"Beginning to read correlation parameters.."<<std::endl;
    double cov11                 = vm["cov11"].as<double>();
    double cov12                 = vm["cov12"].as<double>();
    double cov13                 = vm["cov13"].as<double>();
    double cov14                 = vm["cov14"].as<double>();
    double cov21                 = vm["cov21"].as<double>();
    double cov22                 = vm["cov22"].as<double>();
    double cov23                 = vm["cov23"].as<double>();
    double cov24                 = vm["cov24"].as<double>();
    double cov31                 = vm["cov31"].as<double>();
    double cov32                 = vm["cov32"].as<double>();
    double cov33                 = vm["cov33"].as<double>();
    double cov34                 = vm["cov34"].as<double>();
    double cov41                 = vm["cov41"].as<double>();
    double cov42                 = vm["cov42"].as<double>();
    double cov43                 = vm["cov43"].as<double>();
    double cov44                 = vm["cov44"].as<double>();

    int iparm_2                  = vm["iparm_2"].as<int>();
    int iparm_3                  = vm["iparm_3"].as<int>();
    int iparm_28                 = vm["iparm_28"].as<int>();
    int iparm_31                 = vm["iparm_31"].as<int>();

    std::cout<<"Finished reading parameters."<<std::endl;
    model model(numSds, sigma_K_norm, sigma_Z_norm, sigma_V_norm,
                 sigma_H_norm, logW, wMin, wMax,
                 nDims, nWealth, nZ, nV, nH, nShocks,
                 verbatim, folderName, preLoad, method,
                 dt, dtInner, maxIters, maxItersInner,
                 tol, innerTol, equityIss, hhCap, iparm_2,
                 iparm_3, iparm_28, iparm_31, lambda_d, nu_newborn,
                 lambda_Z, lambda_V, lambda_H, Z_bar,
                 V_bar, H_bar, delta_e, delta_h,
                 a_e, a_h, rho_e, rho_h, phi,
                 alpha_K, gamma_e, gamma_h, chiUnderline,
                 cov11, cov12, cov13, cov14,
                 cov21, cov22, cov23, cov24,
                 cov31, cov32, cov33, cov34,
                 cov41, cov42, cov43, cov44, exportFreq,
                 xiEGuessInput, xiHGuessInput, chiGuessInput,
                 kappaGuessInput, CGscale, precondFreq);
    std::cout<<"Finished creating model"<<std::endl;
    model.solveModel();

    return 0;
}
