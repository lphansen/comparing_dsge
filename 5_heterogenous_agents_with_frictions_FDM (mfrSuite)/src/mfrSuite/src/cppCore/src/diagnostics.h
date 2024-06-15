//
//  diagnostics.h
//  metaModel
//
//  Created by Joseph Huang on 10/9/17.
//  Copyright (C) 2017 Joseph Huang. All rights reserved.
//

#include <fstream>
#include <iomanip>
#include <limits>

using namespace std;
//Function to output vector/matrix

template <class T>
void writeArray (T array, string varName) {
    ofstream myfile;
    myfile.open (varName + ".dat", ios::out | ios::trunc | ios::binary);
    
    for (int i = 0; i < array.size(); ++i) {
        myfile << array(i) << setprecision(12) <<"\n";
    }
    
    myfile.close();
    
}



///Deprecated; to be erased


//int* ptr = matrix_vars.Le.outerIndexPtr();
//double* val = matrix_vars.Le.valuePtr();
//std::cout<<"Max Pi: " <<vars.q.maxCoeff() << "\n";
//std::cout<<"Min Pi: " <<vars.q.minCoeff() << "\n";


//double maxVal = val[1] - val[0];
//double minVal = val[1] - val[0];

//for (int r = 0; r < 30; ++r) {
    //std::cout<<value_vars.zeta_e(r) <<"\n";
//}

//for (int r = 1; r < nnzE; ++r) {
//    if (val[r+1] - val[r] > maxVal) {
//        maxVal = val[r+1] - val[r];
//    }
//}
//for (int r = 1; r < nnzE; ++r) {
//    if (val[r+1] - val[r] < minVal) {
//        minVal = val[r+1] - val[r];
//    }
//}

//std::cout<<"Max val: "<<maxVal<<"\n";
//std::cout<<"Min val: "<<minVal<<"\n";
