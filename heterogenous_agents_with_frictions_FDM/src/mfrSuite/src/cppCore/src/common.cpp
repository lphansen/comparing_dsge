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

#include "common.h"

// Compute (n choose k)
int choose(int n, int k) {
    if (k == 0) {
        return 1;
    } else {
        return ( n * choose(n - 1, k - 1) ) / k;
    }
}
