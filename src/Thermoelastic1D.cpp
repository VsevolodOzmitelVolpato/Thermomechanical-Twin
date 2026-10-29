#include <stdexcept>
#include "Thermoelastic1D.h"

DVector assemble_thermal_load(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref) {
    if ( T.size() != x.size()){
        throw std::invalid_argument("The two vectors have different sizes");}

    DVector f (x.size(), 0.0);

    for( size_t i {1}; i < x.size(); ++i){
        double dT_mean = ((T[i-1] - T_ref) + (T[i] - T_ref)) / 2;
        double force = E * alpha * dT_mean;
        f[i-1] -= force;
        f[i] += force;}

    return f;  
}