#include <iomanip>
#include <iostream>
#include "Fem1D.h"

using std::cout;
using std::endl;

int main() {
    cout << "### Spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)" << endl << endl;
    cout << "|     N |        h | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    size_t mesh[5] = {10, 20, 40, 80, 160};
    double previous_error = 0.0;
    for (const auto &i : mesh) {
        double h = 1.0 / i;
        double error = decaying_sine_error(i, 1e-4, 1000, 0.5);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << h << " | " << error << " | ";
        if (i == 10) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }

    cout << endl;
    cout << "### Temporal convergence (backward Euler, N = 800, t = 0.1)" << endl;
    cout << "| steps |       dt | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;
    size_t temporal_steps[4] = {5, 10, 20, 40};
    previous_error = 0.0;
    for (const auto &i : temporal_steps) {
        double dt = 0.1 / i;
        double error = decaying_sine_error(800, dt, i, 1);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << dt << " | " << error << " | ";
        if (i == 5) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }
    
    cout << endl;
    cout << "### Temporal convergence (Crank-Nicolson, N = 800, t = 0.1)" << endl;
    cout << "| steps |       dt | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    previous_error = 0.0;
    for (const auto &i : temporal_steps) {
        double dt = 0.1 / i;
        double error = decaying_sine_error(800, dt, i, 0.5);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << dt << " | " << error << " | ";
        if (i == 5) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }

    return 0;
}