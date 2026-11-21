#include <iomanip>
#include <iostream>
#include <string>
#include "Fem1D.h"
#include "Thermoelastic1D.h"
#include "Verification2D.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Thermoelastic2D.h"
#include <cmath>
#include <cstddef>
#include <vector>
using std::cout;
using std::endl;

struct ChainErrors {
    double tip = 0.0;      // error of the tip displacement (bar with the right end free)
    double stress = 0.0;   // maximum error of the stress (bar with the right end fixed)
};

ThermoMechanicalProblem sine_problem(std::size_t N, double dt, std::size_t steps, double theta,
                                     RightEnd right_end) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(1.0, N);
    p.T0.resize(p.x.size());
    for (std::size_t i = 0; i < p.x.size(); ++i) {
        p.T0[i] = std::sin(pi * p.x[i]);
    }
    p.rho_c = 1.0;
    p.k = 1.0;
    p.q = 0.0;
    p.T_left = 0.0;
    p.T_right = 0.0;
    p.dt = dt;
    p.n_steps = steps;
    p.theta = theta;
    p.E = 3.0;
    p.alpha = 2.0;
    p.T_ref = 0.0;
    p.right_end = right_end;
    return p;
}

void print_chain_row(std::size_t level, double step, const ChainErrors& errors, const ChainErrors& previous) {
    std::cout << "| " << std::setw(5) << level << " | " << std::scientific << std::setprecision(2) << step
              << " | " << std::setw(9) << errors.tip << " | ";
    if (previous.tip > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous.tip, errors.tip, 2.0);
    } else {
        std::cout << std::setw(5) << "-";
    }
    std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(12) << errors.stress << " |\n";
}

void print_chain_header(const char* level_name, const char* step_name) {
    std::cout << "| " << std::setw(5) << level_name << " | " << std::setw(8) << step_name
              << " | tip error | order | stress error |\n";
    std::cout << "|------:|---------:|----------:|------:|-------------:|\n";
}

// The time tables of the 2D slab have one more column, the difference. The order cannot be obtained from the ratio
// of two neighbouring errors (as above): the mesh has an error of its own, E_h, that does not change when dt changes
// (at n = 32 it is about 4.5e-4) and would spoil that ratio. Instead, the errors of two neighbouring rows are subtracted
// (E_h cancels out) and the order is the ratio of two neighbouring differences: it is the same idea as EXAMPLES 113
// and 114 of the tests.
// One row: steps, dt, error, difference from the error of the previous row, order. What is missing is a dash: the
// difference needs the previous row (previous_error > 0), the order needs the difference of the previous row too
// (previous_difference > 0).
void print_difference_row(std::size_t steps, double dt, double error, double previous_error,
                          double previous_difference) {
    std::cout << "| " << std::setw(5) << steps << " | " << std::scientific << std::setprecision(2) << dt
              << " | " << error << " | ";
    if (previous_error > 0.0) {
        const double difference = previous_error - error;
        std::cout << std::setw(10) << difference << " | ";
        if (previous_difference > 0.0) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(5)
                      << observed_order(previous_difference, difference, 2.0);
        } else {
            std::cout << std::setw(5) << "-";
        }
    } else {
        std::cout << std::setw(10) << "-" << " | " << std::setw(5) << "-";
    }
    std::cout << " |\n";
}

void print_difference_header(const char* level_name, const char* step_name) {
    std::cout << "| " << std::setw(5) << level_name << " | " << std::setw(8) << step_name
              << " | L2 error | difference | order |\n";
    std::cout << "|------:|---------:|---------:|-----------:|------:|\n";
}

ChainErrors chain_errors(std::size_t N, double dt, std::size_t steps, double theta) {

    ChainErrors errors;
    const double pi = std::acos(-1.0);
    double t = dt * steps;

    ThermoMechanicalProblem problem_free = sine_problem(N, dt, steps, theta, RightEnd::Free);
    ThermoMechanicalResult result_free=  solve_thermomechanical(problem_free);
    double final_u = result_free.u.back();
    errors.tip = std::fabs(final_u - 2*problem_free.alpha*(std::exp(-pi*pi*t))/pi);

    ThermoMechanicalProblem problem_fixed = sine_problem(N, dt, steps, theta, RightEnd::Fixed);
    ThermoMechanicalResult result_fixed =  solve_thermomechanical(problem_fixed);
    double exact_solution_fixed = -2*problem_fixed.E*problem_fixed.alpha*(std::exp(-pi*pi*t))/pi;
    double error = 0.0;
    for(const auto &i:result_fixed.sigma){
        if(std::fabs(i - exact_solution_fixed) > error){
            error = std::fabs(i - exact_solution_fixed);}
}
    errors.stress = error;

    return errors;
}

// The steel beam of the tests: 4 m by 1 m, E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, gradient of 100 K per
// metre, reference temperature 293.15 K.
BendingBeam steel_beam() {
    BendingBeam beam;
    beam.length = 4.0;
    beam.height = 1.0;
    beam.E = 210.0e9;
    beam.nu = 0.3;
    beam.alpha = 12.0e-6;
    beam.gradient = 100.0;
    beam.T_ref = 293.15;
    return beam;
}

// The observed order between the error of the previous row (previous) and that of this row (current), with 2 decimals.
// In the first row there is no previous row (previous = 0) and a dash is printed.
void print_order(double previous, double current) {
    if (previous > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous, current, 2.0);
    } else {
        std::cout << std::setw(5) << "-";
    }
}

// One row of the beam table: k cells in height (the mesh is 4 k x k), the two errors and the two orders.
void print_beam_row(std::size_t k, const BendingBeamErrors& errors, const BendingBeamErrors& previous) {
    const std::string cells = std::to_string(4 * k) + " x " + std::to_string(k);
    std::cout << "| " << std::setw(5) << k << " | " << std::setw(8) << cells << " | " << std::scientific
              << std::setprecision(2) << std::setw(18) << errors.displacement << " | ";
    print_order(previous.displacement, errors.displacement);
    std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(12) << errors.stress << " | ";
    print_order(previous.stress, errors.stress);
    std::cout << " |\n";
}

// ---- The steel slab that cools: the von Mises stress at mid-thickness ----------------------------------------------

// The value of the mesh and the textbook one, in Pa, at the same point of the slab.
struct SlabVonMises {
    double fe = 0.0;      // the von Mises stress of the mesh: mean of the two triangles of the cell
    double exact = 0.0;   // the textbook formula at the centre of that cell
};

// The slab of the tests (EXAMPLE 237): steel (E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, rho c = 7850 * 470,
// k = 45), thickness 0.1 m, 4 times as long as it is thick, both faces at 293.15 K, rollers (u = 0) at the two ends.
// Initial temperature 293.15 + 100 sin(pi y / H) K: the heat is a decaying sine, and at time t it is
//     T - T_ref = 100 sin(pi y / H) exp(-t / tau),     tau = H^2 rho c / (pi^2 k) = 83.07 s.
// Far from the ends the slab deforms neither in x nor in z, and the von Mises stress is the textbook formula
//     E alpha (T - T_ref) / (1 - nu).
// Here we look at t = tau (100 Crank-Nicolson steps, dt = tau / 100), at mid-length, in the cell just above
// mid-thickness. k is the number of cells across the thickness: the mesh is 4 k x k (k even, so that there is a row
// of nodes exactly at mid-thickness).
SlabVonMises slab_von_mises(std::size_t k) {
    const double pi = std::acos(-1.0);
    const double height = 0.1;
    const double E = 210.0e9;
    const double nu = 0.3;
    const double alpha = 12.0e-6;
    const double rho_c = 7850.0 * 470.0;
    const double conductivity = 45.0;
    const double T_ref = 293.15;
    const double amplitude = 100.0;
    const double tau = height * height * rho_c / (pi * pi * conductivity);

    const std::size_t nx = 4 * k;
    const std::size_t ny = k;
    const Mesh2D mesh = rectangle_mesh(4.0 * height, height, nx, ny);

    ThermoMechanicalProblem2D problem;
    problem.f = DVector((nx + 1) * (ny + 1), 0.0);
    problem.T0 = DVector((nx + 1) * (ny + 1), T_ref);
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const double y_over_height = static_cast<double>(j) / static_cast<double>(ny);
            problem.T0[j * (nx + 1) + i] = T_ref + amplitude * std::sin(pi * y_over_height);
        }
    }
    for (std::size_t i = 0; i <= nx; ++i) {   // both faces stay at T_ref
        problem.fixed_nodes.push_back(i);
        problem.fixed_temperatures.push_back(T_ref);
        problem.fixed_nodes.push_back(ny * (nx + 1) + i);
        problem.fixed_temperatures.push_back(T_ref);
    }
    for (std::size_t j = 0; j <= ny; ++j) {   // rollers: u = 0 on the two sides x = 0 and x = L
        problem.fixed_dofs.push_back(2 * (j * (nx + 1)));
        problem.fixed_displacements.push_back(0.0);
        problem.fixed_dofs.push_back(2 * (j * (nx + 1) + nx));
        problem.fixed_displacements.push_back(0.0);
    }
    problem.fixed_dofs.push_back(1);          // v = 0 at node 0, against the rigid motion in y
    problem.fixed_displacements.push_back(0.0);
    problem.rho_c = rho_c;
    problem.k = conductivity;
    problem.dt = tau / 100.0;
    problem.n_steps = 100;
    problem.save_every = 100;
    problem.theta = 0.5;
    problem.E = E;
    problem.nu = nu;
    problem.alpha = alpha;
    problem.T_ref = T_ref;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    const ThermoMechanicalState2D& state = states.back();   // step 100, i.e. t = tau
    const std::vector<double> vm = von_mises_field_2d(mesh, state.sigma, E, nu, alpha, state.T, T_ref);

    // The cell in row k / 2 (from y = H / 2 to y = H / 2 + H / k), in column 2 k (at mid-length). The triangle
    // at the bottom right is number 2 * cell, the one at the top left number 2 * cell + 1.
    const std::size_t cell = (k / 2) * nx + nx / 2;
    const double y_center = (static_cast<double>(k / 2) + 0.5) * height / static_cast<double>(k);

    SlabVonMises result;
    result.fe = 0.5 * (vm[2 * cell] + vm[2 * cell + 1]);
    result.exact = E * alpha * amplitude * std::sin(pi * y_center / height) * std::exp(-1.0) / (1.0 - nu);
    return result;
}

// The table: for every k the von Mises stress of the mesh and the textbook one in megapascal, the relative error and
// the order (in the first row there is no previous row: a dash).
void print_slab_table() {
    std::cout << "\n### 2D slab: von Mises of a cooling steel slab at mid-height (4 k x k cells, Crank-Nicolson, "
                 "dt = tau / 100, t = tau = 83.07 s)\n\n";
    std::cout << "|     k |    cells | FE [MPa] | textbook [MPa] | relative error | order |\n";
    std::cout << "|------:|---------:|---------:|---------------:|---------------:|------:|\n";

    double previous_error = 0.0;
    for (std::size_t k : {2, 4, 8, 16, 32}) {
        const SlabVonMises value = slab_von_mises(k);
        const double error = std::fabs(value.fe - value.exact) / value.exact;
        const std::string cells = std::to_string(4 * k) + " x " + std::to_string(k);
        std::cout << "| " << std::setw(5) << k << " | " << std::setw(8) << cells << " | " << std::fixed
                  << std::setprecision(3) << std::setw(8) << value.fe / 1.0e6 << " | " << std::setw(14)
                  << value.exact / 1.0e6 << " | " << std::scientific << std::setprecision(2) << std::setw(14) << error
                  << " | ";
        if (previous_error > 0.0) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(5)
                      << observed_order(previous_error, error, 2.0);
        } else {
            std::cout << std::setw(5) << "-";
        }
        std::cout << " |\n";
        previous_error = error;
    }
}

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
    
    std::cout << "\n### Chain: spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)\n\n";
    print_chain_header("N", "h");
    ChainErrors chain_previous;
    for (std::size_t N : {10, 20, 40, 80, 160}) {
        const ChainErrors errors = chain_errors(N, 1e-4, 1000, 0.5);
        print_chain_row(N, 1.0 / N, errors, chain_previous);
        chain_previous = errors;
    }

    for (double theta : {1.0, 0.5}) {
        std::cout << "\n### Chain: temporal convergence (" << (theta == 1.0 ? "backward Euler" : "Crank-Nicolson")
                  << ", N = 800, t = 0.1)\n\n";
        print_chain_header("steps", "dt");
        chain_previous = ChainErrors();
        for (std::size_t steps : {5, 10, 20, 40}) {
            const double dt = 0.1 / steps;
            const ChainErrors errors = chain_errors(800, dt, steps, theta);
            print_chain_row(steps, dt, errors, chain_previous);
            chain_previous = errors;
        }
    }

        cout << endl;
    cout << "### 2D plate: manufactured solution sin(pi x) sin(pi y) (k = 1)" << endl << endl;
    cout << "|     n |        h | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    size_t cells[6] = {4, 8, 16, 32, 64, 128};
    previous_error = 0.0;
    for (const auto &i : cells) {
        double h = 1.0 / i;
        double error = manufactured_sine_error_2d(i, 1.0);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << h << " | " << error << " | ";
        if (i == 4) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }

    cout << endl;
    cout << "### 2D plate: decaying sine, spatial convergence (Crank-Nicolson, dt = 2.5e-4, t = 0.05)" << endl << endl;
    cout << "|     n |        h | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    size_t decay_cells[5] = {4, 8, 16, 32, 64};
    previous_error = 0.0;
    for (const auto &i : decay_cells) {
        double h = 1.0 / i;
        double error = decaying_sine_error_2d(i, 2.5e-4, 200, 0.5);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << h << " | " << error << " | ";
        if (i == 4) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }

    size_t plate_steps[5] = {4, 8, 16, 32, 64};
    for (double theta : {1.0, 0.5}) {
        cout << endl;
        cout << "### 2D plate: decaying sine, temporal convergence (" << (theta == 1.0 ? "backward Euler" : "Crank-Nicolson")
             << ", n = 32, t = 0.05)" << endl << endl;
        print_difference_header("steps", "dt");

        previous_error = 0.0;
        double previous_difference = 0.0;
        for (const auto &i : plate_steps) {
            double dt = 0.05 / i;
            double error = decaying_sine_error_2d(32, dt, i, theta);

            print_difference_row(i, dt, error, previous_error, previous_difference);
            if (previous_error > 0.0) {
                previous_difference = previous_error - error;
            }
            previous_error = error;
        }
    }

    cout << endl;
    cout << "### 2D beam: thermal bending of a steel beam (4 m x 1 m, G = 100 K/m, 4 k x k cells)" << endl << endl;
    cout << "|     k |    cells | displacement error | order | stress error | order |" << endl;
    cout << "|------:|---------:|-------------------:|------:|-------------:|------:|" << endl;

    BendingBeamErrors beam_previous;
    beam_previous.displacement = 0.0;
    beam_previous.stress = 0.0;
    for (std::size_t k : {1, 2, 4, 8, 16, 32}) {
        const BendingBeamErrors errors = bending_beam_errors_2d(steel_beam(), 4 * k, k);
        print_beam_row(k, errors, beam_previous);
        beam_previous = errors;
    }
    
    print_slab_table();

    return 0;
}
