#include <iomanip>
#include <iostream>
#include "Fem1D.h"
#include "Thermoelastic1D.h"
#include "Verification2D.h"
#include <cmath>
using std::cout;
using std::endl;

struct ChainErrors {
    double tip = 0.0;      // errore sullo spostamento della punta (barra con l'estremo destro libero)
    double stress = 0.0;   // errore massimo sulla tensione (barra con l'estremo destro bloccato)
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

// Le tabelle nel tempo della lastra 2D hanno una colonna in piu', la differenza. L'ordine non si ricava dal rapporto
// fra due errori vicini (come sopra): la mesh ha un errore suo, E_h, che non cambia al cambiare di dt (a n = 32 vale
// circa 4.5e-4) e sporcherebbe quel rapporto. Si sottraggono invece gli errori di due righe vicine (E_h si cancella) e
// l'ordine e' il rapporto fra due differenze vicine: e' la stessa idea degli ESEMPI 113 e 114 dei test.
// Una riga: passi, dt, errore, differenza dall'errore della riga prima, ordine. Quello che manca e' un trattino: la
// differenza vuole la riga prima (previous_error > 0), l'ordine vuole anche la differenza della riga prima
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

    
    return 0;
}