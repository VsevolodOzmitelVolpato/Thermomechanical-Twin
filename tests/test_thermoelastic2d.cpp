#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Solver2D.h"
#include "Thermoelastic2D.h"

// ---- La catena completa: il calore che cambia nel tempo e le tensioni che ne nascono -------------------------------
//
// Fin qui ogni pezzo e' stato provato da solo: il calore nel tempo (solve_transient_2d), la forza termica, gli
// spostamenti, le tensioni. solve_thermomechanical_2d li mette in fila: per ogni istante che si vuole salvare calcola
// la temperatura T, poi la forza termica F(T), poi gli spostamenti u (K u = F con i vincoli), poi le tensioni sigma, e
// restituisce la lista di questi stati. L'accoppiamento e' a senso unico (il calore muove la lastra, la lastra non
// scalda niente) e quasi statico (a ogni istante la lastra e' in equilibrio con la temperatura di quell'istante).
//
// DUE FATTI CHE I TEST USANO.
//   1. La parte elastica e' LINEARE: con i vincoli a zero, u e sigma sono lineari nello scarto di temperatura T - T_ref
//      (la forza termica lo e', K u = F lo e', sigma = D (B u - eps_th) lo e'). Quindi se lo scarto di temperatura di
//      uno stato e' g volte quello di un altro, lo sono anche u e sigma.
//   2. Il calore ha soluzioni che si calcolano a mano. Con la lastra isolata e un carico uniforme q, K 1 = 0 (nessun
//      flusso se T e' uniforme) e la massa porta il carico dentro: T cresce di q / rho_c al secondo in tutti i nodi,
//      per ogni schema theta. Con una striscia di due righe di celle, le facce a T_ref e la riga di mezzo calda, la
//      riga di mezzo e' un'incognita sola e ogni passo la moltiplica per un fattore g che si calcola a mano (ESEMPIO
//      219).
//
// Cosa deve fare la funzione, ed e' quello che i test controllano:
//   - la lista ha n_steps / save_every + 1 stati, il primo al tempo 0 (con T = T0), gli altri ai tempi passi x dt; ogni
//     stato ha le misure giuste (ESEMPI 212 e 213); con zero passi non serve nessun dato del calore (ESEMPIO 213);
//   - ogni stato e' proprio la catena delle funzioni che gia' esistono, nell'ordine e con i dati giusti (ESEMPIO
//     214), e salvare ogni passo o ogni 3 passi da' gli stessi numeri (ESEMPIO 215);
//   - tre casi con la soluzione a mano: la lastra libera che si scalda in modo uniforme si dilata senza tensioni
//     (ESEMPIO 216), anche con un moto rigido imposto (ESEMPIO 217); la lastra incastrata, scaldata allo stesso
//     modo, e' compressa e la pressione cresce linearmente nel tempo (ESEMPIO 218);
//   - la striscia a rulli: il calore decade del fattore g fatto a mano (ESEMPIO 219) e con lui tutti gli spostamenti e
//     le tensioni (ESEMPIO 220); a meta' di una striscia lunga le tensioni sono quelle fatte a mano (ESEMPIO 221); il
//     decadimento ha l'ordine in tempo dello schema theta (ESEMPIO 222); una lastra d'acciaio che si raffredda ha la
//     tensione del libro, -E alpha dT / (1 - nu) (ESEMPIO 223);
//   - leggi di scala: E, alpha e l'ampiezza del calore (ESEMPIO 224), T_ref che non conta se si sposta tutto (ESEMPIO
//     225);
//   - dati sbagliati: quelli che controlla la funzione (ESEMPIO 226) e quelli delle funzioni che chiama (ESEMPIO 227).

namespace {

const double kPi = 3.14159265358979323846;

// Numeri comodi per i conti a mano: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20. Allora
//     lambda = E nu / ((1 + nu) (1 - 2 nu)) = 100 (1/4) / ((5/4) (1/2)) = 40,        mu = E / (2 (1 + nu)) = 40,
//     D = [[120, 40, 0], [40, 120, 0], [0, 0, 40]],        E alpha = 1,        (1 + nu) alpha = 1/80 = 0.0125.
void use_hand_material(ThermoMechanicalProblem2D& problem) {
    problem.E = 100.0;
    problem.nu = 0.25;
    problem.alpha = 0.01;
    problem.T_ref = 20.0;
}

// I tre vincoli di una griglia rettangolare nx x ... celle che tolgono i tre moti rigidi (come per la trave): u e v del
// nodo 0 (l'angolo (0, 0)) e v del nodo nx (l'angolo (L, 0)), tutti a zero.
void pin_corners(ThermoMechanicalProblem2D& problem, std::size_t nx) {
    problem.fixed_dofs = {0, 1, 2 * nx + 1};
    problem.fixed_displacements = {0.0, 0.0, 0.0};
}

// La lastra a rulli di nx x ny celle, rectangle_mesh(length, height, nx, ny), senza carico di calore (f = 0):
//   - le due facce, y = 0 (la riga di nodi 0, ..., nx) e y = H (l'ultima riga), restano alla temperatura t_ref;
//   - la temperatura iniziale e' t_ref + amplitude sin(pi j / ny) nella riga j di nodi, la stessa in tutte le colonne;
//   - i rulli: u = 0 nei nodi dei due lati x = 0 e x = L (le colonne 0 e nx di ogni riga), e v = 0 nel nodo 0 per
//     togliere lo spostamento in verticale. Nessun altro vincolo: le facce si dilatano liberamente.
// Il materiale, i numeri del calore e il tempo li sceglie chi chiama.
ThermoMechanicalProblem2D roller_slab_problem(std::size_t nx, std::size_t ny, double t_ref, double amplitude) {
    const std::size_t n = (nx + 1) * (ny + 1);
    ThermoMechanicalProblem2D problem;
    problem.f = DVector(n, 0.0);
    problem.T0 = DVector(n, t_ref);
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const double ratio = static_cast<double>(j) / static_cast<double>(ny);
            problem.T0[j * (nx + 1) + i] = t_ref + amplitude * std::sin(kPi * ratio);
        }
    }
    for (std::size_t i = 0; i <= nx; ++i) {
        problem.fixed_nodes.push_back(i);
        problem.fixed_temperatures.push_back(t_ref);
        problem.fixed_nodes.push_back(ny * (nx + 1) + i);
        problem.fixed_temperatures.push_back(t_ref);
    }
    for (std::size_t j = 0; j <= ny; ++j) {
        problem.fixed_dofs.push_back(2 * (j * (nx + 1)));
        problem.fixed_displacements.push_back(0.0);
        problem.fixed_dofs.push_back(2 * (j * (nx + 1) + nx));
        problem.fixed_displacements.push_back(0.0);
    }
    problem.fixed_dofs.push_back(1);
    problem.fixed_displacements.push_back(0.0);
    return problem;
}

// La striscia: due righe di celle quadrate (lato 1/2, altezza H = 1, lunghezza nx / 2), con i numeri comodi: ampiezza
// 60, rho_c = 2, k = 3 (quindi k / rho_c = 3/2) e il materiale a mano. Con dt = 1/36 ogni passo moltiplica la riga di
// mezzo per un fattore g che si calcola a mano (ESEMPIO 219).
Mesh2D strip_mesh(std::size_t nx) {
    return rectangle_mesh(0.5 * static_cast<double>(nx), 1.0, nx, 2);
}

ThermoMechanicalProblem2D strip_problem(std::size_t nx, double theta, double dt, std::size_t n_steps,
                                        std::size_t save_every) {
    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, 2, 20.0, 60.0);
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 3.0;
    problem.dt = dt;
    problem.theta = theta;
    problem.n_steps = n_steps;
    problem.save_every = save_every;
    return problem;
}

// Il problema "generale" delle prove di composizione e di scala: una mesh irregolare, e tutti i numeri diversi uno
// dall'altro (cosi' uno scambio fra due dati cambia il risultato).
Mesh2D general_mesh() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

ThermoMechanicalProblem2D general_problem(const Mesh2D& mesh) {
    ThermoMechanicalProblem2D problem;
    problem.rho_c = 2.5;
    problem.k = 0.7;
    problem.f = assemble_load_2d(mesh, 3.0);
    problem.fixed_nodes = {0, 3};
    problem.fixed_temperatures = {310.0, 280.0};
    problem.T0 = {310.0, 305.0, 295.0, 280.0, 290.0};
    problem.dt = 0.2;
    problem.n_steps = 6;
    problem.theta = 0.6;
    problem.save_every = 2;
    problem.E = 70.0;
    problem.nu = 0.2;
    problem.alpha = 0.003;
    problem.T_ref = 290.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};
    return problem;
}

// Il valore assoluto piu' grande di un vettore, e delle tensioni (tutte le componenti di tutti i triangoli).
double max_abs(const std::vector<double>& values) {
    double largest = 0.0;
    for (double value : values) {
        largest = std::max(largest, std::fabs(value));
    }
    return largest;
}

double max_abs(const std::vector<Vector3>& stresses) {
    double largest = 0.0;
    for (const Vector3& s : stresses) {
        for (std::size_t c = 0; c < 3; ++c) {
            largest = std::max(largest, std::fabs(s[c]));
        }
    }
    return largest;
}

// Due vettori uguali a meno di una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance,
                         const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << label << ", componente " << i;
    }
}

// Le tensioni di due liste di triangoli uguali a meno di una tolleranza assoluta, componente per componente.
void expect_stresses_near(const std::vector<Vector3>& got, const std::vector<Vector3>& want, double tolerance,
                          const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t e = 0; e < got.size(); ++e) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(got[e][c], want[e][c], tolerance) << label << ", triangolo " << e << ", componente " << c;
        }
    }
}

// Due stati uguali (stesso tempo, stessa T, stessi u e sigma) a meno di una tolleranza RELATIVA al valore piu' grande
// di ogni campo.
void expect_states_near(const ThermoMechanicalState2D& got, const ThermoMechanicalState2D& want, double relative,
                        const std::string& label) {
    EXPECT_DOUBLE_EQ(got.time, want.time) << label;
    expect_vectors_near(got.T, want.T, relative * max_abs(want.T), label + ", T");
    expect_vectors_near(got.u, want.u, relative * max_abs(want.u), label + ", u");
    expect_stresses_near(got.sigma, want.sigma, relative * max_abs(want.sigma), label + ", sigma");
}

}  // namespace

// ESEMPIO 212. Quanti stati, a quali tempi, di che misura. Il quadrato unitario (4 nodi, 2 triangoli), la lastra
// isolata e senza carico, T0 = T_ref = 20 dappertutto (non succede niente: conta solo la forma del risultato) e
// dt = 0.1.
//   - 6 passi, salvando ogni 2: gli stati sono ai passi 0, 2, 4, 6, quindi 6 / 2 + 1 = 4 stati, ai tempi 0, 0.2, 0.4 e
//     0.6 (il passo per dt);
//   - 10 passi, salvando ogni passo: 11 stati, l'ultimo al tempo 10 x 0.1 = 1;
//   - in ogni stato T ha 4 numeri (uno per nodo), u ne ha 8 (due per nodo) e sigma ne ha 2 (uno per triangolo).
TEST(Thermoelastic2d, TheStatesAreOneEverySaveEveryStepsStartingAtTimeZero) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 1.0;
    problem.k = 1.0;
    problem.f = DVector(4, 0.0);
    problem.T0 = DVector(4, 20.0);
    problem.dt = 0.1;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    problem.n_steps = 6;
    problem.save_every = 2;
    std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 4u);
    const double times[4] = {0.0, 0.2, 0.4, 0.6};
    for (std::size_t m = 0; m < states.size(); ++m) {
        EXPECT_DOUBLE_EQ(states[m].time, times[m]) << "stato " << m;
        EXPECT_EQ(states[m].T.size(), 4u) << "stato " << m;
        EXPECT_EQ(states[m].u.size(), 8u) << "stato " << m;
        EXPECT_EQ(states[m].sigma.size(), 2u) << "stato " << m;
    }

    problem.n_steps = 10;
    problem.save_every = 1;
    states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 11u);
    EXPECT_DOUBLE_EQ(states[0].time, 0.0);
    EXPECT_DOUBLE_EQ(states[10].time, 1.0);
}

// ESEMPIO 213. Con zero passi c'e' un solo stato, quello al tempo 0, e i dati del calore non servono. Il problema "a
// zero" (ThermoMechanicalProblem2D problem;) non ha passi (n_steps = 0, save_every = 1) e nessun dato del calore
// (rho_c = 0, k = 0, dt = 0, nessun carico: con dei passi da fare sarebbe un errore), e la funzione non deve
// lamentarsi. Il quadrato unitario ha T0 = 30 dappertutto e T_ref = 20, quindi dT = 10, e la lastra libera (u e v del
// nodo 0 e v del nodo 1 bloccati) si dilata di e = (1 + nu) alpha dT = 0.0125 x 10 = 0.125 in ogni direzione:
//     u = e x,  v = e y:      nodo 0 = (0, 0): (0, 0),    nodo 1 = (1, 0): (0.125, 0),
//                             nodo 2 = (1, 1): (0.125, 0.125),    nodo 3 = (0, 1): (0, 0.125),
// senza nessuna tensione. T e' T0, identica.
TEST(Thermoelastic2d, WithZeroStepsThereIsOnlyTheInitialStateAndNoHeatDataIsNeeded) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    EXPECT_EQ(problem.n_steps, 0u);
    EXPECT_EQ(problem.save_every, 1u);
    EXPECT_EQ(problem.theta, 1.0);
    EXPECT_EQ(problem.rho_c, 0.0);
    EXPECT_EQ(problem.k, 0.0);
    EXPECT_EQ(problem.dt, 0.0);
    EXPECT_TRUE(problem.f.empty());

    use_hand_material(problem);
    problem.T0 = DVector(4, 30.0);
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    std::vector<ThermoMechanicalState2D> states;
    ASSERT_NO_THROW(states = solve_thermomechanical_2d(mesh, problem));
    ASSERT_EQ(states.size(), 1u);
    EXPECT_EQ(states[0].time, 0.0);
    EXPECT_EQ(states[0].T, problem.T0);
    expect_vectors_near(states[0].u, {0.0, 0.0, 0.125, 0.0, 0.125, 0.125, 0.0, 0.125}, 1e-14, "u");
    expect_stresses_near(states[0].sigma, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, 1e-12, "sigma");
}

// ESEMPIO 214. Ogni stato e' la catena delle funzioni che gia' esistono, nell'ordine giusto e con i dati giusti. Una
// mesh irregolare (la stessa dell'ESEMPIO 198), con tutti i numeri diversi uno dall'altro: rho_c = 2.5, k = 0.7, un
// carico uniforme q = 3, i nodi 0 e 3 a 310 e 280, E = 70, nu = 0.2, alpha = 0.003, T_ref = 290, theta = 0.6, e dei
// vincoli che NON sono zero (u e v del nodo 0 e v del nodo 1: 0.01, -0.02 e 0.03). 6 passi di 0.2, salvando ogni 2: 4
// stati. Per ogni stato s (al passo p = 2 s) si rifa' a mano la catena, con le funzioni di prima:
//     T     = T0 per p = 0, altrimenti solve_transient_2d dopo p passi tutti insieme;
//     D     = plane_strain_matrix(E, nu),            F = assemble_thermal_load_2d(mesh, D, nu, alpha, T, T_ref),
//     u     = solve_displacement_2d(mesh, D, F, fixed_dofs, fixed_displacements),
//     sigma = stress_field_2d(mesh, D, nu, alpha, u, T, T_ref).
// Il tempo e' p x dt. Uno scambio fra due dati (rho_c e k, E e nu), T_ref dimenticato, i vincoli presi a zero, la
// temperatura iniziale usata per le tensioni al posto di quella dell'istante: ognuno cambia qualche numero.
TEST(Thermoelastic2d, EveryStateIsTheChainOfTheExistingFunctions) {
    const Mesh2D mesh = general_mesh();
    ThermoMechanicalProblem2D problem = general_problem(mesh);
    problem.fixed_displacements = {0.01, -0.02, 0.03};

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 4u);

    const Matrix3 D = plane_strain_matrix(problem.E, problem.nu);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const std::size_t passes = 2 * s;
        ThermoMechanicalState2D want;
        want.time = static_cast<double>(passes) * problem.dt;
        want.T = problem.T0;
        if (passes > 0) {
            want.T = solve_transient_2d(mesh, problem.rho_c, problem.k, problem.f, problem.fixed_nodes,
                                        problem.fixed_temperatures, problem.T0, problem.dt, passes, problem.theta);
        }
        const std::vector<double> F =
            assemble_thermal_load_2d(mesh, D, problem.nu, problem.alpha, want.T, problem.T_ref);
        want.u = solve_displacement_2d(mesh, D, F, problem.fixed_dofs, problem.fixed_displacements);
        want.sigma = stress_field_2d(mesh, D, problem.nu, problem.alpha, want.u, want.T, problem.T_ref);

        expect_states_near(states[s], want, 1e-9, "stato " + std::to_string(s));
    }
    EXPECT_EQ(states[0].T, problem.T0);
    EXPECT_GT(max_abs(states[3].sigma), 1e-3) << "il test deve avere delle tensioni da confrontare";
    double moved = 0.0;
    for (std::size_t i = 0; i < problem.T0.size(); ++i) {
        moved = std::max(moved, std::fabs(states[3].T[i] - problem.T0[i]));
    }
    EXPECT_GT(moved, 1.0) << "il test deve avere un calore che cambia nel tempo";
}

// ESEMPIO 215. Salvare ogni 1 passo o ogni 3 passi non cambia i numeri: la mesh e il problema dell'ESEMPIO 214, ma con
// 6 passi e un vincolo a zero. Salvando ogni passo ci sono 7 stati (ai passi 0, ..., 6), salvando ogni 3 ce ne sono 3
// (ai passi 0, 3, 6) e devono essere uguali agli stati 0, 3 e 6 della lista lunga: ripartire ogni 3 passi dal risultato
// di prima e' come fare 6 passi di seguito.
TEST(Thermoelastic2d, SavingEveryThreeStepsGivesTheSameNumbersAsSavingEveryStep) {
    const Mesh2D mesh = general_mesh();
    ThermoMechanicalProblem2D problem = general_problem(mesh);

    problem.save_every = 1;
    const std::vector<ThermoMechanicalState2D> every_step = solve_thermomechanical_2d(mesh, problem);
    problem.save_every = 3;
    const std::vector<ThermoMechanicalState2D> every_third = solve_thermomechanical_2d(mesh, problem);

    ASSERT_EQ(every_step.size(), 7u);
    ASSERT_EQ(every_third.size(), 3u);
    for (std::size_t s = 0; s < every_third.size(); ++s) {
        expect_states_near(every_third[s], every_step[3 * s], 1e-12, "stato " + std::to_string(s));
    }
}

// ESEMPIO 216. La lastra libera che si scalda in modo uniforme. La mesh 3 x 2 celle di [0, 2] x [0, 1] (12 nodi), la
// lastra isolata (nessun nodo a temperatura imposta), T0 = 25 e T_ref = 20, un carico uniforme q = 8 con rho_c = 2 e
// passi di 0.25: ogni passo scalda di q dt / rho_c = 8 x 0.25 / 2 = 1 grado, quindi dopo p passi la temperatura e'
// 25 + p in ogni nodo. Per ogni schema theta: e' una conseguenza solo di K 1 = 0 (il calore non fluisce se T e'
// uniforme) e di f = q M 1 (la massa porta il carico dentro: M (T_new - T_old) = dt f).
// La lastra e' libera (tre vincoli in (0, 0) e (L, 0), come per la trave) e si dilata di e = (1 + nu) alpha dT =
// 0.0125 (5 + p) in ogni direzione, u = e x e v = e y, senza tensioni: la deformazione B u = e (1, 1, 0) e' proprio
// quella termica.
TEST(Thermoelastic2d, AFreePlateThatIsHeatedUniformlyExpandsWithoutStress) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    for (double theta : {1.0, 0.5, 0.0, 0.8}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(12, 25.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        pin_corners(problem, 3);

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", passo " + std::to_string(p);
            const double e = 0.0125 * (5.0 + static_cast<double>(p));
            std::vector<double> u_want;
            for (std::size_t node = 0; node < 12; ++node) {
                u_want.push_back(e * mesh.x(node));
                u_want.push_back(e * mesh.y(node));
            }
            expect_vectors_near(states[p].T, DVector(12, 25.0 + static_cast<double>(p)), 1e-9, label + ", T");
            expect_vectors_near(states[p].u, u_want, 1e-10, label + ", u");
            expect_stresses_near(states[p].sigma, std::vector<Vector3>(mesh.num_triangles(), {0.0, 0.0, 0.0}), 1e-8,
                                 label + ", sigma");
        }
    }
}

// ESEMPIO 217. I vincoli contano: la stessa lastra libera dell'ESEMPIO 216 (theta = 1, 2 passi), ma con gli spostamenti
// imposti che sono un moto rigido, una traslazione (3, -2) e una piccola rotazione di 0.1 radianti:
//     u = 3 - 0.1 y + e x,        v = -2 + 0.1 x + e y.
// Una rotazione rigida non deforma (du/dx = dv/dy = 0 e du/dy + dv/dx = -0.1 + 0.1 = 0), quindi le tensioni restano 0 e
// i valori da imporre nei tre vincoli sono quelli di questa formula: u(0, 0) = 3, v(0, 0) = -2 e, in (L, 0) = (2, 0),
// v = -2 + 0.1 x 2 = -1.8. Chi non passasse gli spostamenti imposti (li prendesse a zero) troverebbe la lastra ferma
// in (0, 0) invece che in (3, -2).
TEST(Thermoelastic2d, TheImposedDisplacementsAreARigidMotionAddedToTheExpansion) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 1.0;
    problem.f = assemble_load_2d(mesh, 8.0);
    problem.T0 = DVector(12, 25.0);
    problem.dt = 0.25;
    problem.n_steps = 2;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 7};
    problem.fixed_displacements = {3.0, -2.0, -1.8};

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 3u);
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double e = 0.0125 * (5.0 + static_cast<double>(p));
        std::vector<double> u_want;
        for (std::size_t node = 0; node < 12; ++node) {
            u_want.push_back(3.0 - 0.1 * mesh.y(node) + e * mesh.x(node));
            u_want.push_back(-2.0 + 0.1 * mesh.x(node) + e * mesh.y(node));
        }
        expect_vectors_near(states[p].u, u_want, 1e-10, "passo " + std::to_string(p) + ", u");
        expect_stresses_near(states[p].sigma, std::vector<Vector3>(mesh.num_triangles(), {0.0, 0.0, 0.0}), 1e-8,
                             "passo " + std::to_string(p) + ", sigma");
    }
}

// ESEMPIO 218. La lastra incastrata che si scalda: la pressione cresce linearmente nel tempo. La mesh 4 x 4 celle di
// [0, 2] x [0, 1] (25 nodi), con tutti i nodi del bordo bloccati (u = v = 0): i 9 nodi interni sono liberi. La lastra
// e' isolata, T0 = T_ref = 20, un carico uniforme q = 8 con rho_c = 2 e passi di 0.25: dopo p passi dT = p gradi. La
// lastra non si muove (u = 0: la forza termica di un campo uniforme e' 0 sui nodi interni, e quelli del bordo sono
// fermi) e la tensione e' quella di un corpo trattenuto, sigma = D (0 - eps_th) = -D eps_th. Con
// e = (1 + nu) alpha dT = 0.0125 dT:
//     eps_th = e (1, 1, 0),        D eps_th = e (120 + 40, 40 + 120, 0) = 0.0125 dT (160, 160, 0) = 2 dT (1, 1, 0),
// cioe' sigma = -2 dT (1, 1, 0) = -E alpha dT / (1 - 2 nu) (1, 1, 0) in ogni triangolo, per theta = 1 e theta = 1/2.
TEST(Thermoelastic2d, AClampedPlateIsCompressedMoreAndMoreAsItHeatsUp) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 4, 4);
    for (double theta : {1.0, 0.5}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(25, 20.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        for (std::size_t node : boundary_nodes(mesh)) {
            problem.fixed_dofs.push_back(2 * node);
            problem.fixed_dofs.push_back(2 * node + 1);
            problem.fixed_displacements.push_back(0.0);
            problem.fixed_displacements.push_back(0.0);
        }
        ASSERT_EQ(problem.fixed_dofs.size(), 32u);

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", passo " + std::to_string(p);
            const double dT = static_cast<double>(p);
            expect_vectors_near(states[p].u, std::vector<double>(50, 0.0), 1e-10, label + ", u");
            expect_stresses_near(states[p].sigma,
                                 std::vector<Vector3>(mesh.num_triangles(), {-2.0 * dT, -2.0 * dT, 0.0}), 1e-8,
                                 label + ", sigma");
        }
    }
}

// ESEMPIO 219. La striscia: il calore decade di un fattore g a ogni passo, che si calcola a mano. Le righe di nodi sono
// tre, y = 0, 1/2 e 1 (h = 1/2): le due facce restano a T_ref = 20 e la riga di mezzo e' l'incognita (la stessa in
// tutte le colonne: la striscia e' uguale in x). T non dipende da x, quindi la riga di un nodo di mezzo si calcola
// come per un'asta 1D, con i vicini (le facce) a scarto zero:
//     massa:      rho_c (2 h / 3) = 2 (1/3) = 2/3,            rigidezza:      k (2 / h) = 3 x 4 = 12.
// Il passo con lo schema theta, (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old, scritto sullo scarto da T_ref,
// diventa
//     (2/3 + 12 theta dt) T_new = (2/3 - 12 (1 - theta) dt) T_old,
// e con dt = 1/36 (12 dt = 1/3):   g = (2/3 - (1 - theta) / 3) / (2/3 + theta / 3) = (1 + theta) / (2 + theta).
// Per theta = 1 (Eulero all'indietro) g = 2/3, per theta = 1/2 (Crank-Nicolson) g = 3/5, per theta = 0 (Eulero in
// avanti) g = 1/2. Con la riga di mezzo a 20 + 60 all'inizio, dopo p passi sta a 20 + 60 g^p, e le facce stanno a 20
// ESATTAMENTE. E' la prova di theta, di rho_c e k (non scambiati), delle facce fissate e del fatto che ogni passo
// riparte dal risultato del precedente.
TEST(Thermoelastic2d, TheMiddleRowOfTheStripDecaysByTheHandFactorEveryStep) {
    struct Scheme {
        double theta;
        double g;
    };
    const Mesh2D mesh = strip_mesh(8);
    for (const Scheme& scheme : {Scheme{1.0, 2.0 / 3.0}, Scheme{0.5, 3.0 / 5.0}, Scheme{0.0, 1.0 / 2.0}}) {
        const std::vector<ThermoMechanicalState2D> states =
            solve_thermomechanical_2d(mesh, strip_problem(8, scheme.theta, 1.0 / 36.0, 3, 1));
        ASSERT_EQ(states.size(), 4u);
        double factor = 1.0;
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(scheme.theta) + ", passo " + std::to_string(p);
            for (std::size_t i = 0; i <= 8; ++i) {
                EXPECT_NEAR(states[p].T[9 + i], 20.0 + 60.0 * factor, 1e-9) << label << ", riga di mezzo, nodo " << i;
                if (p > 0) {
                    EXPECT_EQ(states[p].T[i], 20.0) << label << ", faccia in basso, nodo " << i;
                    EXPECT_EQ(states[p].T[18 + i], 20.0) << label << ", faccia in alto, nodo " << i;
                }
            }
            factor *= scheme.g;
        }
    }
}

// ESEMPIO 220. Con il calore decadono anche gli spostamenti e le tensioni: la parte elastica e' lineare nello scarto
// di temperatura (e i vincoli sono a zero), quindi se lo scarto di T diventa g volte quello di prima, u e sigma
// diventano g volte quelli di prima, ESATTAMENTE (a meno degli errori del gradiente coniugato). La striscia
// dell'ESEMPIO 219 con i tre schemi e 3 passi: lo stato p ha u = g^p u(0) e sigma = g^p sigma(0). Si controlla anche
// che lo stato 0 non sia vuoto (tensioni di decine, spostamenti di decimi) e che la tensione sia di compressione:
// dove la lastra e' piu' calda vorrebbe dilatarsi, e i rulli la fermano.
TEST(Thermoelastic2d, TheDisplacementsAndStressesOfTheStripDecayLikeTheHeat) {
    struct Scheme {
        double theta;
        double g;
    };
    const Mesh2D mesh = strip_mesh(8);
    for (const Scheme& scheme : {Scheme{1.0, 2.0 / 3.0}, Scheme{0.5, 3.0 / 5.0}, Scheme{0.0, 1.0 / 2.0}}) {
        const std::vector<ThermoMechanicalState2D> states =
            solve_thermomechanical_2d(mesh, strip_problem(8, scheme.theta, 1.0 / 36.0, 3, 1));
        ASSERT_EQ(states.size(), 4u);

        const std::vector<double>& u0 = states[0].u;
        const std::vector<Vector3>& sigma0 = states[0].sigma;
        EXPECT_GT(max_abs(sigma0), 20.0);
        EXPECT_GT(max_abs(u0), 0.1);
        double compressive_sum = 0.0;
        for (const Vector3& s : sigma0) {
            compressive_sum += s[0];
        }
        EXPECT_LT(compressive_sum, 0.0) << "sigma_xx deve essere di compressione (negativa)";

        double factor = 1.0;
        for (std::size_t p = 1; p < states.size(); ++p) {
            factor *= scheme.g;
            std::vector<double> u_want = u0;
            for (double& value : u_want) {
                value *= factor;
            }
            std::vector<Vector3> sigma_want = sigma0;
            for (Vector3& s : sigma_want) {
                for (std::size_t c = 0; c < 3; ++c) {
                    s[c] *= factor;
                }
            }
            const std::string label = "theta = " + std::to_string(scheme.theta) + ", passo " + std::to_string(p);
            expect_vectors_near(states[p].u, u_want, 1e-9 * max_abs(u0), label + ", u");
            expect_stresses_near(states[p].sigma, sigma_want, 1e-9 * max_abs(sigma0), label + ", sigma");
        }
    }
}

// ESEMPIO 221. A meta' di una striscia LUNGA le tensioni sono quelle fatte a mano. La striscia dell'ESEMPIO 219 con 24
// colonne (lunga 12, celle quadrate di lato 1/2) e Eulero all'indietro (g = 2/3). Lontano dalle estremita' (dove i
// rulli bloccano u) la striscia e' uguale in x: u = 0 e v dipende solo da y. Con E = 100, nu = 1/4 (lambda = mu = 40),
// alpha = 1/100, e con dT_X la media dello scarto di T dei tre nodi di un triangolo X e e_X = (1 + nu) alpha dT_X:
//   - il triangolo "in basso a destra" della cella fra la riga 0 e la riga 1 ha due nodi nella riga 0 (scarto 0) e uno
//     nella riga 1 (scarto 60): dT_X = 20 e e_X = 1/4. Quello "in alto a sinistra" ha un nodo a 0 e due a 60: dT_X = 40
//     e e_X = 1/2.
//   - la deformazione e' (0, gamma, 0), con gamma = dv/dy uguale nei due triangoli (la stessa differenza di v sullo
//     stesso h). Sui nodi della faccia (liberi in v) la forza verticale e' proporzionale alla somma delle sigma_yy dei
//     due triangoli della cella sopra, e deve essere 0 perche' sotto non c'e' niente: la somma delle sigma_yy dei due
//     triangoli di una cella e' 0 (e cosi' per le celle della riga dopo, dove la forza e' la differenza fra due
//     somme). Con sigma_yy = (lambda + 2 mu) gamma - (2 lambda + 2 mu) e_X = 120 gamma - 160 e_X:
//         (120 gamma - 40) + (120 gamma - 80) = 0,        gamma = 1/2.
//   - la parte elastica della deformazione e' (-e_X, gamma - e_X, 0), quindi
//         sigma_xx = 120 (-e_X) + 40 (gamma - e_X) = 20 - 160 e_X,        sigma_yy = 40 (-e_X) + 120 (gamma - e_X) =
//         60 - 160 e_X.
//     Con e_X = 1/4: sigma = (-20, +20). Con e_X = 1/2: sigma = (-60, -20).
//   - la cella sopra (riga 1 -> riga 2) e' lo specchio: il triangolo in basso ha due nodi a 60 e uno a 0 (dT_X = 40,
//     sigma = (-60, -20)) e quello in alto un nodo a 60 e due a 0 (dT_X = 20, sigma = (-20, +20)).
//   - v cresce di gamma h = 1/4 fra una riga e la successiva: v(riga 1) - v(riga 0) = v(riga 2) - v(riga 1) = 1/4.
// Questi numeri sono quelli di una striscia infinita: l'effetto dei rulli scende di un fattore 3 circa per colonna a
// partire dalle estremita' (18 nella prima colonna, 0.6 nella terza), e nelle tre colonne di mezzo, a 10-12 colonne
// dalle estremita', e' sotto 2e-4 sulle sigma e sotto 1e-7 su v (si chiede 1e-3 per le sigma e 1e-5 per v).
// Dopo un passo tutto e' moltiplicato per g = 2/3.
TEST(Thermoelastic2d, InTheMiddleOfALongStripTheStressesAreTheHandNumbers) {
    const std::size_t nx = 24;
    const Mesh2D mesh = strip_mesh(nx);
    const std::vector<ThermoMechanicalState2D> states =
        solve_thermomechanical_2d(mesh, strip_problem(nx, 1.0, 1.0 / 36.0, 1, 1));
    ASSERT_EQ(states.size(), 2u);

    // sigma (xx, yy) dei triangoli {basso, alto} delle celle della riga 0 e della riga 1, e le tre colonne di mezzo
    const double want[2][2][2] = {{{-20.0, 20.0}, {-60.0, -20.0}}, {{-60.0, -20.0}, {-20.0, 20.0}}};
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double factor = (p == 0) ? 1.0 : 2.0 / 3.0;
        for (std::size_t i = nx / 2 - 1; i <= nx / 2 + 1; ++i) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t half = 0; half < 2; ++half) {
                    const Vector3& s = states[p].sigma[2 * (row * nx + i) + half];
                    const std::string label = "passo " + std::to_string(p) + ", cella (" + std::to_string(i) + ", " +
                                              std::to_string(row) + "), triangolo " + std::to_string(half);
                    EXPECT_NEAR(s[0], factor * want[row][half][0], 1e-3) << label << ", sigma_xx";
                    EXPECT_NEAR(s[1], factor * want[row][half][1], 1e-3) << label << ", sigma_yy";
                    EXPECT_NEAR(s[2], 0.0, 1e-3) << label << ", sigma_xy";
                }
            }
        }
        const std::size_t mid = nx / 2;
        const double v0 = states[p].u[2 * mid + 1];
        const double v1 = states[p].u[2 * ((nx + 1) + mid) + 1];
        const double v2 = states[p].u[2 * (2 * (nx + 1) + mid) + 1];
        EXPECT_NEAR(v1 - v0, factor * 0.25, 1e-5) << "passo " << p;
        EXPECT_NEAR(v2 - v1, factor * 0.25, 1e-5) << "passo " << p;
        EXPECT_NEAR(states[p].u[2 * ((nx + 1) + mid)], 0.0, 1e-9) << "passo " << p << ", u in mezzo";
    }
}

// ESEMPIO 222. Il decadimento ha l'ordine in tempo dello schema theta, come per il calore da solo. Nella striscia
// dell'ESEMPIO 219 la riga di mezzo e' un modo solo, e rigidezza e massa stanno come 12 / (2/3) = 18: senza errore di
// passo il modo decadrebbe come e^(-18 t), e al tempo t = 1/18 varrebbe e^(-1). Con n passi di 1 / (18 n) i due schemi
// danno un fattore g^n, con
//     Eulero all'indietro:  g^n = (1 / (1 + 1/n))^n,        Crank-Nicolson:  g^n = ((1 - 1/(2 n)) / (1 + 1/(2 n)))^n,
// e sigma alla fine e' g^n volte sigma all'inizio. L'errore rispetto a e^(-1) cala di un fattore 2 quando n raddoppia
// per Eulero (ordine 1: gli ordini osservati sono 0.96 e 0.98) e di un fattore 4 per Crank-Nicolson (ordine 2,
// osservato 2.00). Si guarda sigma_xx del triangolo in basso della cella di mezzo della riga 0 a n = 8, 16 e 32 passi.
TEST(Thermoelastic2d, TheStressDecayHasTheTimeOrderOfTheScheme) {
    const Mesh2D mesh = strip_mesh(8);
    const std::size_t triangle = 2 * (0 * 8 + 4);
    const double exact = std::exp(-1.0);
    for (double theta : {1.0, 0.5}) {
        std::vector<double> errors;
        for (std::size_t n : {8u, 16u, 32u}) {
            const std::vector<ThermoMechanicalState2D> states =
                solve_thermomechanical_2d(mesh, strip_problem(8, theta, 1.0 / (18.0 * static_cast<double>(n)), n, n));
            ASSERT_EQ(states.size(), 2u);
            ASSERT_GT(std::fabs(states[0].sigma[triangle][0]), 1.0);
            const double ratio = states[1].sigma[triangle][0] / states[0].sigma[triangle][0];
            errors.push_back(std::fabs(ratio - exact));
        }
        const double order_first = std::log2(errors[0] / errors[1]);
        const double order_second = std::log2(errors[1] / errors[2]);
        const double expected = (theta == 1.0) ? 1.0 : 2.0;
        EXPECT_NEAR(order_first, expected, 0.1) << "theta = " << theta << ", n = 8 -> 16";
        EXPECT_NEAR(order_second, expected, 0.1) << "theta = " << theta << ", n = 16 -> 32";
    }
}

// ESEMPIO 223. Una lastra d'acciaio che si raffredda dalle due facce ha la tensione "del libro". La lastra e' spessa
// H = 0.1 m e lunga 0.4 m, acciaio con E = 210 GPa, nu = 0.3, alpha = 12e-6, rho_c = 7850 x 470 = 3.6895e6 J/(m^3 K) e
// k = 45 W/(m K) (k / rho_c = 1.22e-5 m^2/s). Le facce sono tenute a T_ref = 293.15 K e la lastra parte da
// T_ref + 100 sin(pi y / H): si raffredda, e il modo decade come e^(-t / tau) con tau = H^2 / (pi^2 k / rho_c) = 83 s.
// Con i rulli alle estremita' (u = 0), a meta' lunghezza la lastra e' uguale in x, con u = 0 e v(y): e' la striscia
// dell'ESEMPIO 221 con numeri generici, e la media delle due sigma_xx dei triangoli di una cella e' esattamente
//     sigma_xx = -E alpha dT / (1 - nu),
// con dT la media dello scarto di T delle due righe di nodi della cella (e' la formula di un corpo che non puo'
// dilatarsi in x, in deformazione piana: per dT = 100 K sono 360 MPa di compressione). Con 8 celle in altezza (32 in
// lunghezza), passi di 5 s, Crank-Nicolson e stati salvati a 0, 50 e 100 s:
//   - la formula con lo scarto di T calcolato si verifica a meno di 1e-4 volte 360 MPa (l'effetto dei rulli a meta'
//     lunghezza e' sotto 1e-5);
//   - con lo scarto "continuo", 100 sin(pi y / H) e^(-t / tau) preso a meta' altezza della cella, la differenza e'
//     l'errore del metodo: la media di due nodi invece del valore in mezzo (1 - cos(pi/16) = 1.9%) e il decadimento
//     della mesh, 1.3% troppo veloce in 8 celle. Si chiede che stia sotto il 5% di 360 MPa.
TEST(Thermoelastic2d, ASteelSlabThatIsCooledFromBothSidesHasTheTextbookStress) {
    const std::size_t nx = 32;
    const std::size_t ny = 8;
    const double height = 0.1;
    const double length = 0.4;
    const double t_ref = 293.15;
    const double amplitude = 100.0;
    const Mesh2D mesh = rectangle_mesh(length, height, nx, ny);

    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, ny, t_ref, amplitude);
    problem.rho_c = 7850.0 * 470.0;
    problem.k = 45.0;
    problem.dt = 5.0;
    problem.n_steps = 20;
    problem.save_every = 10;
    problem.theta = 0.5;
    problem.E = 210.0e9;
    problem.nu = 0.3;
    problem.alpha = 12.0e-6;
    problem.T_ref = t_ref;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 3u);

    const double scale = problem.E * problem.alpha * amplitude / (1.0 - problem.nu);
    const double tau = height * height / (kPi * kPi * (problem.k / problem.rho_c));
    const std::size_t column = nx / 2;
    for (const ThermoMechanicalState2D& state : states) {
        for (std::size_t row = 0; row < ny; ++row) {
            const std::size_t cell = row * nx + column;
            const double average_xx = 0.5 * (state.sigma[2 * cell][0] + state.sigma[2 * cell + 1][0]);
            const double below = state.T[row * (nx + 1) + column] - t_ref;
            const double above = state.T[(row + 1) * (nx + 1) + column] - t_ref;
            const double from_the_formula = -problem.E * problem.alpha * 0.5 * (below + above) / (1.0 - problem.nu);
            EXPECT_NEAR(average_xx, from_the_formula, 1e-4 * scale)
                << "t = " << state.time << ", riga di celle " << row << ", formula con la T calcolata";

            const double y_middle = (static_cast<double>(row) + 0.5) * height / static_cast<double>(ny);
            const double continuum =
                -scale * std::sin(kPi * y_middle / height) * std::exp(-state.time / tau);
            EXPECT_NEAR(average_xx, continuum, 0.05 * scale)
                << "t = " << state.time << ", riga di celle " << row << ", soluzione continua";
        }
    }

    // la compressione c'e' davvero: a meta' altezza, al tempo 0, sono 346 MPa (il 96% di 360)
    const std::size_t middle_cell = (ny / 2) * nx + column;
    const double middle_xx = 0.5 * (states[0].sigma[2 * middle_cell][0] + states[0].sigma[2 * middle_cell + 1][0]);
    EXPECT_LT(middle_xx, -0.95 * scale);
    EXPECT_GT(middle_xx, -1.0 * scale);
}

// ESEMPIO 224. Leggi di scala: i numeri che dipendono in modo semplice da un dato. La mesh irregolare e i dati
// dell'ESEMPIO 214, ma con i vincoli a zero (con vincoli diversi da zero le leggi di scala valgono per la parte
// elastica, non per i vincoli).
//   - E doppio: gli spostamenti restano gli stessi (K e F raddoppiano) e le tensioni raddoppiano;
//   - alpha doppio: F e eps_th raddoppiano, quindi gli spostamenti e le tensioni raddoppiano;
//   - il calore doppio (lo scarto da T_ref della temperatura iniziale, delle temperature imposte e il carico f, tutti
//     moltiplicati per 2): la temperatura e' doppia, e con lei gli spostamenti e le tensioni;
//   - il calore cambiato di segno (fattore -1): tutto cambia di segno.
// T non cambia con E e con alpha (il calore non sente la meccanica).
TEST(Thermoelastic2d, DisplacementsAndStressesScaleWithEWithAlphaAndWithTheHeat) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);
    ASSERT_GT(max_abs(base_states[3].sigma), 1e-3);

    ThermoMechanicalProblem2D stiff = base;
    stiff.E = 2.0 * base.E;
    const std::vector<ThermoMechanicalState2D> stiff_states = solve_thermomechanical_2d(mesh, stiff);

    ThermoMechanicalProblem2D expanding = base;
    expanding.alpha = 2.0 * base.alpha;
    const std::vector<ThermoMechanicalState2D> expanding_states = solve_thermomechanical_2d(mesh, expanding);

    for (double factor : {2.0, -1.0}) {
        ThermoMechanicalProblem2D hot = base;
        for (double& value : hot.T0) {
            value = base.T_ref + factor * (value - base.T_ref);
        }
        for (double& value : hot.fixed_temperatures) {
            value = base.T_ref + factor * (value - base.T_ref);
        }
        for (double& value : hot.f) {
            value *= factor;
        }
        const std::vector<ThermoMechanicalState2D> hot_states = solve_thermomechanical_2d(mesh, hot);
        ASSERT_EQ(hot_states.size(), 4u);
        for (std::size_t s = 0; s < 4; ++s) {
            const std::string label = "calore x " + std::to_string(factor) + ", stato " + std::to_string(s);
            std::vector<double> T_want = base_states[s].T;
            for (double& value : T_want) {
                value = base.T_ref + factor * (value - base.T_ref);
            }
            std::vector<double> u_want = base_states[s].u;
            for (double& value : u_want) {
                value *= factor;
            }
            std::vector<Vector3> sigma_want = base_states[s].sigma;
            for (Vector3& t : sigma_want) {
                for (std::size_t c = 0; c < 3; ++c) {
                    t[c] *= factor;
                }
            }
            expect_vectors_near(hot_states[s].T, T_want, 1e-9 * max_abs(T_want), label + ", T");
            expect_vectors_near(hot_states[s].u, u_want, 1e-9 * max_abs(u_want), label + ", u");
            expect_stresses_near(hot_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want), label + ", sigma");
        }
    }

    for (std::size_t s = 0; s < 4; ++s) {
        const std::string label = "stato " + std::to_string(s);
        EXPECT_EQ(stiff_states[s].T, base_states[s].T) << label << ", E doppio: T";
        expect_vectors_near(stiff_states[s].u, base_states[s].u, 1e-9 * max_abs(base_states[s].u) + 1e-15,
                            label + ", E doppio: u");
        std::vector<Vector3> sigma_want = base_states[s].sigma;
        std::vector<double> u_want = base_states[s].u;
        for (Vector3& t : sigma_want) {
            for (std::size_t c = 0; c < 3; ++c) {
                t[c] *= 2.0;
            }
        }
        for (double& value : u_want) {
            value *= 2.0;
        }
        expect_stresses_near(stiff_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want) + 1e-12,
                             label + ", E doppio: sigma");
        EXPECT_EQ(expanding_states[s].T, base_states[s].T) << label << ", alpha doppio: T";
        expect_vectors_near(expanding_states[s].u, u_want, 1e-9 * max_abs(u_want) + 1e-15,
                            label + ", alpha doppio: u");
        expect_stresses_near(expanding_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want) + 1e-12,
                             label + ", alpha doppio: sigma");
    }
}

// ESEMPIO 225. T_ref non conta se si sposta tutto: la mesh irregolare e i dati dell'ESEMPIO 214 (vincoli a zero), con
// T_ref, la temperatura iniziale e le temperature imposte aumentate di 100 gradi. Il calore non sente l'origine delle
// temperature (K e M non dipendono da T), quindi T e' quella di prima piu' 100, e gli spostamenti e le tensioni, che
// dipendono solo da T - T_ref, sono gli stessi. Chi usasse T_ref = 0 al posto di T_ref, o lo dimenticasse, troverebbe
// delle tensioni enormi (T e' 390 e piu' contro un T_ref di 290: 100 gradi di scarto in piu').
TEST(Thermoelastic2d, ShiftingAllTheTemperaturesAndTRefTogetherChangesNothingButT) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);

    ThermoMechanicalProblem2D shifted = base;
    shifted.T_ref += 100.0;
    for (double& value : shifted.T0) {
        value += 100.0;
    }
    for (double& value : shifted.fixed_temperatures) {
        value += 100.0;
    }
    const std::vector<ThermoMechanicalState2D> shifted_states = solve_thermomechanical_2d(mesh, shifted);

    ASSERT_EQ(shifted_states.size(), base_states.size());
    for (std::size_t s = 0; s < base_states.size(); ++s) {
        const std::string label = "stato " + std::to_string(s);
        std::vector<double> T_want = base_states[s].T;
        for (double& value : T_want) {
            value += 100.0;
        }
        expect_vectors_near(shifted_states[s].T, T_want, 1e-9 * max_abs(T_want), label + ", T");
        expect_vectors_near(shifted_states[s].u, base_states[s].u, 1e-9 * max_abs(base_states[s].u) + 1e-15,
                            label + ", u");
        expect_stresses_near(shifted_states[s].sigma, base_states[s].sigma,
                             1e-9 * max_abs(base_states[s].sigma) + 1e-12, label + ", sigma");
    }
}

// ESEMPIO 226. Dati sbagliati, quelli che controlla la funzione stessa: save_every = 0 (non si salverebbe mai: lancia
// std::invalid_argument PRIMA di dividere per zero, n_steps % 0 farebbe cadere il programma) e n_steps che non e' un
// multiplo di save_every (6 passi salvando ogni 4: l'ultimo stato non sarebbe in fondo). Con zero passi qualunque
// save_every >= 1 va bene (0 e' multiplo di tutti), ma save_every = 0 no.
TEST(Thermoelastic2d, SaveEveryMustBeAtLeastOneAndDivideTheSteps) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 1.0;
    problem.k = 1.0;
    problem.f = DVector(4, 0.0);
    problem.T0 = DVector(4, 20.0);
    problem.dt = 0.1;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    problem.n_steps = 4;
    problem.save_every = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);
    problem.n_steps = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);

    problem.n_steps = 6;
    problem.save_every = 4;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);
    problem.save_every = 5;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);

    problem.save_every = 3;
    EXPECT_EQ(solve_thermomechanical_2d(mesh, problem).size(), 3u);
    problem.n_steps = 0;
    EXPECT_EQ(solve_thermomechanical_2d(mesh, problem).size(), 1u);
}

// ESEMPIO 227. Dati sbagliati delle funzioni che si chiamano: la funzione non li ricontrolla, lascia passare
// l'eccezione std::invalid_argument di chi li trova. Il quadrato unitario, lastra isolata, parte da un problema valido
// (si controlla che lo sia) e cambia un dato alla volta:
//   - il materiale: E = 0 o nu = 1/2 (plane_strain_matrix), anche con zero passi;
//   - la temperatura iniziale con 3 numeri invece di 4 (assemble_thermal_load_2d);
//   - i vincoli: 2 gradi di liberta' invece di 3, o valori in numero diverso (solve_displacement_2d);
//   - il calore, con almeno un passo: k = 0, dt = 0, un carico di 3 numeri, theta = 2 (solve_transient_2d).
TEST(Thermoelastic2d, WrongDataIsReportedByTheFunctionThatFindsIt) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D valid;
    use_hand_material(valid);
    valid.rho_c = 1.0;
    valid.k = 1.0;
    valid.f = DVector(4, 0.0);
    valid.T0 = DVector(4, 25.0);
    valid.dt = 0.1;
    valid.n_steps = 2;
    valid.theta = 1.0;
    valid.save_every = 1;
    valid.fixed_dofs = {0, 1, 3};
    valid.fixed_displacements = {0.0, 0.0, 0.0};
    ASSERT_NO_THROW(solve_thermomechanical_2d(mesh, valid));

    ThermoMechanicalProblem2D bad = valid;
    bad.E = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);
    bad.n_steps = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.nu = 0.5;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.T0 = DVector(3, 25.0);
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.fixed_dofs = {0, 1};
    bad.fixed_displacements = {0.0, 0.0};
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.fixed_displacements = {0.0, 0.0};
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.k = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.dt = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.f = DVector(3, 0.0);
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.theta = 2.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);
}