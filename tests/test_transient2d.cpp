#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include "Solver2D.h"

// ---- La lastra che cambia nel tempo: lo schema theta in 2D -----------------------------------------------------
//
// In 1D abbiamo seguito una barra nel tempo con lo schema theta (solve_transient). Qui facciamo lo stesso con la
// lastra: solve_transient_2d(mesh, rho_c, k, f, fixed_nodes, fixed_values, T0, dt, n_steps, theta). Per le
// temperature nei nodi la legge del calore e'
//
//     M dT/dt + K T = f
//
// con M la matrice di massa (assemble_mass_2d(mesh, rho_c)), K quella di rigidezza (assemble_stiffness_2d(mesh, k))
// e f il vettore dei carichi (n numeri, non cambia nel tempo). Da T_old (al tempo t) a T_new (al tempo t + dt) lo
// schema theta calcola K T come una media dei due istanti:
//
//     M (T_new - T_old) / dt + K (theta T_new + (1 - theta) T_old) = f
//
// e, moltiplicando per dt e portando T_new a sinistra, ogni passo e' un sistema lineare
//
//     (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt f
//
// theta = 1 e' Eulero all'indietro (stabile per ogni dt), theta = 1/2 e' Crank-Nicolson (stabile per ogni dt e di
// ordine 2 nel tempo), theta = 0 e' Eulero in avanti (stabile solo se dt e' piccolo). E' lo stesso schema dell'1D:
// cambiano solo le matrici.
//
// I nodi fissati hanno la temperatura imposta, uguale a ogni passo. Si tolgono dal sistema come nello stato
// stazionario: A = M + theta dt K diventa la matrice con le righe e le colonne dei nodi fissati sostituite da quelle
// dell'identita' (dirichlet_matrix_triplets_2d: non cambia da un passo all'altro, si costruisce una volta sola) e il
// secondo membro (M - (1 - theta) dt K) T_old + dt f diventa b con dirichlet_rhs_2d, preso con la A intera (quella
// prima della sostituzione). Si risolve con il gradiente coniugato e il primo passo parte da T0, preso com'e' anche
// sui nodi fissati.
//
// Come "oracoli" (risposte giuste calcolate per un'altra strada) usiamo: una mesh con un solo nodo libero, dove ogni
// passo e' un'equazione con un'incognita; la soluzione con matrici piene e l'eliminazione di Gauss; tre leggi della
// fisica (lo stato stazionario e' un punto fisso, il calore finisce nello stato stazionario, in una lastra isolata il
// calore si conserva); la stabilita' dei tre schemi; e il fatto che 7 passi sono 3 passi seguiti da altri 4.

namespace {

using Dense = std::vector<std::vector<double>>;
using Nodes = std::vector<std::size_t>;

// Confronta due vettori elemento per elemento, con la tolleranza assoluta tol.
void expect_vector(const DVector& got, const DVector& want, double tol, const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ": elemento " << i;
    }
}

// La CSR scritta come tabella piena n x n, letta con get(i, j).
Dense to_dense(const CSR& A) {
    const std::size_t n = A.size();
    Dense out(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out[i][j] = A.get(i, j);
        }
    }
    return out;
}

// Risolve A x = b (tabella piena) con l'eliminazione di Gauss e lo scambio delle righe. E' lenta (n^3) ma non usa
// ne' il CG ne' altre funzioni del progetto: serve da "oracolo".
DVector gauss_solve(Dense A, DVector b) {
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::fabs(A[r][c]) > std::fabs(A[pivot][c])) {
                pivot = r;
            }
        }
        std::swap(A[c], A[pivot]);
        std::swap(b[c], b[pivot]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double ratio = A[r][c] / A[c][c];
            for (std::size_t j = c; j < n; ++j) {
                A[r][j] -= ratio * A[c][j];
            }
            b[r] -= ratio * b[c];
        }
    }
    DVector x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    return x;
}

// Lo schema theta "a mano", con matrici piene. A ogni passo, per ogni nodo libero i,
//     somma sui nodi liberi j di (M + theta dt K)(i, j) T_new[j]
//         = somma su tutti i nodi j di (M - (1 - theta) dt K)(i, j) T_old[j] + dt f[i]
//           - somma sui nodi fissati d di (M + theta dt K)(i, d) g[d]
// e' un sistema solo nei nodi liberi (Gauss); i nodi fissati valgono il valore imposto. T0 e' preso com'e'.
DVector dense_transient(const Mesh2D& m, double rho_c, double k, const DVector& f, const Nodes& fixed,
                        const DVector& values, DVector T, double dt, std::size_t n_steps, double theta) {
    const std::size_t n = m.num_nodes();
    const Dense M = to_dense(assemble_mass_2d(m, rho_c));
    const Dense K = to_dense(assemble_stiffness_2d(m, k));
    std::vector<bool> is_fixed(n, false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    Nodes free_nodes;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_fixed[i]) {
            free_nodes.push_back(i);
        }
    }
    const std::size_t nf = free_nodes.size();
    Dense A(nf, DVector(nf, 0.0));
    for (std::size_t a = 0; a < nf; ++a) {
        for (std::size_t c = 0; c < nf; ++c) {
            A[a][c] = M[free_nodes[a]][free_nodes[c]] + theta * dt * K[free_nodes[a]][free_nodes[c]];
        }
    }
    for (std::size_t step = 0; step < n_steps; ++step) {
        DVector b(nf, 0.0);
        for (std::size_t a = 0; a < nf; ++a) {
            const std::size_t i = free_nodes[a];
            double rhs = dt * f[i];
            for (std::size_t j = 0; j < n; ++j) {
                rhs += (M[i][j] - (1.0 - theta) * dt * K[i][j]) * T[j];
            }
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                rhs -= (M[i][fixed[q]] + theta * dt * K[i][fixed[q]]) * values[q];
            }
            b[a] = rhs;
        }
        if (nf > 0) {
            const DVector x = gauss_solve(A, b);
            for (std::size_t a = 0; a < nf; ++a) {
                T[free_nodes[a]] = x[a];
            }
        }
        for (std::size_t q = 0; q < fixed.size(); ++q) {
            T[fixed[q]] = values[q];
        }
    }
    return T;
}

// Il calore totale della lastra, l'integrale di rho_c T: la somma di tutti i numeri di M T (cioe' 1^T M T).
double total_heat(const Mesh2D& m, double rho_c, const DVector& T) {
    const DVector MT = assemble_mass_2d(m, rho_c) * T;
    double sum = 0.0;
    for (double v : MT) {
        sum += v;
    }
    return sum;
}

// Un numero "a caso" in [0, 1): un piccolo generatore congruenziale, cosi' i test danno gli stessi numeri con
// qualunque compilatore. `state` e' lo stato del generatore, cambia a ogni chiamata.
double next_random(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
}

// Una mesh come rectangle_mesh ma con i nodi INTERNI spostati a caso, in x e in y, di al massimo il 20% del passo
// della griglia: triangoli di forme e aree diverse, con angoli ottusi. Il bordo non si sposta: e' ancora il
// rettangolo.
Mesh2D perturbed_rectangle(double lx, double ly, std::size_t nx, std::size_t ny, unsigned long long seed) {
    const Mesh2D base = rectangle_mesh(lx, ly, nx, ny);
    const std::size_t n = base.num_nodes();
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : boundary_nodes(base)) {
        on_boundary[d] = true;
    }
    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    unsigned long long state = seed;
    std::vector<double> xs(n);
    std::vector<double> ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = base.x(i);
        ys[i] = base.y(i);
        if (!on_boundary[i]) {
            xs[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hx;
            ys[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hy;
        }
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < base.num_triangles(); ++e) {
        triangles.push_back(base.triangle(e));
    }
    return Mesh2D(xs, ys, triangles);
}

// Una mesh in cui i nodi 0 = (5, 5) e 5 = (7, 7) non appartengono a nessun triangolo.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// Un carico non uniforme, con numeri positivi e negativi (nodi che scaldano e nodi che raffreddano).
DVector mixed_load(std::size_t n) {
    DVector f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = 4.0 * std::sin(1.3 * static_cast<double>(i)) + static_cast<double>(i % 3);
    }
    return f;
}

// I dati di una chiamata VALIDA a solve_transient_2d, sulla mesh 2 x 2 (9 nodi, 8 fissati sul bordo e il centro
// libero): i test sugli errori ne cambiano uno solo alla volta.
struct Call {
    Mesh2D mesh = rectangle_mesh(1.0, 1.0, 2, 2);
    double rho_c = 2.0;
    double k = 3.0;
    DVector f = DVector(9, 1.0);
    Nodes fixed = {0, 1, 2, 3, 5, 6, 7, 8};
    DVector values = DVector(8, 5.0);
    DVector T0 = DVector(9, 20.0);
    double dt = 0.01;
    std::size_t n_steps = 3;
    double theta = 1.0;

    DVector run() const {
        return solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, n_steps, theta);
    }
};

// La chiamata c lancia std::invalid_argument, con 0 passi e con 3 passi: gli errori si controllano subito, prima di
// fare qualunque passo.
void expect_invalid(const Call& c, const std::string& what) {
    for (std::size_t steps : {0, 3}) {
        Call copy = c;
        copy.n_steps = steps;
        EXPECT_THROW(copy.run(), std::invalid_argument) << what << " (n_steps = " << steps << ")";
    }
}

}  // namespace

// ESEMPIO 96. La mesh 2 x 2 (rectangle_mesh(1, 1, 2, 2)): 9 nodi, 8 sul bordo (fissati) e uno solo libero, il centro
// c = 4. Con un solo nodo libero ogni passo e' UNA equazione. Il centro sta in 6 triangoli di area 1/8: M(c, c) =
// rho_c / 8 e K(c, c) = 4 k; i quattro nodi sugli assi (1, 3, 5, 7) hanno K(c, j) = -k, gli altri vicini (0 e 8, sulla
// diagonale dei quadrati) K(c, j) = 0; e f(c) = q / 4 con la sorgente uniforme q. Se T0 vale, sui nodi fissati, il
// valore imposto g_j, i termini di massa dei nodi fissati sono uguali a sinistra e a destra e si cancellano: resta
//
//     (rho_c / 8 + 4 theta k dt) T_new = (rho_c / 8 - 4 (1 - theta) k dt) T_old + dt (q / 4 + k S)
//
// con S = g1 + g3 + g5 + g7, cioe' T_n = T_inf + gamma^n (T_0 - T_inf), con
//
//     gamma = (rho_c - 32 (1 - theta) k dt) / (rho_c + 32 theta k dt)        T_inf = (q / 4 + k S) / (4 k)
//
// Numeri: rho_c = 2, k = 3, q = 7, dt = 0.01, g_j = 3 + 2 j (quindi S = 5 + 9 + 13 + 17 = 44), T_0 = 20 al centro.
// T_inf = 133.75 / 12 = 11.1458. Con theta = 1: gamma = 2 / 2.96 = 0.67568 e dopo un passo T = 11.1458 + 0.67568 (20 -
// 11.1458) = 17.128.
TEST(Transient2D, SingleUnknownFollowsTheScalarRecursion) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 2, 2);
    const double rho_c = 2.0;
    const double k = 3.0;
    const double q = 7.0;
    const double dt = 0.01;
    const Nodes fixed = {0, 1, 2, 3, 5, 6, 7, 8};
    DVector values;
    DVector T0(9, 0.0);
    for (std::size_t d : fixed) {
        values.push_back(3.0 + 2.0 * static_cast<double>(d));
        T0[d] = values.back();
    }
    T0[4] = 20.0;
    const DVector f = assemble_load_2d(mesh, q);

    double S = 0.0;
    for (std::size_t d : {1, 3, 5, 7}) {
        S += 3.0 + 2.0 * static_cast<double>(d);
    }
    const double t_inf = (q / 4.0 + k * S) / (4.0 * k);

    for (double theta : {1.0, 0.5, 0.3, 0.0}) {
        const double gamma = (rho_c - 32.0 * (1.0 - theta) * k * dt) / (rho_c + 32.0 * theta * k * dt);
        for (std::size_t steps : {1, 6}) {
            const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            ASSERT_EQ(T.size(), 9u);
            EXPECT_NEAR(T[4], t_inf + std::pow(gamma, steps) * (20.0 - t_inf), 1e-9)
                << "theta = " << theta << ", passi = " << steps;
            for (std::size_t q2 = 0; q2 < fixed.size(); ++q2) {
                EXPECT_EQ(T[fixed[q2]], values[q2]) << "nodo fissato " << fixed[q2];
            }
        }
    }
}

// ESEMPIO 97. Il confronto con un oracolo: lo stesso schema calcolato con matrici piene e l'eliminazione di Gauss
// (dense_transient), su una mesh storta di 30 nodi, con una sorgente che scalda in alcuni nodi e raffredda in altri,
// con tutto il bordo e un nodo interno fissati e con una T0 a caso (anche sui nodi fissati, dove NON vale il valore
// imposto: il primo passo usa T0 com'e' e il risultato dei nodi fissati e' comunque il valore imposto, ESATTAMENTE,
// anche dopo un solo passo). Quattro valori di theta, con 1 e con 10 passi; dt = 0.002 e' abbastanza piccolo anche
// per theta = 0.
TEST(Transient2D, MatchesADenseSolutionOnAnIrregularMesh) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 5, 4, 11);
    const std::size_t n = mesh.num_nodes();
    Nodes fixed = boundary_nodes(mesh);
    fixed.push_back(15);  // un nodo interno (colonna 3, riga 2 della griglia 6 x 5)
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(30.0 + 20.0 * std::cos(0.7 * static_cast<double>(q)));
    }
    const DVector f = mixed_load(n);
    DVector T0(n);
    unsigned long long state = 123;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 100.0 * next_random(state);
    }
    const double rho_c = 1.7;
    const double k = 2.3;
    const double dt = 0.002;
    for (double theta : {1.0, 0.5, 0.25, 0.0}) {
        for (std::size_t steps : {1, 10}) {
            const DVector got = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            const DVector want = dense_transient(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            const std::string what = "theta = " + std::to_string(theta) + ", passi = " + std::to_string(steps);
            expect_vector(got, want, 1e-6, what);
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                EXPECT_EQ(got[fixed[q]], values[q]) << what << ", nodo fissato " << fixed[q];
            }
        }
    }
}

// ESEMPIO 98. Lo stato stazionario e' un punto fisso. Sia T_s una temperatura qualunque e f = K T_s: allora T_s e' lo
// stato stazionario del problema con le temperature imposte uguali a T_s sui nodi fissati (K T_s = f). Se la lastra
// parte da T_s, un passo di qualunque schema (qualunque theta e qualunque dt) la lascia dov'e':
//     (M + theta dt K) T_s = (M - (1 - theta) dt K) T_s + dt K T_s     perche'  K T_s = f
// e quindi dopo 5 passi T_s non e' cambiata. Se il secondo membro avesse un dt di meno, o di troppo, o un segno
// sbagliato, la lastra si muoverebbe. Con theta < 1/2 il dt deve essere piccolo (stabilita'), altrimenti l'errore di
// arrotondamento crescerebbe a ogni passo.
TEST(Transient2D, TheSteadyStateIsAFixedPoint) {
    const Mesh2D mesh = perturbed_rectangle(1.5, 1.0, 6, 4, 7);
    const std::size_t n = mesh.num_nodes();
    const double rho_c = 2.2;
    const double k = 1.9;
    DVector Ts(n);
    for (std::size_t i = 0; i < n; ++i) {
        Ts[i] = 40.0 + 15.0 * std::sin(0.9 * static_cast<double>(i)) + 0.3 * static_cast<double>(i);
    }
    Nodes fixed = boundary_nodes(mesh);
    fixed.push_back(17);  // un nodo interno (colonna 3, riga 2 della griglia 7 x 5)
    DVector values;
    for (std::size_t d : fixed) {
        values.push_back(Ts[d]);
    }
    const DVector f = assemble_stiffness_2d(mesh, k) * Ts;

    const double cases[6][2] = {{1.0, 0.5}, {1.0, 20.0}, {0.5, 0.5}, {0.5, 20.0}, {0.3, 0.002}, {0.0, 0.001}};
    for (const auto& c : cases) {
        const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, Ts, c[1], 5, c[0]);
        expect_vector(T, Ts, 1e-6, "theta = " + std::to_string(c[0]) + ", dt = " + std::to_string(c[1]));
    }
}

// ESEMPIO 99. Il calore arriva allo stato stazionario. Una lastra a 20 gradi, con il bordo a temperature diverse e
// una sorgente che dipende dal punto: con Eulero all'indietro (stabile per ogni dt) e passi lunghi (dt = 0.5, 40
// passi, cioe' 20 secondi, molti tempi di rilassamento) la temperatura diventa quella che da' direttamente
// solve_steady_2d, con lo stesso k, la stessa f e le stesse temperature imposte. (rho_c non conta nello stato
// stazionario: M sparisce quando T non cambia piu'.) Se f entrasse nel passo senza dt, o con un peso sbagliato, lo
// stato finale sarebbe un altro.
TEST(Transient2D, BackwardEulerRelaxesToTheSteadyState) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 6, 6, 21);
    const double rho_c = 1.5;
    const double k = 0.7;
    const DVector f = assemble_load_2d(mesh, [](double x, double y) { return 6.0 + 2.0 * x - 3.0 * y; });
    const Nodes fixed = boundary_nodes(mesh);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(20.0 + 10.0 * std::sin(0.5 * static_cast<double>(q)));
    }
    const DVector steady = solve_steady_2d(mesh, k, f, fixed, values);
    const DVector T0(mesh.num_nodes(), 20.0);
    const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, 0.5, 40, 1.0);
    expect_vector(T, steady, 1e-6, "dopo 40 passi");
}

// ESEMPIO 100. Una lastra isolata: nessun nodo fissato (fixed_nodes vuoto), il calore non esce. Le colonne di K
// sommano a 0 (una temperatura costante non fa passare calore), quindi sommando le righe del sistema 1^T K sparisce e
// il calore totale H = 1^T M T cambia solo per la sorgente:
//     H_new = H_old + dt (somma dei numeri di f)        e la somma di f e' (sorgente q) x (area della lastra).
// Senza sorgente il calore totale resta lo stesso, per ogni theta. Con la sorgente uniforme q cresce di dt q area a
// ogni passo. Infine con Eulero all'indietro e passi lunghi la lastra si porta alla stessa temperatura dappertutto,
// quella che ha lo stesso calore: H / (rho_c area), perche' 1^T M 1 = rho_c area.
TEST(Transient2D, InsulatedPlateConservesTheHeat) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 6, 4, 3);
    const std::size_t n = mesh.num_nodes();
    const double rho_c = 1.4;
    const double k = 0.8;
    double area = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        area += mesh.area(e);
    }
    DVector T0(n);
    unsigned long long state = 77;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 100.0 * next_random(state);
    }
    const double heat0 = total_heat(mesh, rho_c, T0);
    const Nodes none;
    const DVector no_values;

    for (double theta : {1.0, 0.5, 0.25, 0.0}) {
        const double dt = 0.003;
        const DVector cold = solve_transient_2d(mesh, rho_c, k, DVector(n, 0.0), none, no_values, T0, dt, 8, theta);
        EXPECT_NEAR(total_heat(mesh, rho_c, cold), heat0, 1e-9 * heat0) << "senza sorgente, theta = " << theta;

        const double q = 3.0;
        const DVector f = assemble_load_2d(mesh, q);
        const DVector hot = solve_transient_2d(mesh, rho_c, k, f, none, no_values, T0, dt, 8, theta);
        EXPECT_NEAR(total_heat(mesh, rho_c, hot), heat0 + 8.0 * dt * q * area, 1e-9 * heat0)
            << "con la sorgente, theta = " << theta;
    }

    const DVector settled = solve_transient_2d(mesh, rho_c, k, DVector(n, 0.0), none, no_values, T0, 5.0, 40, 1.0);
    const double mean = heat0 / (rho_c * area);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(settled[i], mean, 1e-6) << "nodo " << i;
    }
}

// ESEMPIO 101. Zero passi: il risultato e' T0, UGUALE (EXPECT_EQ sul vettore, bit per bit), anche sui nodi fissati
// dove T0 non vale il valore imposto: il tempo non e' passato e non c'e' niente da imporre.
TEST(Transient2D, ZeroStepsReturnTheInitialTemperature) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 3, 3);
    const std::size_t n = mesh.num_nodes();
    DVector T0(n);
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 1.0 + 0.37 * static_cast<double>(i);
    }
    const Nodes fixed = boundary_nodes(mesh);
    const DVector values(fixed.size(), 100.0);
    const DVector f = assemble_load_2d(mesh, 2.0);
    for (double theta : {0.0, 0.5, 1.0}) {
        EXPECT_EQ(solve_transient_2d(mesh, 1.5, 0.9, f, fixed, values, T0, 0.1, 0, theta), T0)
            << "con nodi fissati, theta = " << theta;
        EXPECT_EQ(solve_transient_2d(mesh, 1.5, 0.9, f, Nodes(), DVector(), T0, 0.1, 0, theta), T0)
            << "senza nodi fissati, theta = " << theta;
    }
}

// ESEMPIO 102. Dopo ogni passo l'unico "stato" della lastra e' la sua temperatura T: i dati (f, le temperature
// imposte, dt) non cambiano nel tempo. Quindi 7 passi sono 3 passi seguiti da altri 4 passi che partono dal risultato
// dei primi 3 (sui nodi fissati vale gia' il valore imposto): lo stesso vettore, a meno dell'errore del CG. Se il
// secondo membro usasse T0 invece della T del passo prima, o se la funzione si ricordasse qualcosa da una chiamata
// all'altra, i due risultati sarebbero diversi.
TEST(Transient2D, SevenStepsAreThreeStepsFollowedByFour) {
    const Mesh2D mesh = perturbed_rectangle(1.5, 1.0, 4, 3, 5);
    const std::size_t n = mesh.num_nodes();
    Nodes fixed = boundary_nodes(mesh);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(10.0 + 3.0 * static_cast<double>(q % 4));
    }
    const DVector f = mixed_load(n);
    DVector T0(n);
    unsigned long long state = 9;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 50.0 * next_random(state);
    }
    for (double theta : {1.0, 0.5, 0.0}) {
        const DVector all = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, T0, 0.002, 7, theta);
        const DVector first = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, T0, 0.002, 3, theta);
        const DVector second = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, first, 0.002, 4, theta);
        expect_vector(second, all, 1e-6, "theta = " + std::to_string(theta));
    }
}

// ESEMPIO 103. La stabilita'. Senza sorgente e con la temperatura 0 imposta su tutto il bordo la lastra si raffredda e
// la sua energia E = sqrt(T^T M T) non puo' crescere. Ogni modo proprio della lastra si raffredda con una velocita'
// lambda >= 0 e un passo ne moltiplica l'ampiezza per
//     1 / (1 + dt lambda)                              Eulero all'indietro
//     (1 - dt lambda / 2) / (1 + dt lambda / 2)        Crank-Nicolson
//     1 - dt lambda                                    Eulero in avanti
// I primi due stanno SEMPRE fra -1 e 1, qualunque sia dt: l'energia non cresce mai, nemmeno con un passo enorme
// (dt = 100 con una lastra di lato 1 e k = rho_c = 1). Il terzo sta fra -1 e 1 solo se dt lambda <= 2: con dt = 0.003
// la lastra di 4 x 4 celle si raffredda, con dt = 0.1 il modo piu' veloce e' moltiplicato ogni volta per un numero
// di modulo piu' grande di 1, la lastra "esplode" e l'energia cresce a ogni passo. Il test controlla le due cose.
TEST(Transient2D, ImplicitSchemesNeverGainEnergyAndExplicitOneBlowsUp) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 4, 4);
    const std::size_t n = mesh.num_nodes();
    const Nodes fixed = boundary_nodes(mesh);
    const DVector values(fixed.size(), 0.0);
    const DVector f(n, 0.0);
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : fixed) {
        on_boundary[d] = true;
    }
    DVector T0(n, 0.0);
    unsigned long long state = 5;
    for (std::size_t i = 0; i < n; ++i) {
        if (!on_boundary[i]) {
            T0[i] = 100.0 * next_random(state);
        }
    }
    const CSR M = assemble_mass_2d(mesh, 1.0);
    const double energy0 = std::sqrt(dot(T0, M * T0));

    const double cases[7][2] = {{1.0, 0.01}, {1.0, 1.0},   {1.0, 100.0}, {0.5, 0.01},
                                {0.5, 1.0},  {0.5, 100.0}, {0.0, 0.003}};
    for (const auto& c : cases) {
        DVector T = T0;
        double energy = energy0;
        for (int step = 1; step <= 10; ++step) {
            T = solve_transient_2d(mesh, 1.0, 1.0, f, fixed, values, T, c[1], 1, c[0]);
            const double now = std::sqrt(dot(T, M * T));
            EXPECT_LE(now, energy * (1.0 + 1e-12)) << "theta = " << c[0] << ", dt = " << c[1] << ", passo " << step;
            energy = now;
        }
    }

    const DVector blown = solve_transient_2d(mesh, 1.0, 1.0, f, fixed, values, T0, 0.1, 3, 0.0);
    EXPECT_GT(std::sqrt(dot(blown, M * blown)), 100.0 * energy0) << "Eulero in avanti con dt troppo grande";
}

// ESEMPIO 104. Gli errori che sono colpa di chi chiama lanciano std::invalid_argument, e li' per li', anche con zero
// passi (non dopo aver fatto qualche passo): rho_c e k non > 0, dt non > 0, theta fuori da [0, 1] (NaN compreso: con
// un NaN i confronti sono tutti falsi, quindi si controlla "se NON e' dentro"), T0 o f con un numero di elementi
// diverso da quello dei nodi, fixed_values con un numero diverso da fixed_nodes. I valori al limite (theta = 0,
// theta = 1) e una lastra senza nodi fissati sono invece validi.
TEST(Transient2D, RejectsInvalidScalarsAndSizes) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    {
        Call ok;
        EXPECT_NO_THROW(ok.run()) << "chiamata valida";
        ok.theta = 0.0;
        EXPECT_NO_THROW(ok.run()) << "theta = 0";
        ok.theta = 1.0;
        EXPECT_NO_THROW(ok.run()) << "theta = 1";
        ok.fixed.clear();
        ok.values.clear();
        EXPECT_NO_THROW(ok.run()) << "nessun nodo fissato";
    }
    Call c;
    c.rho_c = 0.0;
    expect_invalid(c, "rho_c = 0");
    c = Call();
    c.rho_c = -1.0;
    expect_invalid(c, "rho_c < 0");
    c = Call();
    c.rho_c = nan;
    expect_invalid(c, "rho_c = NaN");
    c = Call();
    c.k = 0.0;
    expect_invalid(c, "k = 0");
    c = Call();
    c.k = -3.0;
    expect_invalid(c, "k < 0");
    c = Call();
    c.k = nan;
    expect_invalid(c, "k = NaN");
    c = Call();
    c.dt = 0.0;
    expect_invalid(c, "dt = 0");
    c = Call();
    c.dt = -0.01;
    expect_invalid(c, "dt < 0");
    c = Call();
    c.dt = nan;
    expect_invalid(c, "dt = NaN");
    c = Call();
    c.theta = -0.1;
    expect_invalid(c, "theta < 0");
    c = Call();
    c.theta = 1.1;
    expect_invalid(c, "theta > 1");
    c = Call();
    c.theta = nan;
    expect_invalid(c, "theta = NaN");
    c = Call();
    c.T0.pop_back();
    expect_invalid(c, "T0 con un numero in meno");
    c = Call();
    c.T0.push_back(20.0);
    expect_invalid(c, "T0 con un numero in piu'");
    c = Call();
    c.f.pop_back();
    expect_invalid(c, "f con un numero in meno");
    c = Call();
    c.f.push_back(1.0);
    expect_invalid(c, "f con un numero in piu'");
    c = Call();
    c.values.pop_back();
    expect_invalid(c, "un valore imposto in meno");
    c = Call();
    c.values.push_back(5.0);
    expect_invalid(c, "un valore imposto in piu'");
}

// ESEMPIO 105. I nodi sbagliati. Un nodo fissato che non esiste (9 con 9 nodi, o un numero enorme) o che compare due
// volte (anche con lo stesso valore) lancia std::invalid_argument, anche con zero passi: le matrici si costruiscono
// prima dei passi. E un nodo NON fissato che non sta in nessun triangolo (mesh con i nodi 0 e 5 fuori da ogni
// triangolo) ha la riga di M e di K tutta 0 e non si puo' risolvere: invalid_argument, dal primo passo. Fissandolo
// la lastra si calcola e coincide con l'oracolo.
TEST(Transient2D, RejectsInvalidFixedNodesAndNodesOutsideEveryTriangle) {
    Call c;
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 9};
    expect_invalid(c, "nodo fissato 9 su 9 nodi");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, std::numeric_limits<std::size_t>::max()};
    expect_invalid(c, "nodo fissato enorme");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 7};
    expect_invalid(c, "nodo fissato due volte, stesso valore");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 0};
    c.values = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0};
    expect_invalid(c, "nodo fissato due volte, valori diversi");

    Call d;
    d.mesh = mesh_with_nodes_outside_every_triangle();
    d.f = DVector(6, 1.0);
    d.T0 = DVector(6, 20.0);
    d.fixed = {1, 2, 3, 4};
    d.values = DVector(4, 5.0);
    d.n_steps = 3;
    EXPECT_THROW(d.run(), std::invalid_argument) << "nodi 0 e 5 liberi e fuori da ogni triangolo";

    d.fixed = {0, 1, 5};
    d.values = {1.0, 2.0, 3.0};
    const DVector want =
        dense_transient(d.mesh, d.rho_c, d.k, d.f, d.fixed, d.values, d.T0, d.dt, d.n_steps, d.theta);
    expect_vector(d.run(), want, 1e-9, "nodi 0 e 5 fissati");
}

// ESEMPIO 106. Un problema che il gradiente coniugato non risolve: dei NaN nei dati (nella sorgente, in T0 o in una
// temperatura imposta) rendono il secondo membro un NaN e il CG non converge. La funzione non restituisce dei NaN
// come se niente fosse: lancia std::runtime_error.
TEST(Transient2D, ThrowsWhenTheLinearSolverCannotConverge) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Call c;
    c.f[4] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN nella sorgente";
    c = Call();
    c.T0[4] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN in T0";
    c = Call();
    c.values[0] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN in una temperatura imposta";
}