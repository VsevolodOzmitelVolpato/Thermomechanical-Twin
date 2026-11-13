#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- Dalle forze agli spostamenti: risolvere K u = F ---------------------------------------------------------------
//
// Con la matrice K (assemble_elasticity_stiffness_2d) e le forze F (per esempio quelle termiche,
// assemble_thermal_load_2d) gli spostamenti della lastra, u, sono la soluzione di K u = F. Per il calore e' lo stesso
// passo di solve_steady_2d, con una differenza che cambia tutto: K da sola NON si puo' risolvere. Un corpo libero si
// sposta e ruota senza deformarsi, e i moti rigidi (in 2D sono 3: due traslazioni e una rotazione) hanno K u = 0:
// se u risolve K u = F, la risolve anche u + (un moto rigido qualunque). Per avere UNA soluzione servono dei vincoli,
// cioe' degli spostamenti IMPOSTI su alcuni gradi di liberta': almeno 3, e messi in modo da bloccare tutti e 3 i moti.
//
// solve_displacement_2d(mesh, D, F, fixed_dofs, fixed_values) restituisce u (2 n numeri: u del nodo I al numero 2 I e
// v al 2 I + 1). fixed_dofs sono numeri di gradi di liberta', non di nodi: per bloccare del tutto il nodo I servono
// 2 I e 2 I + 1. Come per il calore il sistema e' A u = b, con A di dirichlet_matrix_triplets_2d e b di
// dirichlet_rhs_2d (le stesse due funzioni, con i gradi di liberta' al posto dei nodi), risolto con il gradiente
// coniugato con il precondizionatore di Jacobi (tolleranza 1e-12, al piu' 10 volte il numero dei gradi di liberta' di
// iterazioni), partendo dagli spostamenti imposti: cosi' i gradi di liberta' fissati escono con il valore imposto
// ESATTAMENTE. Le novita' rispetto al calore sono due.
//   - K ed F si DIVIDONO per il piu' grande valore assoluto delle terne di K, prima del CG (u non cambia). Senza, un
//     materiale rigido (acciaio, E = 210e9: K ha numeri di ordine 1e11) fa rifiutare K al CG, che controlla la
//     simmetria con una tolleranza assoluta di 1e-12 e trova gli arrotondamenti (1e-16 relativi, quindi 1e-5
//     assoluti); e uno molto soffice (E = 1e-12) fa fermare il CG subito e restituire una soluzione sbagliata
//     (ESEMPIO 176).
//   - Alla fine si ricalcola il residuo vero b - A u e si lancia std::runtime_error se norm(b - A u) > 1e-6 norm(b): il
//     CG aggiorna il suo residuo con una formula, e su un sistema impossibile (vincoli che lasciano libera una
//     rotazione e un carico che la fa girare) lo porta a 0 mentre u diventa enorme (ESEMPIO 179).
//
// Cosa deve fare, ed e' quello che i test controllano:
//   - sul quadrato unitario tre problemi si risolvono a mano (ESEMPIO 168);
//   - la trazione di un rettangolo ha una soluzione di libro, per ogni E e nu (ESEMPIO 169);
//   - gli spostamenti imposti escono identici, in qualunque ordine (ESEMPIO 170);
//   - un campo lineare imposto sul bordo e' riprodotto anche dentro, su ogni tipo di mesh (ESEMPIO 171);
//   - un corpo libero scaldato in modo uniforme si dilata come si sa (ESEMPIO 172), e se e' incastrato su tutto il
//     bordo non si muove (ESEMPIO 173);
//   - u soddisfa davvero K u = F, per qualunque carico e spostamento imposto (ESEMPIO 174), ed e' lineare in tutti e
//     due (ESEMPIO 175);
//   - il risultato non dipende dalle unita' del modulo di Young (ESEMPIO 176);
//   - i nodi fuori da ogni triangolo vanno fissati (ESEMPIO 177);
//   - dati sbagliati lanciano std::invalid_argument (ESEMPIO 178) e i sistemi senza soluzione std::runtime_error
//     (ESEMPIO 179);
//   - su una mesh grande il CG converge e il risultato e' esatto (ESEMPIO 180).

namespace {

// Un rettangolo lx x ly diviso in nx x ny celle, come in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Sei rettangoli di forme molto diverse (gli stessi dei test dell'assemblaggio).
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},    // il quadrato unitario: due triangoli
            {2.0, 1.0, 2, 1},    // due celle
            {3.0, 1.5, 4, 3},    // celle 0.75 x 0.5
            {1.0, 1.0, 6, 6},    // 72 triangoli
            {0.2, 0.05, 5, 2},   // piccolo (20 cm per 5 cm, in metri)
            {10.0, 0.1, 1, 7}};  // molto schiacciato: celle 10 x 0.0143
}

// Il quadrato unitario diviso lungo la diagonale da (1, 0) a (0, 1): nodi 0 = (0, 0), 1 = (1, 0), 2 = (0, 1),
// 3 = (1, 1) e triangoli (0, 1, 2) e (1, 3, 2), tutti e due con area 1/2.
Mesh2D unit_square() {
    return Mesh2D({0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {1, 3, 2}});
}

// Il triangolo grande con il nodo 3 dentro (area 9/2), che non e' un rettangolo.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// Un quadrilatero irregolare con un nodo dentro: 0 = (0, 0), 1 = (4, 0), 2 = (3, 3), 3 = (-1, 2) e, dentro,
// 4 = (1, 0.7). Quattro triangoli attorno al nodo 4, tutti di forma e area diverse.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Due triangoli di area DIVERSA: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); (0, 1, 2) ha A = 4 e (1, 3, 2) ha
// A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

// Sei nodi e due soli triangoli, (1, 2, 3) e (1, 3, 4): i nodi 0 = (5, 5) e 5 = (7, 7) non stanno in nessun
// triangolo. Gli altri quattro sono il quadrato unitario: 1 = (0, 0), 2 = (1, 0), 3 = (1, 1), 4 = (0, 1).
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// Due triangoli SEPARATI, senza nodi in comune: (0, 1, 2) e (3, 4, 5). Sono due corpi, ognuno con i suoi 3 moti rigidi.
Mesh2D two_separate_triangles() {
    return Mesh2D({0.0, 1.0, 0.0, 3.0, 4.0, 3.0}, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, {{0, 1, 2}, {3, 4, 5}});
}

// Le mesh in cui ogni nodo sta in almeno un triangolo: sei rettangoli e tre mesh che non sono rettangoli.
std::vector<Mesh2D> sample_meshes() {
    std::vector<Mesh2D> meshes;
    for (const Grid& g : sample_grids()) {
        meshes.push_back(rectangle_mesh(g.lx, g.ly, g.nx, g.ny));
    }
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());
    return meshes;
}

// Materiali a mano (3 x 3, indici xx, yy, xy), simmetrici e definiti positivi.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 general() {  // tutti i numeri diversi, anche l'accoppiamento con il taglio
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}
Matrix3 steel() {  // un materiale vero, rigido: numeri di ordine 1e11
    return plane_strain_matrix(210e9, 0.3);
}

// Un materiale per la forza termica: la matrice D, il coefficiente di Poisson nu e il coefficiente di dilatazione alpha
// (D e nu arrivano separati alle funzioni, quindi si provano anche materiali non fisici: nu e alpha negativi).
struct Material {
    Matrix3 D;
    double nu;
    double alpha;
};

std::vector<Material> sample_materials() {
    return {{hooke_quarter(), 0.25, 1.0e-3},  // E = 1, nu = 1/4
            {general(), -0.3, -2.0e-3},       // D qualunque, nu negativo e alpha negativo
            {steel(), 0.3, 12.0e-6}};         // acciaio: E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin
}

// La temperatura di riferimento, quella senza dilatazione termica (20 gradi centigradi in kelvin).
const double reference_temperature = 293.15;

// Il piu' grande valore assoluto di un vettore.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
}

// La dimensione maggiore della mesh (il piu' grande fra la larghezza e l'altezza).
double mesh_size(const Mesh2D& mesh) {
    double x_min = mesh.x(0);
    double x_max = mesh.x(0);
    double y_min = mesh.y(0);
    double y_max = mesh.y(0);
    for (std::size_t I = 1; I < mesh.num_nodes(); ++I) {
        x_min = std::min(x_min, mesh.x(I));
        x_max = std::max(x_max, mesh.x(I));
        y_min = std::min(y_min, mesh.y(I));
        y_max = std::max(y_max, mesh.y(I));
    }
    return std::max(x_max - x_min, y_max - y_min);
}

// Il nodo piu' lontano dal nodo p.
std::size_t farthest_node(const Mesh2D& mesh, std::size_t p) {
    std::size_t q = p;
    double best = -1.0;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double d = std::hypot(mesh.x(I) - mesh.x(p), mesh.y(I) - mesh.y(p));
        if (d > best) {
            best = d;
            q = I;
        }
    }
    return q;
}

// Tre gradi di liberta' che bloccano i tre moti rigidi, senza altro: u e v del nodo p (le due traslazioni) e una
// componente del nodo q piu' lontano da p (la rotazione). Una rotazione di theta attorno a p sposta q di
// theta (-(y_q - y_p), x_q - x_p): si fissa v se x_q - x_p e' la differenza piu' grande in valore assoluto (cosi' la
// rotazione dovrebbe spostare v di un numero diverso da 0), altrimenti u.
std::vector<std::size_t> isostatic_dofs(const Mesh2D& mesh, std::size_t p) {
    const std::size_t q = farthest_node(mesh, p);
    const double dx = mesh.x(q) - mesh.x(p);
    const double dy = mesh.y(q) - mesh.y(p);
    return {2 * p, 2 * p + 1, std::fabs(dx) >= std::fabs(dy) ? 2 * q + 1 : 2 * q};
}

// Quattro gradi di liberta': u e v di p e u e v del nodo piu' lontano da p (due nodi bloccati del tutto).
std::vector<std::size_t> two_node_clamp(const Mesh2D& mesh, std::size_t p) {
    const std::size_t q = farthest_node(mesh, p);
    return {2 * p, 2 * p + 1, 2 * q, 2 * q + 1};
}

// I gradi di liberta' (u e v) di tutti i nodi del bordo.
std::vector<std::size_t> boundary_dofs(const Mesh2D& mesh) {
    std::vector<std::size_t> dofs;
    for (std::size_t I : boundary_nodes(mesh)) {
        dofs.push_back(2 * I);
        dofs.push_back(2 * I + 1);
    }
    return dofs;
}

// I valori di u nei gradi di liberta' dofs, nello stesso ordine.
std::vector<double> pick(const std::vector<double>& u, const std::vector<std::size_t>& dofs) {
    std::vector<double> values;
    for (std::size_t d : dofs) {
        values.push_back(u[d]);
    }
    return values;
}

// Un campo di spostamento lineare:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// In ogni punto ha la stessa deformazione (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// I 2 n numeri (u_0, v_0, u_1, v_1, ...) del campo f nei nodi della mesh: u del nodo I al numero 2 I, v al 2 I + 1.
std::vector<double> nodal_displacements(const Mesh2D& mesh, const LinearField& f) {
    std::vector<double> u(2 * mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double x = mesh.x(I);
        const double y = mesh.y(I);
        u[2 * I] = f.a0 + f.a1 * x + f.a2 * y;
        u[2 * I + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u;
}

// I moti rigidi: due traslazioni, una rotazione attorno all'origine (u = -y, v = x) e una combinazione.
std::vector<LinearField> rigid_motions() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},     // traslazione in x
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},     // traslazione in y
            {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},    // rotazione
            {0.3, 0.0, -0.7, -1.2, 0.7, 0.0}};  // traslazione e rotazione insieme
}

// I sei campi "puri" (u = x, v = y, u = y, v = x e le due traslazioni) piu' uno con tutti i numeri diversi da 0 e
// deformazioni piccole, come quelle vere (1e-3).
std::vector<LinearField> strain_fields() {
    return {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.3, 1.0e-3, -2.0e-3, -0.2, 3.0e-3, 5.0e-4}};
}

// Forze (o numeri qualunque) che cambiano da un grado di liberta' all'altro senza regole semplici, di ampiezza
// amplitude.
std::vector<double> wavy_vector(std::size_t size, double amplitude) {
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double t = static_cast<double>(i);
        v[i] = amplitude * (std::sin(1.3 * t + 0.4) + 0.3 * std::cos(2.9 * t));
    }
    return v;
}

// Un problema con la soluzione esatta nota: le forze, i gradi di liberta' fissati con i loro spostamenti imposti
// (presi dalla soluzione esatta) e la soluzione esatta stessa (2 n numeri).
struct Problem {
    std::vector<double> F;
    std::vector<std::size_t> dofs;
    std::vector<double> values;
    std::vector<double> exact;
};

// Un corpo LIBERO scaldato in modo uniforme di dT (T = T_ref + dT in tutti i nodi) e solo bloccato nei tre moti
// rigidi (isostatic_dofs attorno al nodo p). Non ci sono tensioni: il corpo si dilata di epsilon_th = c (1, 1, 0) con
// c = (1 + nu) alpha dT, e la soluzione esatta e' u = (a0 + c x, b0 + c y) dove (a0, b0) e' lo spostamento di una
// traslazione che i vincoli impongono (con (a0, b0) = (0, 0) e p = 0 i valori imposti sono tutti 0). Con questo
// spostamento B u = epsilon_th in ogni triangolo e quindi K u = F per qualunque D.
Problem heated_free_body(const Mesh2D& mesh, const Material& m, double dT, std::size_t p, double a0, double b0) {
    const double c = (1.0 + m.nu) * m.alpha * dT;
    Problem pr;
    pr.F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
    pr.exact = nodal_displacements(mesh, LinearField{a0, c, 0.0, b0, 0.0, c});
    pr.dofs = isostatic_dofs(mesh, p);
    pr.values = pick(pr.exact, pr.dofs);
    return pr;
}

// La trazione uniforme di un rettangolo (lati lx x ly, E, nu, plane_strain_matrix): il lato sinistro (x = 0) non puo'
// spostarsi in x (u = 0 su tutti i suoi nodi) e il nodo in basso a sinistra non puo' spostarsi in y (v = 0, cosi' la
// lastra non scivola in y); sul lato destro agisce la tensione sigma, cioe' su ogni nodo la forza sigma per la
// lunghezza di bordo che gli spetta (meta' della distanza dal nodo sotto piu' meta' di quella dal nodo sopra).
// Con sigma_xx = sigma e sigma_yy = 0, in deformazione piana (epsilon_zz = 0, quindi sigma_zz = nu sigma_xx):
//     epsilon_xx = (1 - nu^2) sigma / E,        epsilon_yy = -nu (1 + nu) sigma / E,
// e la soluzione esatta e' u = epsilon_xx x, v = epsilon_yy y: un campo lineare, che gli elementi riproducono.
Problem stretched_rectangle(const Mesh2D& mesh, double lx, double E, double nu, double sigma) {
    std::vector<std::pair<double, std::size_t>> left;   // (y, nodo)
    std::vector<std::pair<double, std::size_t>> right;  // (y, nodo)
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        if (mesh.x(I) <= 1e-9 * lx) {
            left.push_back({mesh.y(I), I});
        } else if (mesh.x(I) >= (1.0 - 1e-9) * lx) {
            right.push_back({mesh.y(I), I});
        }
    }
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());

    Problem pr;
    pr.F.assign(2 * mesh.num_nodes(), 0.0);
    for (std::size_t j = 0; j < right.size(); ++j) {
        const double below = (j == 0) ? right[j].first : right[j - 1].first;
        const double above = (j + 1 == right.size()) ? right[j].first : right[j + 1].first;
        pr.F[2 * right[j].second] = sigma * 0.5 * (above - below);
    }
    for (const auto& node : left) {
        pr.dofs.push_back(2 * node.second);
    }
    pr.dofs.push_back(2 * left.front().second + 1);
    pr.values.assign(pr.dofs.size(), 0.0);

    const double exx = (1.0 - nu * nu) * sigma / E;
    const double eyy = -nu * (1.0 + nu) * sigma / E;
    pr.exact = nodal_displacements(mesh, LinearField{0.0, exx, 0.0, 0.0, 0.0, eyy});
    return pr;
}

// Confronta due vettori elemento per elemento, con una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "grado di liberta' " << i;
    }
}

}  // namespace

// ESEMPIO 168. A mano. Il quadrato unitario ha 4 nodi, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), e due triangoli,
// (0, 1, 2) e (1, 3, 2). Il materiale ha E = 1 e nu = 1/4 (lambda = mu = 0.4, quindi
// D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]). I gradi di liberta' sono (u0, v0, u1, v1, u2, v2, u3, v3), cioe'
// i numeri 0, ..., 7. Il campo di spostamento di ciascuno dei tre problemi e' LINEARE, e un campo lineare gli elementi
// lo riproducono esattamente: la soluzione e' quella del corpo continuo.
//   (a) Trazione. Il lato sinistro non si sposta in x (u0 = u2 = 0, gradi di liberta' 0 e 4) e il nodo 0 non si
//       sposta in y (v0 = 0, il numero 1: senza, la lastra scivolerebbe in y). Sui due nodi del lato destro, 1 e 3,
//       tira una forza di 0.5 in x (i numeri 2 e 6): in tutto 1, cioe' una tensione sigma_xx = 1 su un lato lungo 1.
//       Il corpo continuo ha tensione (1, 0, 0), e la deformazione si ricava da sigma = D epsilon: con sigma_yy = 0,
//       0.4 eps_xx + 1.2 eps_yy = 0, cioe' eps_yy = -eps_xx / 3, e sigma_xx = 1.2 eps_xx + 0.4 eps_yy = (1.2 - 0.4 /
//       3) eps_xx = (16 / 15) eps_xx = 1: eps_xx = 15/16 e eps_yy = -5/16. Gli spostamenti sono u = 15/16 x e v =
//       -5/16 y:  u = (u0, v0, ..., v3) = (0, 0, 15/16, 0, 0, -5/16, 15/16, -5/16). La lastra si allunga di 15/16 e,
//       per il coefficiente di Poisson (nu / (1 - nu) = 1/3 in deformazione piana), si restringe di un terzo di
//       questo.
//   (b) Dilatazione termica. Con T = (7, 7, 7, 7), T_ref = 5 e alpha = 1 la deformazione termica e' epsilon_th =
//       (1 + nu) alpha dT (1, 1, 0) = 1.25 * 2 (1, 1, 0) = 2.5 (1, 1, 0) e le forze (ESEMPIO 157 del passo prima) sono
//       F = (-2, -2, 2, -2, -2, 2, 2, 2). Il corpo e' libero: servono solo i tre vincoli u0 = v0 = v1 = 0 (i numeri 0,
//       1 e 3; v1 e non u1, perche' i nodi 0 e 1 hanno la stessa y e u1 non impedirebbe la rotazione attorno al nodo
//       0). Il corpo si dilata senza tensioni di 2.5 x e 2.5 y: u = (0, 0, 2.5, 0, 0, 2.5, 2.5, 2.5).
//   (c) Spostamento imposto invece della forza. Come (a), ma senza forze (F = 0) e con i due nodi del lato destro
//       spostati a forza di 0.1 in x (u1 = u3 = 0.1, i numeri 2 e 6: sono 5 gradi di liberta' fissati, 0, 1, 4, 2 e 6).
//       Restano liberi v1, v2 e v3 (i numeri 3, 5 e 7). Il campo u = 0.1 x, v = -(0.1 / 3) y ha tensione
//       sigma_yy = 0.4 * 0.1 + 1.2 * (-1/30) = 0: nessuna forza verticale, quindi i gradi di liberta' verticali liberi
//       sono in equilibrio, e la soluzione e' u = (0, 0, 0.1, 0, 0, -1/30, 0.1, -1/30).
TEST(ElasticitySolve2d, ThreeHandSolvedProblemsOnTheUnitSquare) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();

    const std::vector<double> tension = {0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0};
    expect_vectors_near(solve_displacement_2d(mesh, D, tension, {0, 1, 4}, {0.0, 0.0, 0.0}),
                        {0.0, 0.0, 0.9375, 0.0, 0.0, -0.3125, 0.9375, -0.3125}, 1e-12);

    const std::vector<double> heating = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    expect_vectors_near(solve_displacement_2d(mesh, D, heating, {0, 1, 3}, {0.0, 0.0, 0.0}),
                        {0.0, 0.0, 2.5, 0.0, 0.0, 2.5, 2.5, 2.5}, 1e-12);

    const std::vector<double> no_forces(8, 0.0);
    expect_vectors_near(solve_displacement_2d(mesh, D, no_forces, {0, 1, 4, 2, 6}, {0.0, 0.0, 0.0, 0.1, 0.1}),
                        {0.0, 0.0, 0.1, 0.0, 0.0, -1.0 / 30.0, 0.1, -1.0 / 30.0}, 1e-12);
}

// ESEMPIO 169. La trazione di un rettangolo con la soluzione di libro (stretched_rectangle): sul lato destro la
// tensione sigma = 1e-3 E (una deformazione di 1e-3, come quelle vere), a sinistra u = 0, e nel nodo in basso a
// sinistra v = 0. In deformazione piana eps_xx = (1 - nu^2) sigma / E,        eps_yy = -nu (1 + nu) sigma / E, e gli
// spostamenti sono u = eps_xx x, v = eps_yy y. Si prova su sei rettangoli di forma diversa (anche molto schiacciati),
// con E = 1 e con E = 210e9 e con nu = 0, 1/4, 0.3, 0.4: tutti i nodi, non solo gli estremi. Con nu = 0 la lastra non
// si restringe (eps_yy = 0). Con E = 1, nu = 1/4 e sigma = 1 sul quadrato unitario sono i numeri dell'ESEMPIO 168
// (a).
TEST(ElasticitySolve2d, UniaxialTensionOfARectangleMatchesTheTextbookFormulas) {
    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (double E : {1.0, 210e9}) {
            for (double nu : {0.0, 0.25, 0.3, 0.4}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(E) + ", nu = " + std::to_string(nu));
                const Problem pr = stretched_rectangle(mesh, g.lx, E, nu, 1e-3 * E);
                const std::vector<double> got =
                    solve_displacement_2d(mesh, plane_strain_matrix(E, nu), pr.F, pr.dofs, pr.values);
                expect_vectors_near(got, pr.exact, 1e-9 * largest_entry(pr.exact));
            }
        }
    }
}

// ESEMPIO 170. Gli spostamenti imposti escono ESATTAMENTE come sono stati imposti (uguali bit per bit, non "quasi"),
// qualunque sia l'ordine in cui si elencano e anche quando sono tutti. Su una mesh di 4 x 3 celle (20 nodi, 40 gradi
// di liberta') si fissano otto gradi di liberta' sparsi, in disordine, con numeri qualunque (grandi, piccoli,
// negativi), fra i quali 0, 1 e 3, che bloccano i moti rigidi, e con delle forze qualunque sugli altri; poi la stessa
// cosa con le liste in ordine inverso (stesse coppie grado di liberta' - valore): il risultato e' lo stesso. Infine
// si fissano tutti i 40 gradi di liberta': il risultato e' l'elenco dei valori, e il sistema non ha niente da
// risolvere.
TEST(ElasticitySolve2d, ImposedDisplacementsComeBackExactlyInAnyOrder) {
    const Mesh2D mesh = rectangle_mesh(3.0, 1.5, 4, 3);
    const Matrix3 D = hooke_quarter();
    const std::size_t N = 2 * mesh.num_nodes();
    ASSERT_EQ(N, 40u);
    const std::vector<double> F = wavy_vector(N, 0.01);

    const std::vector<std::size_t> dofs = {17, 3, 0, 1, 38, 22, 9, 30};
    const std::vector<double> values = {0.1, -7.3, 2.5e-9, 123.456, -0.5, 1.0e-3, 4.0, -1.0e-7};
    const std::vector<double> u = solve_displacement_2d(mesh, D, F, dofs, values);
    ASSERT_EQ(u.size(), N);
    for (std::size_t q = 0; q < dofs.size(); ++q) {
        EXPECT_EQ(u[dofs[q]], values[q]) << "grado di liberta' " << dofs[q];
    }

    const std::vector<std::size_t> reversed_dofs(dofs.rbegin(), dofs.rend());
    const std::vector<double> reversed_values(values.rbegin(), values.rend());
    const std::vector<double> v = solve_displacement_2d(mesh, D, F, reversed_dofs, reversed_values);
    expect_vectors_near(v, u, 1e-12 * largest_entry(u));
    for (std::size_t q = 0; q < reversed_dofs.size(); ++q) {
        EXPECT_EQ(v[reversed_dofs[q]], reversed_values[q]) << "grado di liberta' " << reversed_dofs[q];
    }

    std::vector<std::size_t> all_dofs;
    for (std::size_t d = 0; d < N; ++d) {
        all_dofs.push_back((d * 7) % N);  // tutti i numeri da 0 a 39 (7 e 40 sono primi fra loro), in disordine
    }
    const std::vector<double> all_values = wavy_vector(N, 3.0);
    const std::vector<double> w = solve_displacement_2d(mesh, D, F, all_dofs, all_values);
    ASSERT_EQ(w.size(), N);
    for (std::size_t q = 0; q < N; ++q) {
        EXPECT_EQ(w[all_dofs[q]], all_values[q]) << "grado di liberta' " << all_dofs[q];
    }
}

// ESEMPIO 171. Il test della toppa (patch test), su ogni tipo di mesh. Con un campo di spostamento LINEARE u =
// a0 + a1 x + a2 y, v = b0 + b1 x + b2 y la deformazione e' la stessa in ogni punto, quindi anche la tensione e' la
// stessa dappertutto, e dentro il corpo (senza forze) e' in equilibrio. Allora imponendo il campo sui gradi di liberta'
// di tutti i nodi del bordo (boundary_nodes), con F = 0, la soluzione deve essere il campo stesso anche nei nodi
// interni, per qualunque materiale D e qualunque forma dei triangoli. Si prova su nove mesh (sei rettangoli, il
// triangolo con un nodo dentro, il quadrilatero irregolare, due triangoli di area diversa) con tre materiali (anche
// non fisico, e l'acciaio) e con undici campi: i moti rigidi, le sei deformazioni "pure" e un campo con deformazioni di
// 1e-3 e una traslazione di 0.3. Nelle mesh senza nodi interni tutto e' imposto e non c'e' niente da risolvere.
TEST(ElasticitySolve2d, ALinearDisplacementFieldIsReproducedOnEveryKindOfMesh) {
    std::vector<LinearField> fields = rigid_motions();
    for (const LinearField& f : strain_fields()) {
        fields.push_back(f);
    }
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = boundary_dofs(mesh);
        const std::vector<double> no_forces(2 * mesh.num_nodes(), 0.0);
        for (const Material& m : sample_materials()) {
            for (std::size_t fi = 0; fi < fields.size(); ++fi) {
                SCOPED_TRACE("mesh " + std::to_string(mi) + ", material with nu = " + std::to_string(m.nu) +
                             ", field " + std::to_string(fi));
                const std::vector<double> want = nodal_displacements(mesh, fields[fi]);
                const std::vector<double> got = solve_displacement_2d(mesh, m.D, no_forces, dofs, pick(want, dofs));
                expect_vectors_near(got, want, 1e-9 * largest_entry(want));
            }
        }
    }
}

// ESEMPIO 172. Un corpo libero scaldato in modo uniforme si dilata senza tensioni (heated_free_body). Con T = T_ref +
// dT in tutti i nodi la deformazione termica e' epsilon_th = c (1, 1, 0), c = (1 + nu) alpha dT, e il campo u = (a0 +
// c x, b0 + c y) ha B u = epsilon_th in ogni triangolo: K u = F per qualunque D e qualunque mesh (a0 e b0 sono una
// traslazione, imposta dai vincoli). I vincoli sono i tre di isostatic_dofs e prendono i valori della soluzione
// esatta: u e v di un nodo p e una componente di un altro. Si prova su nove mesh, tre materiali (anche non fisici: nu
// e alpha negativi) e due riscaldamenti (+100 e -60 kelvin), con p = nodo 0 e p = ultimo nodo, senza traslazione e
// con la traslazione (a0, b0) = (0.3, -0.2): 216 problemi. Con l'acciaio i numeri di K sono di ordine 1e11 (ESEMPIO
// 176).
TEST(ElasticitySolve2d, AUniformlyHeatedFreeBodyExpandsLikeTheExactField) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<std::pair<double, double>> shifts = {{0.0, 0.0}, {0.3, -0.2}};
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                for (std::size_t p : {static_cast<std::size_t>(0), mesh.num_nodes() - 1}) {
                    for (const auto& shift : shifts) {
                        SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu) +
                                     ", dT = " + std::to_string(dT) + ", node " + std::to_string(p) +
                                     ", shift " + std::to_string(shift.first));
                        const Problem pr = heated_free_body(mesh, m, dT, p, shift.first, shift.second);
                        const std::vector<double> got =
                            solve_displacement_2d(mesh, m.D, pr.F, pr.dofs, pr.values);
                        expect_vectors_near(got, pr.exact, 1e-9 * largest_entry(pr.exact));
                    }
                }
            }
        }
    }
}

// ESEMPIO 173. Il caso opposto: una lastra scaldata in modo uniforme e INCASTRATA su tutto il bordo (tutti i gradi di
// liberta' dei nodi del bordo a 0) non si muove: il calore vorrebbe dilatarla, ma non puo'. Le forze termiche (le
// stesse del corpo libero) si annullano sui nodi interni, che restano senza forze, e quelle sul bordo le prendono i
// vincoli. Gli spostamenti sono 0 in tutti i nodi (entro 1e-9 volte la dilatazione libera c L, dove c = (1 + nu)
// alpha dT e L e' la dimensione della lastra). E' il caso da cui nel passo dopo si calcolera' la tensione:
// spostamento 0 ma tensione -D epsilon_th. Nove mesh, tre materiali, due riscaldamenti.
TEST(ElasticitySolve2d, AFullyClampedHeatedPlateStaysAtRest) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = boundary_dofs(mesh);
        const std::vector<double> zeros(dofs.size(), 0.0);
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu) +
                             ", dT = " + std::to_string(dT));
                const std::vector<double> F =
                    assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                             std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                             reference_temperature);
                const double free_expansion = std::fabs((1.0 + m.nu) * m.alpha * dT) * mesh_size(mesh);
                const std::vector<double> got = solve_displacement_2d(mesh, m.D, F, dofs, zeros);
                expect_vectors_near(got, std::vector<double>(2 * mesh.num_nodes(), 0.0), 1e-9 * free_expansion);
            }
        }
    }
}

// ESEMPIO 174. u soddisfa davvero le equazioni K u = F, per qualunque carico e qualunque spostamento imposto: non
// solo i problemi con una soluzione nota in anticipo. Su ogni mesh e materiale si fissano i quattro gradi di liberta'
// di due nodi (two_node_clamp: il nodo 0 e il piu' lontano da lui), con spostamenti imposti diversi da 0 e un vettore
// F qualunque (wavy_vector, di un'ampiezza che sposta i nodi di circa 1e-2 volte le dimensioni). Il residuo K u - F
// deve essere 0 nei gradi di liberta' LIBERI (entro 1e-9 volte la scala dei numeri in gioco, cioe' il piu' grande
// numero di K per il piu' grande spostamento, piu' il piu' grande F); in quelli fissati e' la reazione del vincolo, e
// non c'e' niente da controllare. Il prodotto K u e' quello della K di assemble_elasticity_stiffness_2d, senza
// scalature.
TEST(ElasticitySolve2d, TheSolutionSatisfiesTheEquationsForAnyLoadAndAnyImposedValues) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = two_node_clamp(mesh, 0);
        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu));
            const CSR K = assemble_elasticity_stiffness_2d(mesh, m.D);
            const double k_scale = largest_entry(K.diag());
            const double size = mesh_size(mesh);
            const std::vector<double> F = wavy_vector(2 * mesh.num_nodes(), 1e-2 * k_scale * size);
            const std::vector<double> values = {0.003 * size, -0.002 * size, 0.001 * size, 0.004 * size};

            const std::vector<double> u = solve_displacement_2d(mesh, m.D, F, dofs, values);
            ASSERT_EQ(u.size(), F.size());
            const std::vector<double> Ku = K * u;
            std::vector<bool> is_fixed(u.size(), false);
            for (std::size_t d : dofs) {
                is_fixed[d] = true;
            }
            const double scale = k_scale * largest_entry(u) + largest_entry(F);
            for (std::size_t i = 0; i < u.size(); ++i) {
                if (!is_fixed[i]) {
                    EXPECT_NEAR(Ku[i], F[i], 1e-9 * scale) << "grado di liberta' " << i;
                }
            }
        }
    }
}

// ESEMPIO 175. La soluzione e' LINEARE nel carico e negli spostamenti imposti: con gli stessi vincoli, se u1 risolve
// il problema (F1, g1) e u2 il problema (F2, g2) (g = gli spostamenti imposti), allora u1 + u2 risolve (F1 + F2, g1 +
// g2) e s u1 risolve (s F1, s g1), per ogni numero s (anche negativo). Si prova su cinque mesh (due rettangoli, i tre
// che non lo sono) con due materiali (E = 1 e acciaio) e con s = -2.5, con tre vincoli isostatici e spostamenti
// imposti diversi da 0.
TEST(ElasticitySolve2d, TheSolutionIsLinearInTheLoadAndInTheImposedValues) {
    const std::vector<Mesh2D> all = sample_meshes();
    const std::vector<std::size_t> chosen = {2, 3, 6, 7, 8};
    for (std::size_t mi : chosen) {
        const Mesh2D& mesh = all[mi];
        const std::vector<std::size_t> dofs = isostatic_dofs(mesh, 0);
        const double size = mesh_size(mesh);
        for (const Matrix3& D : {hooke_quarter(), steel()}) {
            SCOPED_TRACE("mesh " + std::to_string(mi) + ", D00 = " + std::to_string(D[0][0]));
            const double k_scale = D[0][0];
            const std::size_t N = 2 * mesh.num_nodes();
            const std::vector<double> F1 = wavy_vector(N, 1e-2 * k_scale * size);
            std::vector<double> F2(N);
            for (std::size_t i = 0; i < N; ++i) {
                F2[i] = 1e-2 * k_scale * size * std::cos(0.7 * static_cast<double>(i) + 1.0);
            }
            const std::vector<double> g1 = {0.002 * size, -0.001 * size, 0.003 * size};
            const std::vector<double> g2 = {-0.004 * size, 0.002 * size, 0.001 * size};
            const double s = -2.5;

            const std::vector<double> u1 = solve_displacement_2d(mesh, D, F1, dofs, g1);
            const std::vector<double> u2 = solve_displacement_2d(mesh, D, F2, dofs, g2);
            std::vector<double> F12(N);
            std::vector<double> sF1(N);
            for (std::size_t i = 0; i < N; ++i) {
                F12[i] = F1[i] + F2[i];
                sF1[i] = s * F1[i];
            }
            const std::vector<double> g12 = {g1[0] + g2[0], g1[1] + g2[1], g1[2] + g2[2]};
            const std::vector<double> sg1 = {s * g1[0], s * g1[1], s * g1[2]};

            std::vector<double> sum(N);
            std::vector<double> scaled(N);
            for (std::size_t i = 0; i < N; ++i) {
                sum[i] = u1[i] + u2[i];
                scaled[i] = s * u1[i];
            }
            const double tolerance = 1e-9 * (largest_entry(u1) + largest_entry(u2));
            expect_vectors_near(solve_displacement_2d(mesh, D, F12, dofs, g12), sum, tolerance);
            expect_vectors_near(solve_displacement_2d(mesh, D, sF1, dofs, sg1), scaled, tolerance);
        }
    }
}

// ESEMPIO 176. Il risultato non dipende dalle unita' del modulo di Young. Le stesse lastre di E = 1 si risolvono con
// E = s, per s da 1e-12 (un materiale quasi senza rigidezza) a 1e15; K cresce con s e il CG vedrebbe numeri di ogni
// ordine. Due controlli, con nu = 0.3 e alpha = 12e-6, su un rettangolo di 4 x 3 celle e uno di 6 x 6 (K ha
// asimmetrie di arrotondamento di 1e-16 volte i suoi numeri: sopra 1e-12 in valore assoluto con E grande, e il CG,
// che controlla la simmetria con una tolleranza assoluta di 1e-12, lancerebbe std::invalid_argument):
//   - scaldata di 100 kelvin e libera, la lastra si dilata di c = (1 + nu) alpha dT = 1.56e-3, per qualunque E (le
//     forze termiche crescono con E e gli spostamenti restano uguali). I vincoli impongono anche una traslazione di
//     (0.3, -0.2): con E molto piccolo i gradi di liberta' fissati, che nel sistema valgono 1, pesano piu' di tutto
//     il resto nella tolleranza relativa a norm(b), e senza scalare il CG si ferma subito, lasciando a 0 tutti i
//     gradi di liberta' liberi, e dice che e' convergente;
//   - tirata con la stessa forza (la trazione dell'ESEMPIO 169, tensione sigma = 1 sul lato destro), la lastra si
//     allunga come 1 / E: gli spostamenti con E = s sono quelli con E = 1 divisi per s.
TEST(ElasticitySolve2d, TheSolutionDoesNotDependOnTheUnitsOfTheModulus) {
    const double nu = 0.3;
    const double alpha = 12.0e-6;
    const double dT = 100.0;
    for (const Grid& g : std::vector<Grid>{{3.0, 1.5, 4, 3}, {1.0, 1.0, 6, 6}}) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const Problem unit_pull = stretched_rectangle(mesh, g.lx, 1.0, nu, 1.0);
        for (double s : {1.0e-12, 1.0e-9, 1.0e-6, 1.0, 1.0e6, 210.0e9, 1.0e15}) {
            SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " + std::to_string(s));
            const Matrix3 D = plane_strain_matrix(s, nu);

            const Problem heated = heated_free_body(mesh, Material{D, nu, alpha}, dT, 0, 0.3, -0.2);
            expect_vectors_near(solve_displacement_2d(mesh, D, heated.F, heated.dofs, heated.values), heated.exact,
                                1e-9 * largest_entry(heated.exact));

            std::vector<double> want = unit_pull.exact;
            for (double& w : want) {
                w /= s;
            }
            expect_vectors_near(solve_displacement_2d(mesh, D, unit_pull.F, unit_pull.dofs, unit_pull.values), want,
                                1e-9 * largest_entry(want));
        }
    }
}

// ESEMPIO 177. I nodi che non stanno in nessun triangolo (la mesh ha i nodi 0 = (5, 5) e 5 = (7, 7) fuori dai due
// triangoli del quadrato 1, 2, 3, 4) non hanno rigidezza: le loro righe di K sono tutte 0 e non esiste uno spostamento
// "giusto". Vanno fissati, tutti e due i gradi di liberta': allora escono con il valore imposto, esattamente, e il
// resto della lastra si risolve come se non ci fossero (qui il quadrato scaldato libero, che si dilata di
// c = (1 + nu) alpha dT in x e in y). Se un grado di liberta' di un nodo fuori da ogni triangolo NON e' fissato (anche
// uno solo dei suoi due) la funzione lancia std::invalid_argument, e non restituisce un numero inventato.
TEST(ElasticitySolve2d, NodesOutsideEveryTriangleMustBeFixed) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const Material m = {steel(), 0.3, 12.0e-6};
    const double dT = 100.0;
    const double c = (1.0 + m.nu) * m.alpha * dT;
    // le temperature dei nodi fuori dai triangoli (1e9 e -1e9) non vengono lette
    const double hot = reference_temperature + dT;
    const std::vector<double> F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                                           {1.0e9, hot, hot, hot, hot, -1.0e9}, reference_temperature);
    ASSERT_EQ(F.size(), 12u);

    // il quadrato: nodo 1 = (0, 0) con u e v (numeri 2 e 3) e v del nodo 2 = (1, 0) (numero 5), tutti a 0
    const std::vector<std::size_t> square = {2, 3, 5};
    const std::vector<double> square_values = {0.0, 0.0, 0.0};
    const std::vector<std::size_t> isolated_dofs = {0, 1, 10, 11};

    // fissati anche i quattro numeri dei nodi 0 e 5 (0, 1, 10, 11): escono come sono, e il quadrato si dilata
    const std::vector<std::size_t> dofs = {0, 1, 2, 3, 5, 10, 11};
    const std::vector<double> values = {5.0, 6.0, 0.0, 0.0, 0.0, -7.0, 8.0};
    const std::vector<double> got = solve_displacement_2d(mesh, m.D, F, dofs, values);
    ASSERT_EQ(got.size(), 12u);
    EXPECT_EQ(got[0], 5.0);
    EXPECT_EQ(got[1], 6.0);
    EXPECT_EQ(got[10], -7.0);
    EXPECT_EQ(got[11], 8.0);
    // i nodi 1, 2, 3, 4 sono (0, 0), (1, 0), (1, 1), (0, 1) e il quadrato si dilata di c in x e in y
    const std::vector<double> want_square = {0.0, 0.0, c, 0.0, c, c, 0.0, c};
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(got[2 + i], want_square[i], 1e-9 * c) << "grado di liberta' " << 2 + i;
    }

    // basta che un grado di liberta' di un nodo fuori da ogni triangolo resti libero ("missing": quelli non fissati)
    const std::vector<std::vector<std::size_t>> missing_sets = {{0, 1, 10, 11}, {0, 1}, {10, 11}, {0}, {1},
                                                                {10},           {11},   {0, 1, 10}, {0, 10, 11}};
    for (const std::vector<std::size_t>& missing : missing_sets) {
        std::vector<std::size_t> some_dofs = square;
        std::vector<double> some_values = square_values;
        for (std::size_t d : isolated_dofs) {
            if (std::find(missing.begin(), missing.end(), d) == missing.end()) {
                some_dofs.push_back(d);
                some_values.push_back(1.0);
            }
        }
        EXPECT_THROW(solve_displacement_2d(mesh, m.D, F, some_dofs, some_values), std::invalid_argument)
            << missing.size() << " gradi di liberta' dei nodi fuori dai triangoli non fissati";
    }
}

// ESEMPIO 178. Dati sbagliati: la funzione lancia std::invalid_argument (e non scrive fuori da un vettore o
// restituisce un risultato inventato). Con la mesh di 2 x 1 celle (6 nodi, 12 gradi di liberta': 0, ..., 11) sono
// sbagliati:
//   - un F che non ha 12 numeri (11, 13, 6, 0);
//   - meno di 3 gradi di liberta' fissati (0, 1 o 2): K ha 3 moti rigidi e senza almeno 3 vincoli e' singolare;
//   - un numero diverso di spostamenti imposti e di gradi di liberta' fissati (in tutti e due i versi);
//   - un grado di liberta' fissato che non esiste (12 o piu', anche un miliardo: l'indice va controllato PRIMA di
//     usarlo per scrivere nel punto di partenza del CG, altrimenti il programma si schianta invece di lanciare
//     l'eccezione);
//   - un grado di liberta' fissato due volte (con lo stesso valore o con valori diversi).
// Vanno bene invece: esattamente 3 vincoli (anche in disordine), tutti i 12, 4 vincoli. I casi di indice sbagliato
// hanno almeno 3 elementi, per non confondere il controllo dell'indice con quello del numero minimo.
TEST(ElasticitySolve2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 2, 1);
    const Matrix3 D = hooke_quarter();
    const std::vector<double> F(12, 0.0);

    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0, 0.0}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {3, 0, 1}, {0.1, 0.0, 0.2}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 11}, {0.0, 0.0, 0.0, 0.5}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                          {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}));

    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(11, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(13, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(6, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {}, {}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1}, {0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {1, 0}, {0.0, 0.0}), std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1}, {0.0, 0.0, 0.0}), std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 12}, {0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 100}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 1000000000}, {0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {1000000000, 0, 1, 3}, {0.0, 0.0, 0.0, 0.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 1}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 3}, {0.0, 0.0, 0.0, 0.5}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {2, 2, 2}, {0.0, 0.0, 0.0}), std::invalid_argument);
}

// ESEMPIO 179. I sistemi senza soluzione: la funzione lancia std::runtime_error, e non restituisce un vettore
// sbagliato, pieno di NaN o di numeri enormi.
//   - Un NaN nelle forze, o in uno spostamento imposto: i confronti con un NaN sono tutti falsi e il CG si ferma
//     subito, non convergente.
//   - Tre vincoli che lasciano libera la rotazione (u e v del nodo 0 e u del nodo 1 con la STESSA y: la rotazione
//     attorno al nodo 0 sposta il nodo 1 solo in y) e un carico che la mette in moto (una forza verticale sul nodo 3,
//     che ha un momento diverso da 0 attorno al nodo 0): il corpo ruota senza fermarsi, non esiste uno stato di
//     equilibrio. Il sistema e' singolare e senza soluzione, e il CG non converge davvero: sul quadrato unitario,
//     pero', il suo residuo (che aggiorna con una formula, senza mai ricalcolare b - A u) arriva a 0 in 9 iterazioni
//     mentre u ha numeri di ordine 1e15, e dice che ha convergenza; solo il residuo ricalcolato da u (circa 0.36
//     volte norm(b)) lo smaschera. Con 4 x 3 celle e con 6 x 6 invece il CG supera il numero massimo di iterazioni.
//   - Due triangoli separati (due corpi), con i tre vincoli in uno solo e una forza sull'altro: l'altro e' libero e la
//     forza lo mette in moto.
// Con un carico IN EQUILIBRIO (le forze termiche) gli stessi vincoli lasciano una rotazione libera ma il sistema ha
// infinite soluzioni, e una la si trova: nessuna eccezione. Allo stesso modo i vincoli in tutti e due i corpi risolvono
// il problema dei due triangoli.
TEST(ElasticitySolve2d, ThrowsRuntimeErrorWhenTheSystemCannotBeSolved) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const Matrix3 D = hooke_quarter();

    const Mesh2D square = unit_square();
    const std::vector<double> heating = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    std::vector<double> with_nan = heating;
    with_nan[5] = not_a_number;
    EXPECT_NO_THROW(solve_displacement_2d(square, D, heating, {0, 1, 3}, {0.0, 0.0, 0.0}));
    EXPECT_THROW(solve_displacement_2d(square, D, with_nan, {0, 1, 3}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_THROW(solve_displacement_2d(square, D, heating, {0, 1, 3}, {0.0, not_a_number, 0.0}), std::runtime_error);

    // la rotazione attorno al nodo 0 e' libera: u0 = v0 = u1 = 0 (nodi 0 e 1 con y = 0)
    std::vector<double> push_up(8, 0.0);
    push_up[7] = 1.0;  // forza verticale sul nodo 3 = (1, 1)
    std::vector<double> push_right(8, 0.0);
    push_right[6] = 1.0;  // forza orizzontale sul nodo 3
    EXPECT_THROW(solve_displacement_2d(square, D, push_up, {0, 1, 2}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_THROW(solve_displacement_2d(square, D, push_right, {0, 1, 2}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_displacement_2d(square, D, heating, {0, 1, 2}, {0.0, 0.0, 0.0}));
    EXPECT_NO_THROW(solve_displacement_2d(square, D, push_up, {0, 1, 3}, {0.0, 0.0, 0.0}));

    for (const Grid& g : std::vector<Grid>{{3.0, 1.5, 4, 3}, {1.0, 1.0, 6, 6}}) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        std::vector<double> load(2 * mesh.num_nodes(), 0.0);
        load[2 * (mesh.num_nodes() - 1) + 1] = -1000.0;  // forza verso il basso sull'ultimo nodo
        const std::size_t bottom_right = g.nx;  // il nodo (lx, 0): ha la stessa y del nodo 0
        EXPECT_THROW(solve_displacement_2d(mesh, steel(), load, {0, 1, 2 * bottom_right}, {0.0, 0.0, 0.0}),
                     std::runtime_error)
            << "rettangolo " << g.nx << " x " << g.ny;
        EXPECT_NO_THROW(solve_displacement_2d(mesh, steel(), load, {0, 1, 2 * bottom_right + 1}, {0.0, 0.0, 0.0}))
            << "rettangolo " << g.nx << " x " << g.ny;
    }

    const Mesh2D two_pieces = two_separate_triangles();
    std::vector<double> pull(12, 0.0);
    pull[6] = 1.0;  // forza in x sul nodo 3, del secondo triangolo
    EXPECT_THROW(solve_displacement_2d(two_pieces, D, pull, {0, 1, 3}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_displacement_2d(two_pieces, D, pull, {0, 1, 3, 6, 7, 10}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

// ESEMPIO 180. Una lastra grande: un quadrato di acciaio di 1 m di lato con 40 x 40 celle (1681 nodi, 3362 gradi di
// liberta'), scaldato di 100 kelvin e libero (tre vincoli isostatici). La soluzione esatta e' la dilatazione
// u = c x, v = c y con c = (1 + nu) alpha dT = 1.56e-3: il CG fa circa 700 iterazioni (sulle mesh degli altri test mai
// piu' di 100) con i numeri di K di ordine 1e11, abbastanza da vedere se la scalatura, la tolleranza e il numero
// massimo di iterazioni vanno bene su un problema di taglia vera. Lo spostamento di ogni nodo deve coincidere con la
// formula entro 1e-9 volte la dilatazione c (l'errore vero e' circa 3e-13); la funzione ci mette circa mezzo secondo.
TEST(ElasticitySolve2d, ALargeSteelPlateIsSolvedAccurately) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 40, 40);
    ASSERT_EQ(mesh.num_nodes(), 1681u);
    const Material m = {steel(), 0.3, 12.0e-6};

    const Problem pr = heated_free_body(mesh, m, 100.0, 0, 0.0, 0.0);
    const std::vector<double> u = solve_displacement_2d(mesh, m.D, pr.F, pr.dofs, pr.values);
    ASSERT_EQ(u.size(), 3362u);
    double worst = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        worst = std::max(worst, std::fabs(u[i] - pr.exact[i]));
    }
    const double c = (1.0 + m.nu) * m.alpha * 100.0;
    EXPECT_LT(worst, 1e-9 * c);
}