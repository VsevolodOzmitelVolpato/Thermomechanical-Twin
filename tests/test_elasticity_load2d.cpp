#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- Dalle forze termiche dei triangoli alle forze termiche di tutta la mesh ---------------------------------------
//
// element_thermal_load(mesh, e, D, eps_th) da' le 6 forze di UN triangolo scaldato, sui suoi gradi di liberta' LOCALI
// (u0, v0, u1, v1, u2, v2). Il vettore di tutta la mesh, F, ha 2 n numeri (n = numero dei nodi) e i suoi indici sono i
// gradi di liberta' GLOBALI di K: il nodo I ha la forza in x al numero 2 I e quella in y al numero 2 I + 1. Come per il
// carico del calore (assemble_load_2d) e' la somma dei contributi dei triangoli, con la corrispondenza di K:
//
//     F[P] = somma, sui triangoli e che contengono P, di f_e[i]
//            dove i e' il grado di liberta' locale di P, cioe' con t = mesh.triangle(e):  P = 2 t[i / 2] + i % 2.
//
// La novita' rispetto al calore e' che ogni triangolo ha il SUO riscaldamento. La temperatura T e' data nei nodi ed e'
// lineare dentro ogni triangolo, quindi la deformazione termica eps_th = (1 + nu) alpha (T - T_ref) (1, 1, 0) e'
// lineare. La forza A B^T D eps_th (B e' costante nel triangolo) ha bisogno solo del VALORE MEDIO di eps_th, che per
// una funzione lineare e' quello nel baricentro, cioe' la media dei valori nei tre nodi (il baricentro ha coordinate
// baricentriche 1/3, 1/3, 1/3): non e' un'approssimazione, e' esatto. Percio' per ogni triangolo
//
//     dT_e = (T[t[0]] + T[t[1]] + T[t[2]]) / 3 - T_ref,
//     eps_e = plane_strain_thermal_strain(nu, alpha, dT_e),
//     f_e = element_thermal_load(mesh, e, D, eps_e).
//
// assemble_thermal_load_2d(mesh, D, nu, alpha, T, T_ref) restituisce F. Lancia std::invalid_argument se T non ha un
// numero per nodo o se nu non sta fra -1 e 1/2 (lo controlla plane_strain_thermal_strain).
//
// Cosa deve avere una vera forza termica, ed e' quello che i test controllano:
//   - sul quadrato unitario si calcola a mano (ESEMPIO 157);
//   - con una temperatura uniforme la tensione termica D eps_th e' costante: le forze si annullano sui nodi interni e
//     restano sul bordo, dove ogni lato spinge i suoi due nodi verso l'esterno (ESEMPIO 158);
//   - i moti rigidi non fanno lavoro: la risultante e il momento delle forze sono 0, qualunque sia T (ESEMPIO 159);
//   - per uno spostamento lineare il lavoro u^T F si sa in anticipo, con l'integrale di T - T_ref (ESEMPIO 160);
//   - un corpo libero scaldato in modo uniforme si dilata senza tensioni: K u = F (ESEMPIO 161);
//   - F e' lineare in T - T_ref e in alpha, e senza variazione di temperatura non c'e' nessuna forza (ESEMPIO 162);
//   - il nodo da cui si comincia a scrivere un triangolo, e l'ordine dei triangoli, non cambiano F (ESEMPIO 163);
//   - i nodi fuori da ogni triangolo hanno forze nulle e la loro temperatura non conta (ESEMPIO 164);
//   - e' la somma dei contributi dei triangoli, su ogni tipo di mesh (ESEMPIO 165);
//   - dati sbagliati lanciano std::invalid_argument (ESEMPIO 166).

namespace {

// Un rettangolo lx x ly diviso in nx x ny celle, come in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Sei rettangoli di forme molto diverse (gli stessi dei test dell'assemblaggio di K).
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
// 4 = (1, 0.7). Quattro triangoli attorno al nodo 4, tutti di forma e area diverse (in tutto 10.5).
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Due triangoli di area DIVERSA: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); (0, 1, 2) ha A = 4 e (1, 3, 2) ha
// A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

// Sei nodi e due soli triangoli, (1, 2, 3) e (1, 3, 4): i nodi 0 e 5 non stanno in nessun triangolo.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
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

// Tutte le mesh di sample_meshes() e in piu' quella con due nodi fuori da ogni triangolo.
std::vector<Mesh2D> sample_meshes_with_isolated_nodes() {
    std::vector<Mesh2D> meshes = sample_meshes();
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    return meshes;
}

// Materiali a mano (3 x 3, indici xx, yy, xy), simmetrici: la forza termica non ha bisogno che siano fisici.
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
// (D e nu arrivano separati alla funzione, quindi si provano anche materiali non fisici: nu e alpha negativi).
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

// Una temperatura che cambia da nodo a nodo senza regole semplici (intorno a 300, con scarti di decine di kelvin).
std::vector<double> wavy_temperature(const Mesh2D& mesh) {
    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double x = mesh.x(I);
        const double y = mesh.y(I);
        T[I] = 300.0 + 40.0 * std::sin(1.7 * x + 0.4 * y) + 25.0 * std::cos(0.9 * y) - 3.0 * x * y +
               0.5 * static_cast<double>(I % 7);
    }
    return T;
}

// Un'altra temperatura, diversa dalla precedente.
std::vector<double> other_temperature(const Mesh2D& mesh) {
    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        T[I] = 280.0 + 10.0 * std::cos(2.0 * mesh.x(I)) - 4.0 * mesh.y(I) + 1.5 * static_cast<double>(I % 3);
    }
    return T;
}

// F per un materiale m.
std::vector<double> thermal_load(const Mesh2D& mesh, const Material& m, const std::vector<double>& T, double T_ref) {
    return assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, T_ref);
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

// Il lavoro u^T F e la somma dei valori assoluti dei suoi termini (la "scala" di un lavoro che puo' venire 0 per
// cancellazione).
struct Work {
    double value;
    double scale;
};

Work work_of(const std::vector<double>& u, const std::vector<double>& F) {
    Work w{0.0, 0.0};
    for (std::size_t i = 0; i < u.size() && i < F.size(); ++i) {
        w.value += u[i] * F[i];
        w.scale += std::fabs(u[i] * F[i]);
    }
    return w;
}

// L'integrale di T - T_ref sulla lastra, per una temperatura lineare in ogni triangolo: per ogni triangolo e' l'area
// per la media dei tre valori nei nodi (esatto).
double integral_of_temperature_change(const Mesh2D& mesh, const std::vector<double>& T, double T_ref) {
    double integral = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        integral += mesh.area(e) * ((T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref);
    }
    return integral;
}

// La chiave di un lato, senza direzione: i suoi due nodi in ordine crescente.
std::pair<std::size_t, std::size_t> edge_key(std::size_t p, std::size_t q) {
    return {std::min(p, q), std::max(p, q)};
}

// Le forze che una tensione COSTANTE sigma = (sxx, syy, sxy) produce sui nodi, calcolate dal bordo e non dai triangoli.
// Ogni lato che compare in un solo triangolo e' di bordo: se va dal nodo p al nodo q (in senso antiorario nel suo
// triangolo) e (dx, dy) = q - p, la sua normale esterna per la lunghezza e' (dy, -dx) e spinge ciascuno dei suoi due
// nodi con la forza  (1/2) sigma (dy, -dx) = (1/2) (sxx dy - sxy dx, sxy dy - syy dx). I lati interni, che stanno in
// due triangoli, si annullano; i nodi interni e quelli fuori da ogni triangolo restano a 0.
std::vector<double> boundary_forces(const Mesh2D& mesh, const Vector3& sigma) {
    std::map<std::pair<std::size_t, std::size_t>, int> uses;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        for (std::size_t a = 0; a < 3; ++a) {
            ++uses[edge_key(t[a], t[(a + 1) % 3])];
        }
    }
    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        for (std::size_t a = 0; a < 3; ++a) {
            const std::size_t p = t[a];
            const std::size_t q = t[(a + 1) % 3];
            if (uses.at(edge_key(p, q)) != 1) {
                continue;
            }
            const double dx = mesh.x(q) - mesh.x(p);
            const double dy = mesh.y(q) - mesh.y(p);
            const double fx = 0.5 * (sigma[0] * dy - sigma[2] * dx);
            const double fy = 0.5 * (sigma[2] * dy - sigma[1] * dx);
            F[2 * p] += fx;
            F[2 * p + 1] += fy;
            F[2 * q] += fx;
            F[2 * q + 1] += fy;
        }
    }
    return F;
}

// La stessa mesh con i tre nodi di ogni triangolo scritti a partire da un altro: (a, b, c) diventa (b, c, a) se shift
// e' 1 e (c, a, b) se shift e' 2. Sono gli stessi triangoli, ancora antiorari.
Mesh2D with_rotated_triangles(const Mesh2D& mesh, std::size_t shift) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        xs.push_back(mesh.x(I));
        ys.push_back(mesh.y(I));
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        triangles.push_back({t[shift % 3], t[(shift + 1) % 3], t[(shift + 2) % 3]});
    }
    return Mesh2D(xs, ys, triangles);
}

// La stessa mesh con i triangoli elencati in ordine inverso.
Mesh2D with_reversed_triangle_order(const Mesh2D& mesh) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        xs.push_back(mesh.x(I));
        ys.push_back(mesh.y(I));
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = mesh.num_triangles(); e > 0; --e) {
        triangles.push_back(mesh.triangle(e - 1));
    }
    return Mesh2D(xs, ys, triangles);
}

// Il "conto lento": prima la forza di ogni triangolo (con la media dei tre nodi scritta al contrario), poi per ogni
// grado di liberta' globale P si cercano, in tutti i triangoli e in tutti i loro sei gradi di liberta' locali, quelli
// che hanno numero globale P e se ne sommano le forze. Serve solo come confronto, e solo per mesh piccole.
std::vector<double> slow_thermal_load(const Mesh2D& mesh, const Matrix3& D, double nu, double alpha,
                                      const std::vector<double>& T, double T_ref) {
    std::vector<Vector6> forces;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const double mean = T[t[2]] / 3.0 + T[t[1]] / 3.0 + T[t[0]] / 3.0;
        forces.push_back(element_thermal_load(mesh, e, D, plane_strain_thermal_strain(nu, alpha, mean - T_ref)));
    }
    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t P = 0; P < F.size(); ++P) {
        for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
            const Triangle& t = mesh.triangle(e);
            for (std::size_t a = 0; a < 3; ++a) {
                for (std::size_t c = 0; c < 2; ++c) {
                    if (2 * t[a] + c == P) {
                        F[P] += forces[e][2 * a + c];
                    }
                }
            }
        }
    }
    return F;
}

// Confronta due vettori elemento per elemento, con una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "grado di liberta' " << i;
    }
}

}  // namespace

// ESEMPIO 157. A mano. Il quadrato unitario ha 4 nodi, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), e due triangoli,
// T0 = (0, 1, 2) e T1 = (1, 3, 2), tutti e due di area 1/2. Il materiale ha E = 1 e nu = 1/4 (lambda = mu = 0.4, quindi
// D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]) e alpha = 1 (un numero comodo, non un materiale vero); la
// temperatura di riferimento e' T_ref = 5 e quella dei nodi T = (5, 6, 7, 8).
// T0 ha i nodi (0, 1, 2): la media e' (5 + 6 + 7) / 3 = 6, quindi dT = 1. T1 ha i nodi (1, 3, 2): (6 + 8 + 7) / 3 = 7,
// quindi dT = 2. La deformazione termica e' (1 + nu) alpha dT (1, 1, 0) = 1.25 dT (1, 1, 0) e la tensione e'
// D eps_th = 1.25 dT (1.6, 1.6, 0) = 2 dT (1, 1, 0): sigma = (2, 2, 0) in T0 e (4, 4, 0) in T1.
// Per una tensione (s, s, 0) le forze del triangolo, A B^T sigma, sono (s / 2) (b_0, c_0, b_1, c_1, b_2, c_2), con
// b_i = y_p - y_q e c_i = x_q - x_p (p e q sono i due nodi dopo il nodo i). In T0 i (b_i, c_i) sono (-1, -1), (1, 0),
// (0, 1); in T1, con i nodi (1, 3, 2), sono (0, -1), (1, 1), (-1, 0). Quindi
//     f_0 = (-1, -1, 1, 0, 0, 1)        (s = 2),        f_1 = (0, -2, 2, 2, -2, 0)        (s = 4).
// I gradi di liberta' locali di T0 sono i globali (0, 1, 2, 3, 4, 5) e quelli di T1 i globali (2, 3, 6, 7, 4, 5). Si
// somma ogni numero nel suo posto, per esempio  F[2] = f_0[2] + f_1[0] = 1 + 0 = 1  e  F[3] = f_0[3] + f_1[1] = 0 - 2:
//     F = (-1, -1, 1, -2, -2, 1, 2, 2).
// Le forze sulle componenti x sommano a -1 + 1 - 2 + 2 = 0 e quelle sulle y a 0: la lastra libera non ha forze
// esterne. Con la temperatura uniforme T = (7, 7, 7, 7) (dT = 2 in tutti e due i triangoli) i quattro angoli sono
// spinti verso l'esterno lungo le diagonali, F = (-2, -2, 2, -2, -2, 2, 2, 2).
TEST(ElasticityLoad2d, ThermalLoadOfTwoRightTrianglesMatchesHandComputation) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();

    const std::vector<double> got = assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {5.0, 6.0, 7.0, 8.0}, 5.0);
    const std::vector<double> want = {-1.0, -1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0};
    ASSERT_EQ(got.size(), want.size()) << "dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-14) << "grado di liberta' " << i;
    }

    const std::vector<double> got_uniform = assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {7.0, 7.0, 7.0, 7.0}, 5.0);
    const std::vector<double> want_uniform = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    ASSERT_EQ(got_uniform.size(), want_uniform.size()) << "dimensione";
    for (std::size_t i = 0; i < want_uniform.size(); ++i) {
        EXPECT_NEAR(got_uniform[i], want_uniform[i], 1e-14) << "grado di liberta' " << i << " (temperatura uniforme)";
    }
}

// ESEMPIO 158. Riscaldamento uniforme. Se T e' la stessa in tutti i nodi, ogni triangolo ha lo stesso dT, la stessa
// deformazione termica eps_th = (1 + nu) alpha dT (1, 1, 0) e quindi la stessa tensione sigma = D eps_th. Le forze di
// una tensione costante su un triangolo sono A B^T sigma: il nodo i riceve (1/2) sigma (b_i, c_i), e (b_i, c_i) e' la
// somma delle normali esterne (moltiplicate per la lunghezza) dei due lati che arrivano al nodo. Sommando i triangoli
// i lati interni compaiono due volte, con normali opposte, e si annullano: restano solo i LATI DI BORDO. Un lato di
// bordo, dal nodo p al nodo q in senso antiorario, con (dx, dy) = q - p, ha normale esterna per lunghezza (dy, -dx) e
// spinge ciascuno dei suoi due nodi con la forza (1/2) sigma (dy, -dx). Per esempio, con sigma = (s, s, 0), il lato
// verticale destro di una lastra, lungo l, spinge i suoi due nodi con (s l / 2, 0) ciascuno. I nodi interni (e quelli
// fuori da ogni triangolo) hanno forza 0. Il confronto usa la formula del bordo, che non passa per le funzioni
// provate. Provato su dieci mesh e tre materiali, con T = 350 e T_ref = 293.15.
TEST(ElasticityLoad2d, UniformHeatingPushesTheBoundaryOutwardAndLeavesTheInteriorBalanced) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];

            const double e = (1.0 + mat.nu) * mat.alpha * (350.0 - reference_temperature);
            const Vector3 sigma = {e * (mat.D[0][0] + mat.D[0][1]), e * (mat.D[1][0] + mat.D[1][1]),
                                   e * (mat.D[2][0] + mat.D[2][1])};
            const std::vector<double> want = boundary_forces(mesh, sigma);
            ASSERT_GT(largest_entry(want), 0.0);

            const std::vector<double> got = thermal_load(mesh, mat, std::vector<double>(mesh.num_nodes(), 350.0),
                                                         reference_temperature);
            expect_vectors_near(got, want, 1e-12 * largest_entry(want));
        }
    }
}

// ESEMPIO 159. Equilibrio. Un moto rigido, una traslazione o una rotazione, non deforma nessun triangolo e quindi non
// fa nessun lavoro contro le forze termiche: u^T F = 0 per u = (1, 0, 1, 0, ...), (0, 1, 0, 1, ...), (-y_I, x_I, ...)
// e le loro combinazioni, qualunque sia la temperatura. Con le tre scelte vuol dire che la somma delle forze in x e'
// 0, quella delle forze in y e' 0 e anche il momento, la somma di x F_y - y F_x, e' 0: un corpo libero che si scalda
// non ha forze esterne, non si sposta e non si mette a ruotare. E' quello che si rompe se u e v sono scambiati o se un
// grado di liberta' finisce nel nodo sbagliato. Provato su dieci mesh, tre materiali e la temperatura "a onde".
TEST(ElasticityLoad2d, ThermalForcesAreInEquilibriumAndRigidMotionsDoNoWork) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const std::vector<double> F =
                thermal_load(mesh, materials[m], wavy_temperature(mesh), reference_temperature);
            ASSERT_EQ(F.size(), 2 * mesh.num_nodes());
            ASSERT_GT(largest_entry(F), 0.0);

            const std::vector<LinearField> motions = rigid_motions();
            for (std::size_t r = 0; r < motions.size(); ++r) {
                const Work w = work_of(nodal_displacements(mesh, motions[r]), F);
                EXPECT_NEAR(w.value, 0.0, 1e-12 * w.scale) << "moto rigido numero " << r;
            }
        }
    }
}

// ESEMPIO 160. Il lavoro di un campo lineare. Per uno spostamento lineare u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y
// la deformazione e' la stessa ovunque, eps = (a1, b2, a2 + b1), e il lavoro delle forze termiche e' la somma, sui
// triangoli, di A eps^T D eps_th,e. Con eps_th,e = (1 + nu) alpha dT_e (1, 1, 0) viene
//     u^T F = (1 + nu) alpha (eps^T D (1, 1, 0)) * somma_e A_e dT_e = (1 + nu) alpha (eps^T D (1, 1, 0)) * I,
// con I l'integrale di T - T_ref sulla lastra (A_e dT_e, con dT_e la media dei tre nodi, e' l'integrale esatto sul
// triangolo di una temperatura lineare). Il lavoro dipende dalla temperatura SOLO attraverso I, e non dalla traslazione
// del campo (a0, b0). Il confronto non passa per nessuna funzione provata e controlla insieme la media dei nodi, T_ref,
// il fattore (1 + nu) alpha e il posto delle forze. Provato con sette campi su dieci mesh, tre materiali e la
// temperatura "a onde".
TEST(ElasticityLoad2d, ThermalForcesDoTheWorkOfALinearDisplacementField) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    const std::vector<LinearField> fields = strain_fields();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        const std::vector<double> T = wavy_temperature(mesh);
        const double integral = integral_of_temperature_change(mesh, T, reference_temperature);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Material& mat = materials[m];
            const std::vector<double> F = thermal_load(mesh, mat, T, reference_temperature);
            for (std::size_t f = 0; f < fields.size(); ++f) {
                SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m) +
                             ", campo numero " + std::to_string(f));
                const LinearField& field = fields[f];
                const double eps[3] = {field.a1, field.b2, field.a2 + field.b1};
                double eps_D_ones = 0.0;
                for (std::size_t i = 0; i < 3; ++i) {
                    eps_D_ones += eps[i] * (mat.D[i][0] + mat.D[i][1]);
                }
                const double want = (1.0 + mat.nu) * mat.alpha * eps_D_ones * integral;

                const Work w = work_of(nodal_displacements(mesh, field), F);
                EXPECT_NEAR(w.value, want, 1e-12 * w.scale);
            }
        }
    }
}

// ESEMPIO 161. Un corpo libero, scaldato in modo uniforme, si dilata senza nessuna tensione. Lo spostamento e' quello
// della dilatazione termica, u = e x, v = e y, con e = (1 + nu) alpha dT: e' lineare, quindi la sua deformazione e'
// proprio eps_th = e (1, 1, 0) in ogni triangolo e la tensione D (eps - eps_th) e' 0 ovunque. Le forze di deformazione
// sono K u (la rigidezza per lo spostamento) e quelle del calore sono F: se non c'e' tensione si bilanciano,
//     K u = F        (per qualunque D, anche non fisica).
// Lega le due funzioni, K del passo precedente e F di questo, che devono usare gli stessi gradi di liberta'. Provato su
// dieci mesh e tre materiali, con T = 350 e T_ref = 293.15.
TEST(ElasticityLoad2d, AUniformlyHeatedFreeBodyExpandsWithoutStress) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];

            const std::vector<double> F =
                thermal_load(mesh, mat, std::vector<double>(mesh.num_nodes(), 350.0), reference_temperature);
            ASSERT_GT(largest_entry(F), 0.0);

            const double e = (1.0 + mat.nu) * mat.alpha * (350.0 - reference_temperature);
            const std::vector<double> u = nodal_displacements(mesh, {0.0, e, 0.0, 0.0, 0.0, e});
            const std::vector<double> force = assemble_elasticity_stiffness_2d(mesh, mat.D) * u;
            expect_vectors_near(force, F, 1e-9 * largest_entry(F));
        }
    }
}

// ESEMPIO 162. Niente variazione, niente forza; e F e' lineare. La temperatura entra solo come T - T_ref, e F e'
// proporzionale a (T - T_ref) e ad alpha:
//   - se T = T_ref in ogni nodo (qualunque sia T_ref: 0, 0.1, -5, 293.15, 1e6) non c'e' dilatazione e F = 0; e anche
//     con alpha = 0;
//   - F(T, T_ref) = F(T - T_ref, 0): sommare lo stesso numero a T e a T_ref non cambia niente;
//   - F(T_ref + c (T - T_ref), T_ref) = c F(T, T_ref), per c = 2.5 e per c = -1 (la forza cambia segno);
//   - F con 2 alpha e' il doppio di F;
//   - la somma: se T1 e T2 sono due temperature, F(T1 + T2 - T_ref) = F(T1) + F(T2) (i due campi di
//     T - T_ref si sommano).
// Provato su dieci mesh e tre materiali.
TEST(ElasticityLoad2d, NoTemperatureChangeMeansNoForceAndTheForceIsLinear) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];
            const std::size_t n = mesh.num_nodes();
            const double ref = reference_temperature;

            const std::vector<double> T1 = wavy_temperature(mesh);
            const std::vector<double> F = thermal_load(mesh, mat, T1, ref);
            const double scale = largest_entry(F);
            ASSERT_GT(scale, 0.0);

            for (double value : {0.0, 0.1, -5.0, 293.15, 1.0e6}) {
                const std::vector<double> flat = thermal_load(mesh, mat, std::vector<double>(n, value), value);
                EXPECT_LE(largest_entry(flat), 1e-9 * scale) << "T = T_ref = " << value;
            }
            const std::vector<double> no_expansion = assemble_thermal_load_2d(mesh, mat.D, mat.nu, 0.0, T1, ref);
            EXPECT_EQ(largest_entry(no_expansion), 0.0) << "alpha = 0";

            std::vector<double> shifted(n);
            for (std::size_t I = 0; I < n; ++I) {
                shifted[I] = T1[I] - ref;
            }
            expect_vectors_near(thermal_load(mesh, mat, shifted, 0.0), F, 1e-12 * scale);

            for (double c : {2.5, -1.0}) {
                std::vector<double> scaled(n);
                std::vector<double> want(F.size());
                for (std::size_t I = 0; I < n; ++I) {
                    scaled[I] = ref + c * (T1[I] - ref);
                }
                for (std::size_t P = 0; P < F.size(); ++P) {
                    want[P] = c * F[P];
                }
                expect_vectors_near(thermal_load(mesh, mat, scaled, ref), want, 1e-12 * scale * std::fabs(c));
            }

            std::vector<double> doubled(F.size());
            const std::vector<double> twice_alpha =
                assemble_thermal_load_2d(mesh, mat.D, mat.nu, 2.0 * mat.alpha, T1, ref);
            for (std::size_t P = 0; P < F.size(); ++P) {
                doubled[P] = 2.0 * F[P];
            }
            expect_vectors_near(twice_alpha, doubled, 1e-12 * scale);

            const std::vector<double> T2 = other_temperature(mesh);
            std::vector<double> sum_field(n);
            for (std::size_t I = 0; I < n; ++I) {
                sum_field[I] = T1[I] + T2[I] - ref;
            }
            const std::vector<double> F2 = thermal_load(mesh, mat, T2, ref);
            std::vector<double> sum_forces(F.size());
            for (std::size_t P = 0; P < F.size(); ++P) {
                sum_forces[P] = F[P] + F2[P];
            }
            expect_vectors_near(thermal_load(mesh, mat, sum_field, ref), sum_forces,
                                1e-12 * (scale + largest_entry(F2)));
        }
    }
}

// ESEMPIO 163. Scrivere un triangolo a partire da un altro dei suoi nodi, (a, b, c) invece di (b, c, a), o elencare i
// triangoli in un altro ordine, non cambia la mesh e quindi non cambia F: la media dei tre nodi e le forze dei sei
// gradi di liberta' si spostano insieme. Si confronta F della mesh con F della stessa mesh con i triangoli
// "ruotati" di un posto e di due posti e con i triangoli in ordine inverso, su dieci mesh e tre materiali. Cambia solo
// l'ordine delle somme, quindi i numeri sono gli stessi a meno di un errore di arrotondamento.
TEST(ElasticityLoad2d, TheOrderOfTheTrianglesAndOfTheirNodesDoesNotMatter) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        const std::vector<double> T = wavy_temperature(mesh);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const std::vector<double> F = thermal_load(mesh, materials[m], T, reference_temperature);
            const double scale = largest_entry(F);
            ASSERT_GT(scale, 0.0);

            for (std::size_t shift : {1u, 2u}) {
                const Mesh2D rotated = with_rotated_triangles(mesh, shift);
                expect_vectors_near(thermal_load(rotated, materials[m], T, reference_temperature), F,
                                    1e-12 * scale);
            }
            const Mesh2D reversed = with_reversed_triangle_order(mesh);
            expect_vectors_near(thermal_load(reversed, materials[m], T, reference_temperature), F, 1e-12 * scale);
        }
    }
}

// ESEMPIO 164. Nodi fuori da ogni triangolo. La mesh ha 6 nodi, ma solo (1, 2, 3) e (1, 3, 4) stanno in un triangolo: i
// nodi 0 e 5 non hanno niente da assemblare. F ha comunque 12 numeri, due per nodo, e le forze dei nodi 0 e 5, cioe'
// F[0], F[1], F[10], F[11], sono 0. La loro temperatura non viene nemmeno letta: se in quei nodi si mette un numero
// qualunque, anche NaN, infinito o 1e300, F non cambia nemmeno di un bit.
TEST(ElasticityLoad2d, NodesOutsideEveryTriangleGetNoForceAndTheirTemperatureIsNotRead) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const std::vector<double> clean = {0.0, 350.0, 330.0, 360.0, 320.0, 0.0};
    const std::vector<double> want =
        assemble_thermal_load_2d(mesh, hooke_quarter(), 0.25, 1.0e-3, clean, reference_temperature);
    ASSERT_EQ(want.size(), 12u);
    ASSERT_GT(largest_entry(want), 0.0);
    for (std::size_t P : {0u, 1u, 10u, 11u}) {
        EXPECT_EQ(want[P], 0.0) << "grado di liberta' " << P;
    }

    const double infinity = std::numeric_limits<double>::infinity();
    for (double garbage : {std::numeric_limits<double>::quiet_NaN(), infinity, -infinity, 1.0e300, -7.0}) {
        std::vector<double> T = clean;
        T[0] = garbage;
        T[5] = -garbage;
        const std::vector<double> got =
            assemble_thermal_load_2d(mesh, hooke_quarter(), 0.25, 1.0e-3, T, reference_temperature);
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t P = 0; P < want.size(); ++P) {
            EXPECT_EQ(got[P], want[P]) << "grado di liberta' " << P << ", temperatura " << garbage;
        }
    }
}

// ESEMPIO 165. Il "conto lento": prima la forza di ogni triangolo, con la media dei tre nodi scritta al contrario, poi,
// per ogni grado di liberta' globale P, si cercano fra tutti i triangoli e tutti i loro sei gradi di liberta' locali
// quelli che hanno numero globale P (2 t[a] + c == P, con a il nodo locale e c la componente) e se ne sommano le
// forze. E' lo stesso conto dei gradi di liberta' globali di prima, scritto con tre cicli invece di uno. Le forze
// assemblate devono essere le stesse. Le mesh: sei rettangoli, il triangolo con un nodo dentro, il quadrilatero
// irregolare, due triangoli di area diversa e la mesh con due nodi fuori da ogni triangolo; i materiali sono tre, la
// temperatura e' quella "a onde" e quella "diversa".
TEST(ElasticityLoad2d, ThermalLoadMatchesTheSlowAssemblyOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];
            for (const std::vector<double>& T : {wavy_temperature(mesh), other_temperature(mesh)}) {
                const std::vector<double> slow = slow_thermal_load(mesh, mat.D, mat.nu, mat.alpha, T,
                                                                   reference_temperature);
                ASSERT_GT(largest_entry(slow), 0.0);
                expect_vectors_near(thermal_load(mesh, mat, T, reference_temperature), slow,
                                    1e-12 * largest_entry(slow));
            }
        }
    }
}

// ESEMPIO 166. Dati sbagliati. T deve avere un numero per nodo (il quadrato unitario ha 4 nodi: 0, 1, 3, 5 o 8 numeri
// sono sbagliati) e nu deve stare fra -1 e 1/2, estremi esclusi (NaN compreso): in tutti questi casi la funzione
// lancia std::invalid_argument e non un altro tipo di errore. Con nu appena dentro (-0.99, 0.49) va tutto bene. Gli
// altri dati non vengono controllati: alpha puo' essere 0 o negativo, la temperatura puo' essere negativa e T_ref
// qualunque.
TEST(ElasticityLoad2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const std::vector<double> T = {5.0, 6.0, 7.0, 8.0};

    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, T, 5.0));

    for (std::size_t size : {0u, 1u, 3u, 5u, 8u}) {
        EXPECT_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, std::vector<double>(size, 5.0), 5.0),
                     std::invalid_argument)
            << size << " temperature";
    }

    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double nu : {-1.0, -2.0, 0.5, 0.75, 1.0, not_a_number, infinity, -infinity}) {
        EXPECT_THROW(assemble_thermal_load_2d(mesh, D, nu, 1.0, T, 5.0), std::invalid_argument) << "nu = " << nu;
    }
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, -0.99, 1.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.49, 1.0, T, 5.0));

    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 0.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, -1.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {-40.0, 6.0, -7.0, 8.0}, -300.0));
}

// ESEMPIO 167. Una mesh grande. Un quadrato 1 x 1 di 100 x 100 celle ha 101 x 101 = 10201 nodi e 20000 triangoli: F
// ha 20402 numeri. La temperatura e' lineare, T = T_ref + 10 + 100 x, quindi l'integrale di T - T_ref e' 10 + 100 / 2 =
// 60 (area 1) per qualunque finezza. Il lavoro e' quello dell'ESEMPIO 160, (1 + nu) alpha (eps^T D (1, 1, 0)) 60: per
// u = x (eps = (1, 0, 0)) e' (1 + nu) alpha (D00 + D01) 60, per v = y (eps = (0, 1, 0)) e' (1 + nu) alpha (D10 + D11)
// 60, per u = y (eps = (0, 0, 1)) e' (1 + nu) alpha (D20 + D21) 60, che per l'acciaio fa 0. I moti rigidi non fanno
// lavoro, quindi le forze in x e in y sommano a 0 e anche il momento. Acciaio.
TEST(ElasticityLoad2d, AssemblyOfALargeMeshIsRight) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 100, 100);
    ASSERT_EQ(mesh.num_nodes(), 10201u);
    const Matrix3 D = steel();
    const double nu = 0.3;
    const double alpha = 12.0e-6;

    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        T[I] = reference_temperature + 10.0 + 100.0 * mesh.x(I);
    }
    const std::vector<double> F = assemble_thermal_load_2d(mesh, D, nu, alpha, T, reference_temperature);
    ASSERT_EQ(F.size(), 20402u);
    ASSERT_GT(largest_entry(F), 0.0);

    const double common = (1.0 + nu) * alpha * 60.0;
    const Work along_x = work_of(nodal_displacements(mesh, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}), F);
    EXPECT_NEAR(along_x.value, common * (D[0][0] + D[0][1]), 1e-11 * along_x.scale);
    const Work along_y = work_of(nodal_displacements(mesh, {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), F);
    EXPECT_NEAR(along_y.value, common * (D[1][0] + D[1][1]), 1e-11 * along_y.scale);
    const Work shear = work_of(nodal_displacements(mesh, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}), F);
    EXPECT_NEAR(shear.value, common * (D[2][0] + D[2][1]), 1e-11 * shear.scale);

    const std::vector<LinearField> motions = rigid_motions();
    for (std::size_t r = 0; r < motions.size(); ++r) {
        const Work w = work_of(nodal_displacements(mesh, motions[r]), F);
        EXPECT_NEAR(w.value, 0.0, 1e-11 * w.scale) << "moto rigido numero " << r;
    }
}