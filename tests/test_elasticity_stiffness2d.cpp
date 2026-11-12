#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- La rigidezza del triangolo: Ke = A B^T D B ---------------------------------------------------------------------
//
// Dalla matrice B si arriva alla rigidezza con l'ENERGIA. Un triangolo deformato accumula energia: per unita' di area
// e' (1/2) eps^T D eps (come una molla, (1/2) k x^2) e, siccome eps e' costante nel triangolo, basta moltiplicare per
// l'area A. Con eps = B u_e:
//
//     U = (1/2) A (B u_e)^T D (B u_e) = (1/2) u_e^T (A B^T D B) u_e = (1/2) u_e^T Ke u_e.
//
// Come per la molla, dove U = (1/2) k x^2 da' la forza k x, le 6 forze sui gradi di liberta' del triangolo sono la
// derivata di U rispetto ai 6 spostamenti: f_e = Ke u_e. Quindi Ke (6 x 6, spessore 1) e' la "costante della molla" del
// triangolo, e Ke[i][j] e' la forza sul grado di liberta' i quando si sposta di 1 il solo grado di liberta' j.
//
// Cosa deve avere una vera rigidezza, ed e' quello che i test controllano:
//   - e' simmetrica (la forza su i spostando j di 1 e' quella su j spostando i di 1);
//   - un moto rigido (traslazione, rotazione) non deforma, quindi niente energia e niente forze: Ke u_rigido = 0;
//   - un campo lineare ha deformazione costante: le sue forze sono quelle della tensione sigma = D eps e la sua energia
//     e' A eps^T D eps;
//   - non dipende dalla dimensione del triangolo: A cresce come s^2 e B^T B cala come 1/s^2. (Per spostare di d un
//     lato lungo L serve una forza = tensione x lunghezza = E (d / L) L = E d, che non dipende da L);
//   - un materiale isotropo non ha una direzione preferita: ruotando il triangolo e gli spostamenti, le forze ruotano
//     allo stesso modo, K' = Q K Q^T.

// Ke e' una tabella 6 x 6: se il tipo ha altre dimensioni il test non compila e dice quale.
static_assert(std::tuple_size<Matrix6>::value == 6, "Matrix6 deve avere 6 righe");
static_assert(std::tuple_size<Matrix6::value_type>::value == 6, "Matrix6 deve avere 6 colonne");

namespace {

// I vertici di un triangolo, in senso antiorario.
struct Tri {
    double x0, y0, x1, y1, x2, y2;
};

// Una mesh con quel solo triangolo (nodi 0, 1, 2 in quest'ordine).
Mesh2D mesh_of(const Tri& t) {
    return Mesh2D({t.x0, t.x1, t.x2}, {t.y0, t.y1, t.y2}, {{0, 1, 2}});
}

// Il triangolo t ingrandito di s (rispetto all'origine) e spostato di (dx, dy).
Tri scaled_and_shifted(const Tri& t, double s, double dx, double dy) {
    return {s * t.x0 + dx, s * t.y0 + dy, s * t.x1 + dx, s * t.y1 + dy, s * t.x2 + dx, s * t.y2 + dy};
}

// Il triangolo t ruotato di phi (in radianti) attorno all'origine: (x, y) diventa (c x - s y, s x + c y).
Tri rotated(const Tri& t, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    return {c * t.x0 - s * t.y0, s * t.x0 + c * t.y0, c * t.x1 - s * t.y1,
            s * t.x1 + c * t.y1, c * t.x2 - s * t.y2, s * t.x2 + c * t.y2};
}

// Sei triangoli antiorari, di forme e dimensioni molto diverse (gli stessi dei test di B).
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // con un angolo retto
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scaleno
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scaleno, lontano dall'origine
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // con coordinate negative
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // molto schiacciato
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // piccolo (millimetri)
}

// Tre triangoli antiorari con nodi in comune e aree diverse (2, 1.75 e 1.5).
Mesh2D irregular_mesh() {
    return Mesh2D({0.0, 2.0, 3.0, 1.0, -1.0}, {0.0, 0.0, 1.5, 2.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {0, 3, 4}});
}

// L'area del triangolo e dai suoi tre nodi (formula del determinante), senza usare mesh.area.
double triangle_area(const Mesh2D& mesh, std::size_t e) {
    const Triangle& t = mesh.triangle(e);
    const double x0 = mesh.x(t[0]);
    const double y0 = mesh.y(t[0]);
    return 0.5 * ((mesh.x(t[1]) - x0) * (mesh.y(t[2]) - y0) - (mesh.x(t[2]) - x0) * (mesh.y(t[1]) - y0));
}

// Materiali a mano (3 x 3, indici xx, yy, xy), simmetrici e definiti positivi.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 stretched() {  // xx e yy diversi: se gli indici di D sono scambiati si vede
    return {{{2.0, 1.0, 0.0}, {1.0, 3.0, 0.0}, {0.0, 0.0, 1.0}}};
}
Matrix3 general() {  // tutti i numeri diversi, anche l'accoppiamento con il taglio
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}

// Materiali veri (da plane_strain_matrix), fra cui uno rigido come l'acciaio, uno con nu < 0 e uno quasi
// incomprimibile.
std::vector<Matrix3> sample_materials() {
    return {general(), plane_strain_matrix(1.0, 0.25), plane_strain_matrix(210e9, 0.3), plane_strain_matrix(1.0, -0.5),
            plane_strain_matrix(1.0, 0.49)};
}

// Un campo di spostamento lineare:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// In ogni punto ha la stessa deformazione (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// I sei numeri (u_0, v_0, u_1, v_1, u_2, v_2) del campo f nei tre nodi del triangolo e.
Vector6 nodal_values(const Mesh2D& mesh, std::size_t e, const LinearField& f) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u{};
    for (std::size_t i = 0; i < 3; ++i) {
        const double x = mesh.x(t[i]);
        const double y = mesh.y(t[i]);
        u[2 * i] = f.a0 + f.a1 * x + f.a2 * y;
        u[2 * i + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u;
}

// I sei campi "puri", uno per ciascun numero a0, b0, a1, b2, a2, b1 (le due traslazioni, u = x, v = y, u = y, v = x),
// piu' un campo qualunque con tutti i numeri diversi da 0. I primi sei sono una base di tutti i campi lineari.
std::vector<LinearField> sample_fields() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3}};
}

// La deformazione e la tensione di un campo lineare, e il numero eps^T D eps.
Vector3 strain_of(const LinearField& f) {
    return {f.a1, f.b2, f.a2 + f.b1};
}
Vector3 stress_of(const Matrix3& D, const Vector3& eps) {
    Vector3 sigma{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            sigma[i] += D[i][j] * eps[j];
        }
    }
    return sigma;
}
double strain_times_stress(const Matrix3& D, const Vector3& eps) {
    const Vector3 sigma = stress_of(D, eps);
    return eps[0] * sigma[0] + eps[1] * sigma[1] + eps[2] * sigma[2];
}

// Le forze sui 6 gradi di liberta' prodotte da una tensione costante sigma: A B^T sigma. Per il nodo i, con
// p = (i+1) % 3, q = (i+2) % 3, b_i = y_p - y_q e c_i = x_q - x_p (come nella matrice B), sono
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2.
Vector6 nodal_forces(const Mesh2D& mesh, std::size_t e, const Vector3& sigma) {
    const Triangle& t = mesh.triangle(e);
    Vector6 f{};
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t p = (i + 1) % 3;
        const std::size_t q = (i + 2) % 3;
        const double b = mesh.y(t[p]) - mesh.y(t[q]);
        const double c = mesh.x(t[q]) - mesh.x(t[p]);
        f[2 * i] = (b * sigma[0] + c * sigma[2]) / 2.0;
        f[2 * i + 1] = (c * sigma[1] + b * sigma[2]) / 2.0;
    }
    return f;
}

// Il piu' grande dei numeri di Ke (o di u) in valore assoluto, per fissare le tolleranze.
double largest(const Matrix6& K) {
    double m = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}
double largest(const Vector6& u) {
    double m = 0.0;
    for (double v : u) {
        m = std::max(m, std::fabs(v));
    }
    return m;
}

// Il prodotto Ke u e il numero u^T Ke u.
Vector6 times(const Matrix6& K, const Vector6& u) {
    Vector6 f{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            f[i] += K[i][j] * u[j];
        }
    }
    return f;
}
double energy_of(const Matrix6& K, const Vector6& u) {
    const Vector6 f = times(K, u);
    double s = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
        s += u[i] * f[i];
    }
    return s;
}

// Q K Q^T, dove Q ha sulla diagonale tre blocchi 2 x 2 di rotazione R(phi) = [[c, -s], [s, c]], uno per nodo:
// e' la matrice di rigidezza di un triangolo ruotato di phi, se il materiale e' isotropo.
Matrix6 rotated_matrix(const Matrix6& K, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    Matrix6 Q{};
    for (std::size_t n = 0; n < 3; ++n) {
        Q[2 * n][2 * n] = c;
        Q[2 * n][2 * n + 1] = -s;
        Q[2 * n + 1][2 * n] = s;
        Q[2 * n + 1][2 * n + 1] = c;
    }
    Matrix6 R{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            for (std::size_t k = 0; k < 6; ++k) {
                for (std::size_t l = 0; l < 6; ++l) {
                    R[i][j] += Q[i][k] * K[k][l] * Q[j][l];
                }
            }
        }
    }
    return R;
}

}  // namespace

// ESEMPIO 131. A mano. Il triangolo (0,0), (1,0), (0,1) ha A = 1/2 e (vedi la matrice B)
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// Con E = 1 e nu = 1/4 (lambda = mu = 0.4) e' D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. Si calcola prima D B
// (3 x 6) e poi Ke[i][j] = A (colonna i di B) . (colonna j di D B):
//     D B = [[-1.2, -0.4, 1.2, 0,   0,   0.4],
//            [-0.4, -1.2, 0.4, 0,   0,   1.2],
//            [-0.4, -0.4, 0,   0.4, 0.4, 0  ]]
//     Ke[0][0] = 1/2 ((-1)(-1.2) + 0 (-0.4) + (-1)(-0.4)) = 0.8,     Ke[0][2] = 1/2 ((-1)(1.2) + 0 + (-1) 0) = -0.6,
// e cosi' per tutte le 36 entrate:
//     Ke = [[ 0.8,  0.4, -0.6, -0.2, -0.2, -0.2],
//           [ 0.4,  0.8, -0.2, -0.2, -0.2, -0.6],
//           [-0.6, -0.2,  0.6,  0,    0,    0.2],
//           [-0.2, -0.2,  0,    0.2,  0.2,  0  ],
//           [-0.2, -0.2,  0,    0.2,  0.2,  0  ],
//           [-0.2, -0.6,  0.2,  0,    0,    0.6]].
// La colonna 0 e' la forza quando solo u_0 = 1 (il nodo 0 si sposta di 1 in x): (0.8, 0.4, -0.6, -0.2, -0.2, -0.2). Le
// forze in x sommano a 0.8 - 0.6 - 0.2 = 0 e quelle in y a 0.4 - 0.2 - 0.2 = 0: nessuna forza netta sul triangolo.
// Il secondo triangolo, (0,0), (2,0), (0,1) con A = 1, ha un materiale con xx e yy diversi, D = [[2, 1, 0], [1, 3, 0],
// [0, 0, 1]], per vedere se ogni indice di D sta al suo posto:
//     D B = [[-1,   -1,   1,   0,   0,   1],
//            [-0.5, -3,   0.5, 0,   0,   3],
//            [-1,   -0.5, 0,   0.5, 1,   0]]
//     Ke = [[ 1.5,  1,    -0.5, -0.5,  -1,   -0.5],
//           [ 1,    3.25, -0.5, -0.25, -0.5, -3  ],
//           [-0.5, -0.5,   0.5,  0,     0,    0.5],
//           [-0.5, -0.25,  0,    0.25,  0.5,  0  ],
//           [-1,   -0.5,   0,    0.5,   1,    0  ],
//           [-0.5, -3,     0.5,  0,     0,    3  ]].
TEST(ElementElasticityStiffness, MatchesHandComputation) {
    const double want_unit[6][6] = {{0.8, 0.4, -0.6, -0.2, -0.2, -0.2},
                                    {0.4, 0.8, -0.2, -0.2, -0.2, -0.6},
                                    {-0.6, -0.2, 0.6, 0.0, 0.0, 0.2},
                                    {-0.2, -0.2, 0.0, 0.2, 0.2, 0.0},
                                    {-0.2, -0.2, 0.0, 0.2, 0.2, 0.0},
                                    {-0.2, -0.6, 0.2, 0.0, 0.0, 0.6}};
    const Matrix6 K1 = element_elasticity_stiffness(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0, hooke_quarter());
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(K1[i][j], want_unit[i][j], 1e-14)
                << "triangolo (0,0) (1,0) (0,1), Ke[" << i << "][" << j << "]";
        }
    }

    const double want_wide[6][6] = {{1.5, 1.0, -0.5, -0.5, -1.0, -0.5},
                                    {1.0, 3.25, -0.5, -0.25, -0.5, -3.0},
                                    {-0.5, -0.5, 0.5, 0.0, 0.0, 0.5},
                                    {-0.5, -0.25, 0.0, 0.25, 0.5, 0.0},
                                    {-1.0, -0.5, 0.0, 0.5, 1.0, 0.0},
                                    {-0.5, -3.0, 0.5, 0.0, 0.0, 3.0}};
    const Matrix6 K2 = element_elasticity_stiffness(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0, stretched());
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(K2[i][j], want_wide[i][j], 1e-14)
                << "triangolo (0,0) (2,0) (0,1), Ke[" << i << "][" << j << "]";
        }
    }
}

// ESEMPIO 132. Ke e' simmetrica: Ke[i][j] = Ke[j][i]. A B^T D B lo e' sempre se D e' simmetrica. Provata su sei
// triangoli e cinque materiali (anche rigidi come l'acciaio, con nu < 0 e quasi incomprimibili). Una Ke tutta zeri
// sarebbe simmetrica senza dimostrare niente: per questo prima si controlla che non sia nulla.
TEST(ElementElasticityStiffness, IsSymmetric) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            const double big = largest(K);
            ASSERT_GT(big, 0.0) << "Ke e' tutta zeri (triangolo " << k << ", materiale " << m << ")";
            for (std::size_t i = 0; i < 6; ++i) {
                for (std::size_t j = i + 1; j < 6; ++j) {
                    EXPECT_NEAR(K[i][j], K[j][i], 1e-13 * big)
                        << "triangolo " << k << ", materiale " << m << ", Ke[" << i << "][" << j << "]";
                }
            }
        }
    }
}

// ESEMPIO 133. Un moto rigido non deforma, quindi non da' forze: Ke u_rigido = 0. Quattro moti rigidi: traslazione in
// x, traslazione in y, rotazione di 1 attorno all'origine (u = -y, v = x) e rotazione di 0.5 attorno al punto (0.7,
// -0.3), cioe' u = -theta (y - yc), v = theta (x - xc). Sei triangoli e tre materiali. Sono tre moti rigidi
// indipendenti, quindi Ke da sola non si puo' invertire: per risolvere un problema serviranno i vincoli.
TEST(ElementElasticityStiffness, RigidMotionsProduceNoForce) {
    const LinearField rigid[4] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                  {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},
                                  {-0.15, 0.0, -0.5, -0.35, 0.5, 0.0}};
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 3; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            ASSERT_GT(largest(K), 0.0) << "Ke e' tutta zeri (triangolo " << k << ", materiale " << m << ")";
            for (std::size_t r = 0; r < 4; ++r) {
                const Vector6 u = nodal_values(mesh, 0, rigid[r]);
                const Vector6 f = times(K, u);
                const double tol = 1e-13 * largest(K) * largest(u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(f[i], 0.0, tol)
                        << "triangolo " << k << ", materiale " << m << ", moto " << r << ", forza " << i;
                }
            }
        }
    }
}

// ESEMPIO 134. Le forze di un campo lineare. Un campo lineare ha deformazione eps = (a1, b2, a2 + b1) e tensione
// sigma = D eps, costanti. Le forze che ne nascono sui sei gradi di liberta' sono A B^T sigma, cioe' per il nodo i
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2,
// con b_i e c_i dalle coordinate dei nodi. Ke u_e deve dare queste forze. I primi sei campi sono una base: se Ke da' le
// forze giuste per tutti, e' giusta in ognuna delle 36 entrate. Provato su sei triangoli e due materiali.
TEST(ElementElasticityStiffness, GivesTheForcesOfLinearFields) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t r = 0; r < fields.size(); ++r) {
                const Vector6 u = nodal_values(mesh, 0, fields[r]);
                const Vector6 got = times(K, u);
                const Vector6 want = nodal_forces(mesh, 0, stress_of(materials[m], strain_of(fields[r])));
                const double tol = 1e-13 * largest(K) * largest(u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], want[i], tol)
                        << "triangolo " << k << ", materiale " << m << ", campo " << r << ", forza " << i;
                }
            }
        }
    }
}

// ESEMPIO 135. L'energia di un campo lineare. Con eps = (a1, b2, a2 + b1) costante in tutto il triangolo, l'energia
// u_e^T Ke u_e e' A eps^T D eps (l'area A dalle coordinate dei nodi). Non usa B e non usa le formule di b_i e c_i: e'
// la definizione di Ke come molla. Vale 0 per un moto rigido e deve essere positiva per ogni vera deformazione.
TEST(ElementElasticityStiffness, StrainEnergyOfLinearFieldsIsExact) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t r = 0; r < fields.size(); ++r) {
                const Vector6 u = nodal_values(mesh, 0, fields[r]);
                const double want = triangle_area(mesh, 0) * strain_times_stress(materials[m], strain_of(fields[r]));
                const double tol = 1e-13 * largest(K) * largest(u) * largest(u);
                EXPECT_NEAR(energy_of(K, u), want, tol)
                    << "triangolo " << k << ", materiale " << m << ", campo " << r;
            }
        }
    }
}

// ESEMPIO 136. La rigidezza non dipende dalla dimensione. Se il triangolo si ingrandisce di s (e magari si sposta), A
// cresce di s^2 e B^T B cala di 1/s^2: Ke e' la stessa. (In 2D e con spessore 1 una lastra grande non e' piu' rigida
// di una piccola: la forza per spostare di d un lato lungo L e' E (d / L) L = E d.) Provato con s = 1, 2, 0.25 e 10.
TEST(ElementElasticityStiffness, DoesNotDependOnTheSize) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.3)};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 base = element_elasticity_stiffness(mesh_of(triangles[k]), 0, materials[m]);
            ASSERT_GT(largest(base), 0.0) << "Ke e' tutta zeri (triangolo " << k << ", materiale " << m << ")";
            for (double s : scales) {
                for (const auto& shift : shifts) {
                    const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                    const Matrix6 K = element_elasticity_stiffness(mesh_of(moved), 0, materials[m]);
                    for (std::size_t i = 0; i < 6; ++i) {
                        for (std::size_t j = 0; j < 6; ++j) {
                            EXPECT_NEAR(K[i][j], base[i][j], 1e-10 * largest(base))
                                << "triangolo " << k << ", materiale " << m << ", s = " << s << ", Ke[" << i << "]["
                                << j << "]";
                        }
                    }
                }
            }
        }
    }
}

// ESEMPIO 137. Un materiale isotropo non ha una direzione preferita. Se si ruota il triangolo di phi attorno
// all'origine e si ruotano anche gli spostamenti dei nodi (u' = R u in ogni nodo, cioe' u_e' = Q u_e), le forze
// ruotano allo stesso modo: f' = Q f. Siccome f = K u, questo vuol dire K' = Q K Q^T. Serve la D vera di
// plane_strain_matrix: il taglio mu e lo scorrimento gamma = du/dy + dv/dx stanno insieme solo con le convenzioni
// giuste (D[2][2] = mu), altrimenti Ke dipenderebbe dalla direzione. Provato con 4 angoli, sei triangoli e due nu.
TEST(ElementElasticityStiffness, IsIndifferentToRotation) {
    const double angles[4] = {0.3, 1.5707963267948966, -1.0, 3.0};
    const Matrix3 materials[2] = {plane_strain_matrix(1.0, 0.3), plane_strain_matrix(1.0, 0.25)};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh_of(triangles[k]), 0, materials[m]);
            ASSERT_GT(largest(K), 0.0) << "Ke e' tutta zeri (triangolo " << k << ", materiale " << m << ")";
            for (double phi : angles) {
                const Matrix6 got = element_elasticity_stiffness(mesh_of(rotated(triangles[k], phi)), 0, materials[m]);
                const Matrix6 want = rotated_matrix(K, phi);
                for (std::size_t i = 0; i < 6; ++i) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        EXPECT_NEAR(got[i][j], want[i][j], 1e-12 * largest(K))
                            << "triangolo " << k << ", materiale " << m << ", phi = " << phi << ", Ke[" << i
                            << "][" << j << "]";
                    }
                }
            }
        }
    }
}

// ESEMPIO 138. Ke di un triangolo di una mesh vera, dove i nodi non sono 0, 1, 2 e le aree sono diverse. In ogni
// triangolo e di un campo lineare, con i valori nei nodi della mesh, l'energia u_e^T Ke u_e deve essere
// area(e) eps^T D eps. Sulla mesh rettangolare di 4 x 3 celle (24 triangoli) la somma e' l'energia di tutta la lastra,
// cioe' (area della lastra) eps^T D eps = 2 x 1.5 x eps^T D eps. Provato anche su 3 triangoli irregolari a mano.
TEST(ElementElasticityStiffness, WorksOnEveryTriangleOfAMesh) {
    const LinearField field = {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3};
    const Matrix3 D = hooke_quarter();
    const double psi = strain_times_stress(D, strain_of(field));

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        const Matrix6 K = element_elasticity_stiffness(irregular, e, D);
        const double want = triangle_area(irregular, e) * psi;
        EXPECT_NEAR(energy_of(K, nodal_values(irregular, e, field)), want, 1e-12 * want)
            << "mesh irregolare, triangolo " << e;
    }

    const Mesh2D rectangle = rectangle_mesh(2.0, 1.5, 4, 3);
    ASSERT_EQ(rectangle.num_triangles(), 24u);
    double total = 0.0;
    for (std::size_t e = 0; e < rectangle.num_triangles(); ++e) {
        const Matrix6 K = element_elasticity_stiffness(rectangle, e, D);
        const double energy = energy_of(K, nodal_values(rectangle, e, field));
        EXPECT_NEAR(energy, triangle_area(rectangle, e) * psi, 1e-12 * triangle_area(rectangle, e) * psi)
            << "mesh rettangolare, triangolo " << e;
        total += energy;
    }
    EXPECT_NEAR(total, 2.0 * 1.5 * psi, 1e-12 * 2.0 * 1.5 * psi);
}

// ESEMPIO 139. Un triangolo che non esiste: std::out_of_range, come per B e per le altre funzioni dell'elemento.
TEST(ElementElasticityStiffness, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    const Matrix3 D = hooke_quarter();
    EXPECT_NO_THROW(element_elasticity_stiffness(mesh, 0, D));
    EXPECT_NO_THROW(element_elasticity_stiffness(mesh, mesh.num_triangles() - 1, D));
    EXPECT_THROW(element_elasticity_stiffness(mesh, mesh.num_triangles(), D), std::out_of_range);
    EXPECT_THROW(element_elasticity_stiffness(mesh, 1000, D), std::out_of_range);
}