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

// ---- Il triangolo che si deforma: la matrice B --------------------------------------------------------------------
//
// Nel calore l'incognita di un triangolo erano 3 temperature, una per nodo, e dentro il triangolo la temperatura era la
// funzione lineare  T = N_0 T_0 + N_1 T_1 + N_2 T_2.  Ora le incognite sono gli SPOSTAMENTI (u, v): ogni nodo ne ha
// due, quindi un triangolo ha 6 numeri, nell'ordine
//
//     u_e = (u_0, v_0, u_1, v_1, u_2, v_2)        (per ogni nodo prima x e poi y)
//
// e dentro il triangolo gli spostamenti sono lineari, con le stesse funzioni N_i del calore:
//
//     u(x, y) = N_0 u_0 + N_1 u_1 + N_2 u_2,        v(x, y) = N_0 v_0 + N_1 v_1 + N_2 v_2.
//
// Le derivate le conosciamo gia' (vedi element_stiffness): grad(N_i) = (b_i, c_i) / (2 A), con p = (i+1) % 3,
// q = (i+2) % 3, b_i = y_p - y_q e c_i = x_q - x_p. Quindi, per le tre deformazioni del piano,
//
//     eps_xx   = du/dx           = sum_i  b_i u_i / (2 A)
//     eps_yy   = dv/dy           = sum_i  c_i v_i / (2 A)
//     gamma_xy = du/dy + dv/dx   = sum_i (c_i u_i + b_i v_i) / (2 A)
//
// Sono tre combinazioni lineari dei 6 numeri: eps = B u_e, con la matrice 3 x 6
//
//               [ b_0   0    b_1   0    b_2   0  ]
//     1 / (2 A) [  0   c_0    0   c_1    0   c_2 ]       (righe: eps_xx, eps_yy, gamma_xy;
//               [ c_0  b_0   c_1  b_1   c_2  b_2 ]        colonne: u_0, v_0, u_1, v_1, u_2, v_2).
//
// I gradienti sono costanti, quindi la deformazione e' la STESSA in tutto il triangolo ("triangolo a deformazione
// costante"). Una funzione lineare e' riprodotta esattamente: e' il controllo piu' forte (ESEMPIO 126).

// B ha 3 righe (le deformazioni) e 6 colonne (i gradi di liberta'), u_e ha 6 numeri: se nei tipi i numeri sono
// scambiati o sbagliati il test non compila e dice quale.
static_assert(std::tuple_size<Matrix3x6>::value == 3, "Matrix3x6 deve avere 3 righe");
static_assert(std::tuple_size<Matrix3x6::value_type>::value == 6, "Matrix3x6 deve avere 6 colonne");
static_assert(std::tuple_size<Vector6>::value == 6, "Vector6 deve avere 6 numeri");

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

// Sei triangoli antiorari, di forme e dimensioni molto diverse (gli stessi dei test di element_stiffness).
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // con un angolo retto
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scaleno
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scaleno, lontano dall'origine
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // con coordinate negative
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // molto schiacciato
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // piccolo (millimetri)
}

// Tre triangoli antiorari con nodi in comune e aree diverse (2, 1.75 e 1.5): servono a vedere se B usa i nodi e l'area
// del triangolo giusto della mesh.
Mesh2D irregular_mesh() {
    return Mesh2D({0.0, 2.0, 3.0, 1.0, -1.0}, {0.0, 0.0, 1.5, 2.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {0, 3, 4}});
}

// Un campo di spostamento lineare:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// In ogni punto ha la stessa deformazione: (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// I sei numeri (u_0, v_0, u_1, v_1, u_2, v_2) del campo f nei tre nodi del triangolo e della mesh.
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

// La deformazione eps = B u.
Vector3 strain_of(const Matrix3x6& B, const Vector6& u) {
    Vector3 eps{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            eps[i] += B[i][j] * u[j];
        }
    }
    return eps;
}

// Il piu' grande degli elementi di B in valore assoluto (per fissare le tolleranze).
double largest(const Matrix3x6& B) {
    double m = 0.0;
    for (const auto& row : B) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}

// Controlla che B del triangolo e dia la deformazione esatta (a1, b2, a2 + b1) del campo f.
void expect_exact_strain(const Mesh2D& mesh, std::size_t e, const LinearField& f, double tol, const std::string& what) {
    const Vector3 got = strain_of(strain_displacement_matrix(mesh, e), nodal_values(mesh, e, f));
    const Vector3 want = {f.a1, f.b2, f.a2 + f.b1};
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ", componente " << i;
    }
}

}  // namespace

// ESEMPIO 125. A mano. Il triangolo (0,0), (1,0), (0,1) ha A = 1/2, quindi 2 A = 1:
//     i = 0 (p = 1, q = 2):  b_0 = y_1 - y_2 = 0 - 1 = -1,   c_0 = x_2 - x_1 = 0 - 1 = -1
//     i = 1 (p = 2, q = 0):  b_1 = y_2 - y_0 = 1 - 0 =  1,   c_1 = x_0 - x_2 = 0 - 0 =  0
//     i = 2 (p = 0, q = 1):  b_2 = y_0 - y_1 = 0 - 0 =  0,   c_2 = x_1 - x_0 = 1 - 0 =  1
// e quindi
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// Si legge cosi': spostare il solo nodo 1 di 1 in x (u_1 = 1) stira il triangolo in x di 1 (eps_xx = 1) e basta;
// v_2 = 1 lo stira in y; u_2 = 1 e v_1 = 1 danno ciascuno uno scorrimento gamma_xy = 1.
// Il triangolo (0,0), (2,0), (0,1) ha A = 1, quindi 2 A = 2, con b = (-1, 1, 0) e c = (-2, 0, 2):
//     B = [[-0.5,  0,   0.5, 0,   0, 0],
//          [ 0,   -1,   0,   0,   0, 1],
//          [-1,   -0.5, 0,   0.5, 1, 0]].
TEST(StrainDisplacementMatrix, MatchesHandComputation) {
    const double want_unit[3][6] = {{-1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                    {0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
                                    {-1.0, -1.0, 0.0, 1.0, 1.0, 0.0}};
    const Matrix3x6 B1 = strain_displacement_matrix(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(B1[i][j], want_unit[i][j], 1e-14) << "triangolo (0,0) (1,0) (0,1), B[" << i << "][" << j << "]";
        }
    }

    const double want_wide[3][6] = {{-0.5, 0.0, 0.5, 0.0, 0.0, 0.0},
                                    {0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
                                    {-1.0, -0.5, 0.0, 0.5, 1.0, 0.0}};
    const Matrix3x6 B2 = strain_displacement_matrix(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(B2[i][j], want_wide[i][j], 1e-14) << "triangolo (0,0) (2,0) (0,1), B[" << i << "][" << j << "]";
        }
    }
}

// ESEMPIO 126. B deve dare la deformazione ESATTA di ogni campo di spostamento lineare, perche' gli spostamenti del
// triangolo sono lineari. Si prende  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y, si calcolano i 6 numeri nei nodi e si
// controlla che B u_e dia (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1). Sei campi "puri": le due traslazioni
// (deformazione 0), u = x (eps_xx = 1), v = y (eps_yy = 1), u = y e v = x (gamma_xy = 1 ciascuno); piu' un campo
// qualunque con tutti i numeri diversi da 0. Provati su sei triangoli di forme molto diverse.
// A mano, sul triangolo (0,0), (2,0), (0,1) con u = 0.1 + 0.01 x + 0.02 y,  v = -0.2 + 0.03 x + 0.04 y:
//     u_e = (0.1, -0.2, 0.12, -0.14, 0.12, -0.16)
//     eps_xx   = -0.5 * 0.1 + 0.5 * 0.12                                          = 0.01  (= a1)
//     eps_yy   = -1 * (-0.2) + 1 * (-0.16)                                        = 0.04  (= b2)
//     gamma_xy = -1 * 0.1 - 0.5 * (-0.2) + 0.5 * (-0.14) + 1 * 0.12               = 0.05  (= a2 + b1 = 0.02 + 0.03)
TEST(StrainDisplacementMatrix, ReproducesTheStrainOfLinearFields) {
    const Mesh2D hand = mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0});
    const Vector6 u_hand = {0.1, -0.2, 0.12, -0.14, 0.12, -0.16};
    const Vector3 eps_hand = strain_of(strain_displacement_matrix(hand, 0), u_hand);
    EXPECT_NEAR(eps_hand[0], 0.01, 1e-14);
    EXPECT_NEAR(eps_hand[1], 0.04, 1e-14);
    EXPECT_NEAR(eps_hand[2], 0.05, 1e-14);

    const LinearField fields[] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},                    // traslazione in x
                                  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},                    // traslazione in y
                                  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},                    // u = x
                                  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},                    // v = y
                                  {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},                    // u = y
                                  {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},                    // v = x
                                  {0.3, 1.2e-3, -0.7e-3, -0.2, 0.5e-3, 2.0e-3}};     // un campo qualunque
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 7; ++m) {
            const std::string what = "triangolo " + std::to_string(k) + ", campo " + std::to_string(m);
            expect_exact_strain(mesh, 0, fields[m], 1e-11, what);
        }
    }
}

// ESEMPIO 127. Un moto rigido non deforma. Una traslazione (tx, ty) e una piccola rotazione di un angolo theta
// attorno al punto (xc, yc) danno gli spostamenti
//     u = tx - theta (y - yc),        v = ty + theta (x - xc),
// che sono lineari: du/dx = 0, dv/dy = 0 e gamma_xy = du/dy + dv/dx = -theta + theta = 0. Quindi B u_e deve essere
// (0, 0, 0), qualunque sia il triangolo. Vale solo con lo scorrimento gamma_xy = du/dy + dv/dx: con un solo dei due
// termini una rotazione sembrerebbe una deformazione. (Una B tutta zeri passerebbe questo controllo senza dimostrare
// niente: per questo prima si controlla che B non sia nulla.)
TEST(StrainDisplacementMatrix, RigidMotionsDoNotDeform) {
    const double centers[2][2] = {{0.0, 0.0}, {0.7, -0.3}};
    const double angles[4] = {0.0, 0.01, 0.3, -0.5};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        const Matrix3x6 B = strain_displacement_matrix(mesh, 0);
        ASSERT_GT(largest(B), 0.0) << "B e' tutta zeri (triangolo " << k << "): una B nulla non deforma mai";
        for (const auto& center : centers) {
            for (double theta : angles) {
                const double tx = 0.2;
                const double ty = -0.4;
                const LinearField rigid = {tx + theta * center[1], 0.0, -theta, ty - theta * center[0], theta, 0.0};
                const Vector3 eps = strain_of(B, nodal_values(mesh, 0, rigid));
                for (std::size_t i = 0; i < 3; ++i) {
                    EXPECT_NEAR(eps[i], 0.0, 1e-11)
                        << "triangolo " << k << ", theta = " << theta << ", componente " << i;
                }
            }
        }
    }
}

// ESEMPIO 128. Come B cambia con il triangolo. Se il triangolo si ingrandisce di s (e magari si sposta), i lati b_i e
// c_i crescono di s e 2 A di s^2: B diventa B / s. Una deformazione e' uno spostamento diviso una lunghezza, quindi con
// gli stessi spostamenti nei nodi un triangolo piu' grande si deforma meno. Spostare il triangolo senza ingrandirlo
// (s = 1) non cambia B.
TEST(StrainDisplacementMatrix, ScalesWithTheInverseOfTheSize) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Matrix3x6 base = strain_displacement_matrix(mesh_of(triangles[k]), 0);
        ASSERT_GT(largest(base), 0.0) << "B e' tutta zeri (triangolo " << k << ")";
        for (double s : scales) {
            for (const auto& shift : shifts) {
                const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                const Matrix3x6 B = strain_displacement_matrix(mesh_of(moved), 0);
                const double tol = 1e-10 * largest(base) / s;
                for (std::size_t i = 0; i < 3; ++i) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        EXPECT_NEAR(B[i][j], base[i][j] / s, tol)
                            << "triangolo " << k << ", s = " << s << ", B[" << i << "][" << j << "]";
                    }
                }
            }
        }
    }
}

// ESEMPIO 129. B di un triangolo di una mesh vera, dove i nodi del triangolo non sono 0, 1, 2 e i triangoli hanno aree
// diverse. Ogni triangolo, preso con i suoi nodi (mesh.triangle(e)) e la sua area (mesh.area(e)), deve riprodurre la
// stessa deformazione costante di un campo lineare dato sui nodi della mesh. Provato su una mesh a mano di 3 triangoli
// irregolari e sulla mesh rettangolare di 4 x 3 celle (24 triangoli).
TEST(StrainDisplacementMatrix, WorksOnEveryTriangleOfAMesh) {
    const LinearField field = {0.3, 1.2e-3, -0.7e-3, -0.2, 0.5e-3, 2.0e-3};

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        expect_exact_strain(irregular, e, field, 1e-12, "mesh irregolare, triangolo " + std::to_string(e));
    }

    const Mesh2D rectangle = rectangle_mesh(2.0, 1.5, 4, 3);
    ASSERT_EQ(rectangle.num_triangles(), 24u);
    for (std::size_t e = 0; e < rectangle.num_triangles(); ++e) {
        expect_exact_strain(rectangle, e, field, 1e-12, "mesh rettangolare, triangolo " + std::to_string(e));
    }
}

// ESEMPIO 130. Un triangolo che non esiste: std::out_of_range, come per le altre funzioni dell'elemento.
TEST(StrainDisplacementMatrix, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    EXPECT_NO_THROW(strain_displacement_matrix(mesh, 0));
    EXPECT_NO_THROW(strain_displacement_matrix(mesh, mesh.num_triangles() - 1));
    EXPECT_THROW(strain_displacement_matrix(mesh, mesh.num_triangles()), std::out_of_range);
    EXPECT_THROW(strain_displacement_matrix(mesh, 1000), std::out_of_range);
}