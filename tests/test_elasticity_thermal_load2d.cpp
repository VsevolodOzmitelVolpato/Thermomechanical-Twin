#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- La forza termica del triangolo: f_th = A B^T D eps_th ---------------------------------------------------------
//
// Una molla scaldata vorrebbe allungarsi di x_th. Se e' libera lo fa e non nasce nessuna forza; se la si porta a x la
// forza e' k (x - x_th) = k x - k x_th. Il pezzo k x_th e' la "forza termica": e' la forza che servirebbe per
// stirare di x_th la molla fredda.
// Per il triangolo e' lo stesso. La deformazione e' eps = B u_e, quella che il calore vorrebbe e' eps_th, e la tensione
// nasce dalla differenza, sigma = D (eps - eps_th). L'energia, per l'area A del triangolo, e'
//     U = (1/2) A (B u_e - eps_th)^T D (B u_e - eps_th)
// e la sua derivata rispetto ai 6 spostamenti, cioe' le 6 forze sui gradi di liberta', e'
//     A B^T D (B u_e - eps_th) = Ke u_e - f_th,        con        f_th = A B^T D eps_th.
// Quindi f_th e' la formula A B^T sigma (le forze di una tensione costante) con la tensione D eps_th, quella che il
// calore vorrebbe produrre. Un triangolo scaldato e libero, senza forze esterne, si sposta in modo che Ke u_e = f_th.
// Le f_th spingono ogni nodo verso l'esterno (il triangolo vuole espandersi) ma sono in equilibrio fra loro.
//
// Cosa deve avere una vera forza termica, ed e' quello che i test controllano:
//   - e' la A B^T sigma di sigma = D eps_th, e e' lineare in eps_th: senza calore, niente forza;
//   - una deformazione termica costante non spinge il triangolo e non lo fa ruotare: la somma delle forze e il loro
//     momento sono 0 (un moto rigido non fa lavoro);
//   - il lavoro delle f_th su un campo lineare u e' A eps(u)^T D eps_th;
//   - se il triangolo si dilata liberamente, con eps = eps_th e quindi sigma = 0, le forze dell'elasticita' Ke u_e sono
//     proprio le f_th (e' lo stesso di dire che Ke u_th = f_th, come per la molla);
//   - e' "tensione per lunghezza": ingrandire il triangolo di s moltiplica le forze per s (la rigidezza Ke, invece, non
//     cambia);
//   - un materiale isotropo non ha una direzione preferita: ruotando triangolo e deformazione, le forze ruotano;
//   - su una mesh intera, un riscaldamento uniforme non carica i nodi interni (le forze dei triangoli attorno si
//     annullano): resta solo la forza sul bordo, sigma per la normale uscente per la lunghezza.

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

// Sei triangoli antiorari, di forme e dimensioni molto diverse (gli stessi dei test di B e di Ke).
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

// Deformazioni termiche (eps_xx, eps_yy, gamma_xy) di prova: una unitaria uguale in x e y, una vera, (1 + nu) alpha dT
// con nu = 0.3, alpha = 12e-6 e dT = 50, un raffreddamento e due generali con il taglio. (Il calore di un materiale
// isotropo non da' taglio, ma la funzione vale per ogni eps_th e il taglio fa vedere se gli indici di D e B sono
// giusti.)
std::vector<Vector3> sample_thermal_strains() {
    return {{1.0, 1.0, 0.0},
            {7.8e-4, 7.8e-4, 0.0},
            {-3.0e-3, -3.0e-3, 0.0},
            {2.0e-3, -1.0e-3, 1.5e-3},
            {1.0, 2.0, 1.0}};
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
// piu' un campo qualunque con tutti i numeri diversi da 0.
std::vector<LinearField> sample_fields() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3}};
}

// Il campo di una dilatazione libera: parte dall'origine e ha deformazione costante eps = (eps_xx, eps_yy, gamma_xy),
//     u = eps_xx x + (gamma / 2) y,        v = eps_yy y + (gamma / 2) x.
// Con eps = (e, e, 0) e' u = e x, v = e y.
LinearField free_expansion(const Vector3& eps) {
    return {0.0, eps[0], 0.5 * eps[2], 0.0, 0.5 * eps[2], eps[1]};
}

// La deformazione di un campo lineare, la tensione D eps e il prodotto scalare di due terne.
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
double product(const Vector3& a, const Vector3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

// Il piu' grande dei numeri di una tabella o di un vettore in valore assoluto, per fissare le tolleranze.
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
double largest(const Vector3& u) {
    return std::max(std::max(std::fabs(u[0]), std::fabs(u[1])), std::fabs(u[2]));
}

// Il prodotto Ke u e il lavoro u . f delle forze f sugli spostamenti u.
Vector6 times(const Matrix6& K, const Vector6& u) {
    Vector6 f{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            f[i] += K[i][j] * u[j];
        }
    }
    return f;
}
double work_of(const Vector6& u, const Vector6& f) {
    double s = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
        s += u[i] * f[i];
    }
    return s;
}

// Le forze f di un triangolo ruotate di phi: in ogni nodo (f_x, f_y) diventa (c f_x - s f_y, s f_x + c f_y).
Vector6 rotated_forces(const Vector6& f, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    Vector6 r{};
    for (std::size_t n = 0; n < 3; ++n) {
        r[2 * n] = c * f[2 * n] - s * f[2 * n + 1];
        r[2 * n + 1] = s * f[2 * n] + c * f[2 * n + 1];
    }
    return r;
}

// La deformazione (eps_xx, eps_yy, gamma_xy) vista da assi ruotati di phi, cioe' R eps R^T con eps il tensore
// [[eps_xx, g], [g, eps_yy]] e g = gamma / 2 (lo scorrimento "ingegneristico" e' il doppio del tensore). Con
// c = cos(phi) e s = sin(phi):
//     eps_xx' = c^2 eps_xx + s^2 eps_yy - 2 s c g,        eps_yy' = s^2 eps_xx + c^2 eps_yy + 2 s c g,
//     gamma'  = 2 (s c (eps_xx - eps_yy) + (c^2 - s^2) g).
Vector3 rotated_strain(const Vector3& eps, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    const double g = 0.5 * eps[2];
    return {c * c * eps[0] + s * s * eps[1] - 2.0 * s * c * g, s * s * eps[0] + c * c * eps[1] + 2.0 * s * c * g,
            2.0 * (s * c * (eps[0] - eps[1]) + (c * c - s * s) * g)};
}

}  // namespace

// ESEMPIO 140. A mano. Il triangolo (0,0), (1,0), (0,1) ha A = 1/2 e (vedi la matrice B)
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// Con E = 1 e nu = 1/4 (lambda = mu = 0.4) e' D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. Per eps_th = (1, 1, 0)
// la tensione e' D eps_th = (1.2 + 0.4, 0.4 + 1.2, 0) = (1.6, 1.6, 0), e le forze sono f_th = A B^T (1.6, 1.6, 0): il
// grado di liberta' j ha A (B[0][j] 1.6 + B[1][j] 1.6 + B[2][j] 0), cioe'
//     f_th = 1/2 (-1.6, -1.6, 1.6, 0, 0, 1.6) = (-0.8, -0.8, 0.8, 0, 0, 0.8).
// I segni hanno senso: il nodo 0, in (0,0), e' spinto verso (-, -), il nodo 1, in (1,0), verso +x e il nodo 2, in
// (0,1), verso +y: ogni nodo e' spinto verso l'esterno, come se il triangolo volesse espandersi.
// Il secondo triangolo, (0,0), (2,0), (0,1) con A = 1, ha un materiale con xx e yy diversi, D = [[2, 1, 0], [1, 3, 0],
// [0, 0, 1]], e una deformazione generale con il taglio, eps_th = (1, 2, 1): D eps_th = (4, 7, 1) e, con
//     B = [[-0.5, 0, 0.5, 0, 0, 0], [0, -1, 0, 0, 0, 1], [-1, -0.5, 0, 0.5, 1, 0]],
//     f_th = A B^T (4, 7, 1) = (-3, -7.5, 2, 0.5, 1, 7).
TEST(ElementThermalLoad, MatchesHandComputation) {
    const double want_unit[6] = {-0.8, -0.8, 0.8, 0.0, 0.0, 0.8};
    const Vector6 f1 =
        element_thermal_load(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0, hooke_quarter(), {1.0, 1.0, 0.0});
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(f1[i], want_unit[i], 1e-14) << "triangolo (0,0) (1,0) (0,1), f[" << i << "]";
    }

    const double want_wide[6] = {-3.0, -7.5, 2.0, 0.5, 1.0, 7.0};
    const Vector6 f2 =
        element_thermal_load(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0, stretched(), {1.0, 2.0, 1.0});
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(f2[i], want_wide[i], 1e-14) << "triangolo (0,0) (2,0) (0,1), f[" << i << "]";
    }
}

// ESEMPIO 141. Le forze di una tensione costante. La tensione che il calore vorrebbe produrre e' sigma = D eps_th,
// costante nel triangolo, e le forze sono A B^T sigma, cioe' per il nodo i
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2,
// con b_i e c_i dalle coordinate dei nodi (qui sono scritte per esteso, senza B). Provato su sei triangoli, cinque
// materiali e cinque deformazioni termiche, e sui tre triangoli di una mesh irregolare (con i nodi in comune e le aree
// diverse, 2, 1.75 e 1.5), dove il triangolo e non e' sempre il numero 0. Senza calore (eps_th = 0) le forze sono 0,
// esattamente.
TEST(ElementThermalLoad, GivesTheForcesOfAConstantStress) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<Vector3> strains = sample_thermal_strains();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 got = element_thermal_load(mesh, 0, materials[m], strains[s]);
                const Vector6 want = nodal_forces(mesh, 0, stress_of(materials[m], strains[s]));
                ASSERT_GT(largest(want), 0.0) << "triangolo " << k << ", materiale " << m << ", eps_th " << s;
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], want[i], 1e-13 * largest(want))
                        << "triangolo " << k << ", materiale " << m << ", eps_th " << s << ", forza " << i;
                }
            }
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Vector6 none = element_thermal_load(mesh, 0, materials[m], {0.0, 0.0, 0.0});
            for (std::size_t i = 0; i < 6; ++i) {
                EXPECT_EQ(none[i], 0.0) << "senza calore, triangolo " << k << ", materiale " << m << ", forza " << i;
            }
        }
    }

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        for (std::size_t s = 0; s < strains.size(); ++s) {
            const Vector6 got = element_thermal_load(irregular, e, materials[0], strains[s]);
            const Vector6 want = nodal_forces(irregular, e, stress_of(materials[0], strains[s]));
            for (std::size_t i = 0; i < 6; ++i) {
                EXPECT_NEAR(got[i], want[i], 1e-13 * largest(want))
                    << "mesh irregolare, triangolo " << e << ", eps_th " << s << ", forza " << i;
            }
        }
    }
}

// ESEMPIO 142. Il lavoro su un campo lineare. Un campo lineare u ha deformazione eps = (a1, b2, a2 + b1), costante, e
// le forze f_th fanno su di esso il lavoro u_e . f_th = A eps^T D eps_th (l'area A dalle coordinate dei nodi). Non usa
// B e non usa le formule di b_i e c_i. Per un moto rigido (eps = 0) il lavoro e' 0. I primi sei campi sono una base: se
// il lavoro e' giusto per tutti, le forze sono giuste in ognuno dei 6 numeri. Provato su sei triangoli, due materiali e
// cinque deformazioni termiche.
TEST(ElementThermalLoad, WorkOfLinearFieldsIsExact) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const std::vector<Vector3> strains = sample_thermal_strains();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangolo " << k << ", materiale " << m << ", eps_th " << s;
                for (std::size_t r = 0; r < fields.size(); ++r) {
                    const Vector6 u = nodal_values(mesh, 0, fields[r]);
                    const double want =
                        triangle_area(mesh, 0) * product(strain_of(fields[r]), stress_of(materials[m], strains[s]));
                    EXPECT_NEAR(work_of(u, f), want, 1e-13 * largest(f) * largest(u))
                        << "triangolo " << k << ", materiale " << m << ", eps_th " << s << ", campo " << r;
                }
            }
        }
    }
}

// ESEMPIO 143. Equilibrio. Una deformazione termica costante non spinge il triangolo e non lo fa ruotare: le tre forze
// sommano a zero in x e in y, e il loro momento, la somma di x_i f_y - y_i f_x, e' 0 (e' lo stesso di dire che un moto
// rigido non fa lavoro). Dipende solo dal fatto che sigma e' costante e simmetrica: sommando b_i e c_i sui tre nodi
// viene 0. Una forza tutta zeri sarebbe in equilibrio senza dimostrare niente: per questo prima si controlla che non
// sia nulla. Provato su sei triangoli, cinque materiali e cinque deformazioni termiche.
TEST(ElementThermalLoad, IsInEquilibrium) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<Vector3> strains = sample_thermal_strains();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Tri& t = triangles[k];
        const Mesh2D mesh = mesh_of(t);
        double size = 0.0;
        for (double v : {t.x0, t.y0, t.x1, t.y1, t.x2, t.y2}) {
            size = std::max(size, std::fabs(v));
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangolo " << k << ", materiale " << m << ", eps_th " << s;
                const double tol = 1e-13 * largest(f);
                EXPECT_NEAR(f[0] + f[2] + f[4], 0.0, tol)
                    << "somma in x, triangolo " << k << ", materiale " << m << ", eps_th " << s;
                EXPECT_NEAR(f[1] + f[3] + f[5], 0.0, tol)
                    << "somma in y, triangolo " << k << ", materiale " << m << ", eps_th " << s;
                const double moment = t.x0 * f[1] - t.y0 * f[0] + t.x1 * f[3] - t.y1 * f[2] + t.x2 * f[5] - t.y2 * f[4];
                EXPECT_NEAR(moment, 0.0, tol * size)
                    << "momento, triangolo " << k << ", materiale " << m << ", eps_th " << s;
            }
        }
    }
}

// ESEMPIO 144. Dilatazione libera. Se il triangolo si dilata liberamente, u = e x, v = e y (in generale il campo con
// deformazione eps = eps_th), la tensione D (B u - eps_th) e' 0 e non ci sono forze elastiche: Ke u_e = f_th, cioe' le
// f_th sono le forze che, su un triangolo freddo, darebbero lo stesso spostamento (come k x_th per la molla). Mette
// insieme Ke e f_th: se f_th ha il segno o il fattore sbagliato (o se il materiale D non e' usato nello stesso modo),
// la dilatazione libera non e' piu' in equilibrio. Provato su sei triangoli, tre materiali e cinque deformazioni
// termiche.
TEST(ElementThermalLoad, FreeExpansionBalancesTheStiffness) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Vector3> strains = sample_thermal_strains();
    const Matrix3 materials[3] = {general(), plane_strain_matrix(1.0, 0.25), plane_strain_matrix(210e9, 0.3)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 3; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangolo " << k << ", materiale " << m << ", eps_th " << s;
                const Vector6 u = nodal_values(mesh, 0, free_expansion(strains[s]));
                const Vector6 got = times(K, u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], f[i], 1e-13 * largest(K) * largest(u))
                        << "triangolo " << k << ", materiale " << m << ", eps_th " << s << ", forza " << i;
                }
            }
        }
    }
}

// ESEMPIO 145. Le forze termiche sono "tensione per lunghezza". Se il triangolo si ingrandisce di s (e magari si
// sposta), A cresce di s^2 e B cala di 1/s: f_th = A B^T D eps_th cresce di s. (In 2D e con spessore 1 la forza di una
// tensione sigma su un lato lungo L e' sigma L.) E' il contrario di Ke, che non dipende dalla dimensione. Provato con
// s = 1, 2, 0.25 e 10, due spostamenti, sei triangoli e due materiali.
TEST(ElementThermalLoad, ScalesWithTheSizeOfTheTriangle) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.3)};
    const Vector3 eps = {2.0e-3, -1.0e-3, 1.5e-3};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Vector6 base = element_thermal_load(mesh_of(triangles[k]), 0, materials[m], eps);
            ASSERT_GT(largest(base), 0.0) << "triangolo " << k << ", materiale " << m;
            for (double s : scales) {
                for (const auto& shift : shifts) {
                    const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                    const Vector6 f = element_thermal_load(mesh_of(moved), 0, materials[m], eps);
                    for (std::size_t i = 0; i < 6; ++i) {
                        EXPECT_NEAR(f[i], s * base[i], 1e-10 * s * largest(base))
                            << "triangolo " << k << ", materiale " << m << ", s = " << s << ", forza " << i;
                    }
                }
            }
        }
    }
}

// ESEMPIO 146. Un materiale isotropo non ha una direzione preferita. Se si ruota il triangolo di phi attorno
// all'origine e si ruota anche la deformazione termica (eps' = R eps R^T, con lo scorrimento gamma = 2 eps_xy), le
// forze ruotano allo stesso modo: f' = Q f, in ogni nodo (f_x, f_y) -> R (f_x, f_y). Serve la D vera di
// plane_strain_matrix: D[2][2] = mu lega il taglio e lo scorrimento solo con le convenzioni giuste, altrimenti la
// tensione D eps_th, e quindi le forze, dipenderebbe dalla direzione. Una dilatazione uguale in x e y e' invariante per
// rotazione; la seconda deformazione ha il taglio e cambia. Provato con 4 angoli, sei triangoli e due nu.
TEST(ElementThermalLoad, IsIndifferentToRotation) {
    const double angles[4] = {0.3, 1.5707963267948966, -1.0, 3.0};
    const Matrix3 materials[2] = {plane_strain_matrix(1.0, 0.3), plane_strain_matrix(1.0, 0.25)};
    const Vector3 strains[2] = {{2.0e-3, 2.0e-3, 0.0}, {2.0e-3, -1.0e-3, 1.5e-3}};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            for (std::size_t s = 0; s < 2; ++s) {
                const Vector6 base = element_thermal_load(mesh_of(triangles[k]), 0, materials[m], strains[s]);
                ASSERT_GT(largest(base), 0.0) << "triangolo " << k << ", materiale " << m << ", eps_th " << s;
                for (double phi : angles) {
                    const Vector6 got = element_thermal_load(mesh_of(rotated(triangles[k], phi)), 0, materials[m],
                                                             rotated_strain(strains[s], phi));
                    const Vector6 want = rotated_forces(base, phi);
                    for (std::size_t i = 0; i < 6; ++i) {
                        EXPECT_NEAR(got[i], want[i], 1e-12 * largest(base))
                            << "triangolo " << k << ", materiale " << m << ", eps_th " << s << ", phi = " << phi
                            << ", forza " << i;
                    }
                }
            }
        }
    }
}

// ESEMPIO 147. Una mesh intera. Si sommano le forze dei triangoli su ogni nodo (il grado di liberta' x del nodo n e'
// 2 n, quello y e' 2 n + 1) per un riscaldamento uniforme, con la stessa D e la stessa eps_th in tutti i triangoli. Per
// il teorema della divergenza la somma sul nodo n e' sigma g_n, con sigma = D eps_th e
//     g_n = somma, sui lati del bordo che toccano n, di (normale uscente) x (meta' della lunghezza del lato):
// i nodi interni non hanno nessun lato del bordo e non ricevono forze (le forze dei triangoli attorno si annullano);
// un nodo sul bordo destro, lontano dagli angoli, ha g = (hy, 0) (due lati lunghi hy, ognuno conta per meta'); un
// angolo ha un lato per parte e g = (+-hy/2, +-hx/2). Sulla mesh rettangolare 2 x 1.5 con 4 x 3 celle (24 triangoli,
// hx = 0.5 e hy = 0.5).
TEST(ElementThermalLoad, UniformHeatingLoadsOnlyTheBoundaryOfAMesh) {
    const double lx = 2.0;
    const double ly = 1.5;
    const std::size_t nx = 4;
    const std::size_t ny = 3;
    const Mesh2D mesh = rectangle_mesh(lx, ly, nx, ny);
    const Matrix3 D = general();
    const Vector3 eps = {2.0e-3, -1.0e-3, 1.5e-3};
    const Vector3 sigma = stress_of(D, eps);

    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Vector6 f = element_thermal_load(mesh, e, D, eps);
        const Triangle& t = mesh.triangle(e);
        for (std::size_t i = 0; i < 3; ++i) {
            F[2 * t[i]] += f[2 * i];
            F[2 * t[i] + 1] += f[2 * i + 1];
        }
    }

    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    const double edge = 1e-9 * lx;
    const double tol = 1e-13 * largest(sigma) * std::max(hx, hy);
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (std::size_t n = 0; n < mesh.num_nodes(); ++n) {
        const bool left = std::fabs(mesh.x(n)) < edge;
        const bool right = std::fabs(mesh.x(n) - lx) < edge;
        const bool bottom = std::fabs(mesh.y(n)) < edge;
        const bool top = std::fabs(mesh.y(n) - ly) < edge;
        const double share = ((left || right) && (bottom || top)) ? 0.5 : 1.0;  // un angolo ha un solo lato per parte
        const double gx = ((right ? 1.0 : 0.0) - (left ? 1.0 : 0.0)) * hy * share;
        const double gy = ((top ? 1.0 : 0.0) - (bottom ? 1.0 : 0.0)) * hx * share;
        EXPECT_NEAR(F[2 * n], sigma[0] * gx + sigma[2] * gy, tol)
            << "nodo " << n << " (" << mesh.x(n) << ", " << mesh.y(n) << "), forza in x";
        EXPECT_NEAR(F[2 * n + 1], sigma[2] * gx + sigma[1] * gy, tol)
            << "nodo " << n << " (" << mesh.x(n) << ", " << mesh.y(n) << "), forza in y";
        sum_x += F[2 * n];
        sum_y += F[2 * n + 1];
    }
    EXPECT_NEAR(sum_x, 0.0, tol * static_cast<double>(mesh.num_nodes()));
    EXPECT_NEAR(sum_y, 0.0, tol * static_cast<double>(mesh.num_nodes()));
}

// ESEMPIO 148. Un triangolo che non esiste: std::out_of_range, come per B e per Ke.
TEST(ElementThermalLoad, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    const Matrix3 D = hooke_quarter();
    const Vector3 eps = {1.0, 1.0, 0.0};
    EXPECT_NO_THROW(element_thermal_load(mesh, 0, D, eps));
    EXPECT_NO_THROW(element_thermal_load(mesh, mesh.num_triangles() - 1, D, eps));
    EXPECT_THROW(element_thermal_load(mesh, mesh.num_triangles(), D, eps), std::out_of_range);
    EXPECT_THROW(element_thermal_load(mesh, 1000, D, eps), std::out_of_range);
}