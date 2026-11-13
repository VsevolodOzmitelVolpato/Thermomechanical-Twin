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

// ---- Dagli spostamenti alle tensioni: sigma = D (B u - eps_th) -----------------------------------------------------
//
// Con gli spostamenti u della lastra (solve_displacement_2d) si calcola la tensione in ogni triangolo. Dentro un
// triangolo lo spostamento e' lineare, quindi la deformazione B u_e e' costante, e lo e' anche la tensione: tre numeri
// (sigma_xx, sigma_yy, sigma_xy) per triangolo. Come in 1D (sigma = E (u' - alpha dT)) la tensione nasce solo dalla
// parte di deformazione che NON e' dovuta al calore:
//     sigma = D (B u_e - eps_th),        eps_th = (1 + nu) alpha dT (1, 1, 0).
// Le funzioni sono due.
//   - element_stress_2d(mesh, e, D, u_e, eps_th) e' la tensione di UN triangolo: gli servono i suoi sei spostamenti
//     (u0, v0, u1, v1, u2, v2) e la sua deformazione termica.
//   - stress_field_2d(mesh, D, nu, alpha, u, T, T_ref) e' quella di TUTTA la mesh: per ogni triangolo prende dai 2 n
//     numeri di u i suoi sei spostamenti (il locale i e' il globale 2 t[i / 2] + i % 2), calcola dT come la media delle
//     temperature dei suoi tre nodi meno T_ref (come assemble_thermal_load_2d) e chiama la funzione del triangolo.
// Cosa devono fare, ed e' quello che i test controllano:
//   - la tensione di un triangolo si calcola a mano (ESEMPIO 181), e per ogni campo lineare, ogni forma di triangolo e
//     ogni D, anche non simmetrica (ESEMPIO 182); un triangolo che non esiste lancia std::out_of_range (ESEMPIO 183);
//   - sul quadrato unitario le tensioni dei problemi dei passi prima si calcolano a mano (ESEMPIO 184);
//   - un campo lineare ha la stessa tensione in tutti i triangoli, su ogni mesh (ESEMPIO 185); la lastra scaldata e
//     incastrata ha tensione -D eps_th, quella libera nessuna (ESEMPIO 186);
//   - ogni triangolo usa la media delle temperature dei suoi nodi (ESEMPIO 187) e i suoi nodi (ESEMPIO 188);
//   - le forze sui nodi delle tensioni sono K u - F (ESEMPIO 189), e in un corpo libero scaldato sono tutte 0
//     (ESEMPIO 190): le tensioni sono in equilibrio;
//   - il risultato e' lineare in spostamenti, temperatura e materiale (ESEMPIO 191) e non legge i nodi fuori da ogni
//     triangolo (ESEMPIO 192); i dati sbagliati lanciano std::invalid_argument (ESEMPIO 193);
//   - insieme a solve_displacement_2d: la trazione di un rettangolo (ESEMPIO 194), la lastra scaldata fra due pareti,
//     con la tensione di libro sigma_xx = -E alpha dT / (1 - nu) (ESEMPIO 195), e la stessa lastra in acciaio con 40 x
//     40 celle (ESEMPIO 196).

namespace {

// Un rettangolo lx x ly diviso in nx x ny celle, come in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Sei rettangoli di forme molto diverse (gli stessi dei test degli spostamenti).
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},    // il quadrato unitario: due triangoli
            {2.0, 1.0, 2, 1},    // due celle
            {3.0, 1.5, 4, 3},    // celle 0.75 x 0.5
            {1.0, 1.0, 6, 6},    // 72 triangoli
            {0.2, 0.05, 5, 2},   // piccolo (20 cm per 5 cm, in metri)
            {10.0, 0.1, 1, 7}};  // molto schiacciato: celle 10 x 0.0143
}

// Il quadrato unitario diviso lungo la diagonale da (1, 0) a (0, 1): nodi 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 =
// (1, 1) e triangoli (0, 1, 2) e (1, 3, 2), tutti e due con area 1/2.
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
// Una D che NON e' simmetrica (nessun materiale vero lo e', ma la funzione usa D come la riceve): servono le righe, non
// le colonne.
Matrix3 nonsymmetric() {
    return {{{2.0, 0.5, 0.1}, {0.7, 3.0, -0.2}, {0.3, -0.4, 1.0}}};
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

// Il piu' grande valore assoluto di un vettore, di tre numeri e di una tabella 3 x 3.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
}
double largest_entry(const Vector3& v) {
    return std::max(std::fabs(v[0]), std::max(std::fabs(v[1]), std::fabs(v[2])));
}
double largest_entry(const Matrix3& D) {
    return std::max(largest_entry(D[0]), std::max(largest_entry(D[1]), largest_entry(D[2])));
}
// Il piu' grande valore assoluto di tutte le tensioni.
double largest_entry(const std::vector<Vector3>& sigma) {
    double m = 0.0;
    for (const Vector3& s : sigma) {
        m = std::max(m, largest_entry(s));
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

// La dimensione del triangolo piu' piccolo: la radice dell'area.
double smallest_size(const Mesh2D& mesh) {
    double s = std::sqrt(mesh.area(0));
    for (std::size_t e = 1; e < mesh.num_triangles(); ++e) {
        s = std::min(s, std::sqrt(mesh.area(e)));
    }
    return s;
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

// Un campo di spostamento lineare:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// In ogni punto ha la stessa deformazione (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// La deformazione (eps_xx, eps_yy, gamma_xy) del campo f, la stessa in ogni punto.
Vector3 strain_of(const LinearField& f) {
    return {f.a1, f.b2, f.a2 + f.b1};
}

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

// I sei spostamenti (u0, v0, u1, v1, u2, v2) del campo f nei tre nodi del triangolo e.
Vector6 local_values(const Mesh2D& mesh, std::size_t e, const LinearField& f) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u_e{};
    for (std::size_t i = 0; i < 3; ++i) {
        const double x = mesh.x(t[i]);
        const double y = mesh.y(t[i]);
        u_e[2 * i] = f.a0 + f.a1 * x + f.a2 * y;
        u_e[2 * i + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u_e;
}

// I sei spostamenti del triangolo e presi dal vettore u di tutta la mesh: il grado di liberta' locale i, da 0 a 5, e'
// quello globale 2 t[i / 2] + i % 2.
Vector6 gather(const Mesh2D& mesh, std::size_t e, const std::vector<double>& u) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u_e{};
    for (std::size_t i = 0; i < 6; ++i) {
        u_e[i] = u[2 * t[i / 2] + i % 2];
    }
    return u_e;
}

// La media delle temperature dei tre nodi del triangolo e, meno T_ref.
double mean_excess(const Mesh2D& mesh, std::size_t e, const std::vector<double>& T, double T_ref) {
    const Triangle& t = mesh.triangle(e);
    return (T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref;
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

// Tutti i campi lineari: i moti rigidi e quelli con deformazione.
std::vector<LinearField> all_fields() {
    std::vector<LinearField> fields = rigid_motions();
    for (const LinearField& f : strain_fields()) {
        fields.push_back(f);
    }
    return fields;
}

// Numeri che cambiano da un grado di liberta' all'altro senza regole semplici, di ampiezza amplitude.
std::vector<double> wavy_vector(std::size_t size, double amplitude) {
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double t = static_cast<double>(i);
        v[i] = amplitude * (std::sin(1.3 * t + 0.4) + 0.3 * std::cos(2.9 * t));
    }
    return v;
}

// Gli stessi numeri spostati di shift posizioni (l'ultimo passa al posto del primo): un vettore diverso, della stessa
// ampiezza.
std::vector<double> shifted(const std::vector<double>& v, std::size_t shift) {
    std::vector<double> w(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        w[i] = v[(i + shift) % v.size()];
    }
    return w;
}

// D v, riga per colonna: (D v)[i] e' la somma su k di D[i][k] v[k].
Vector3 times(const Matrix3& D, const Vector3& v) {
    Vector3 r{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            r[i] += D[i][k] * v[k];
        }
    }
    return r;
}

Vector3 minus(const Vector3& a, const Vector3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Una tolleranza per le tensioni: 1e-9 volte il piu' grande numero di D per la somma delle deformazioni in gioco:
// quella meccanica (strain), quella termica (thermal) e lo spostamento diviso per la dimensione dei triangoli (B u_e
// sottrae numeri vicini e perde delle cifre in proporzione).
double stress_tolerance(const Matrix3& D, double strain, double thermal, double displacement, double size) {
    return 1e-9 * largest_entry(D) * (strain + thermal + displacement / size);
}

// Confronta due tensioni (tre numeri) con una tolleranza assoluta.
void expect_stress_near(const Vector3& got, const Vector3& want, double tolerance) {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "componente " << i;
    }
}

// Confronta due vettori elemento per elemento, con una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "grado di liberta' " << i;
    }
}

// Le forze sui nodi delle tensioni: per ogni triangolo e i sei numeri A B^T sigma_e (come le forze termiche A B^T D
// eps_th), sommati nei gradi di liberta' globali 2 t[i / 2] + i % 2.
std::vector<double> nodal_forces(const Mesh2D& mesh, const std::vector<Vector3>& sigma) {
    std::vector<double> f(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Matrix3x6 B = strain_displacement_matrix(mesh, e);
        const Triangle& t = mesh.triangle(e);
        for (std::size_t i = 0; i < 6; ++i) {
            for (std::size_t r = 0; r < 3; ++r) {
                f[2 * t[i / 2] + i % 2] += mesh.area(e) * B[r][i] * sigma[e][r];
            }
        }
    }
    return f;
}

// Un problema con la soluzione esatta nota: le forze, i gradi di liberta' fissati con i loro spostamenti imposti e la
// soluzione esatta stessa (2 n numeri).
struct Problem {
    std::vector<double> F;
    std::vector<std::size_t> dofs;
    std::vector<double> values;
    std::vector<double> exact;
};

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

// Una lastra scaldata in modo uniforme di dT (T = T_ref + dT in tutti i nodi) e trattenuta fra due pareti: i nodi del
// lato sinistro (x = 0) e di quello destro (x = lx) non si spostano in x (u = 0), e il nodo in basso a sinistra non si
// sposta in y (v = 0, cosi' la lastra non scivola in y); sopra e sotto la lastra e' libera. In deformazione piana
// (epsilon_zz = 0, quindi sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT) con epsilon_xx = 0 e sigma_yy = 0 le
// equazioni di Hooke con il calore danno
//     sigma_xx = -E alpha dT / (1 - nu),        epsilon_yy = (1 + nu) alpha dT / (1 - nu),
// e la soluzione esatta e' u = 0, v = epsilon_yy y: un campo lineare, che gli elementi riproducono. La lastra non puo'
// allungarsi in x, e il calore che non puo' dilatarla diventa compressione.
Problem walls_problem(const Mesh2D& mesh, double lx, double E, double nu, double alpha, double dT) {
    Problem pr;
    pr.F = assemble_thermal_load_2d(mesh, plane_strain_matrix(E, nu), nu, alpha,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        if (mesh.x(I) <= 1e-9 * lx || mesh.x(I) >= (1.0 - 1e-9) * lx) {
            pr.dofs.push_back(2 * I);
        }
    }
    pr.dofs.push_back(1);  // v del nodo 0, che sta in (0, 0)
    pr.values.assign(pr.dofs.size(), 0.0);

    const double eyy = (1.0 + nu) * alpha * dT / (1.0 - nu);
    pr.exact = nodal_displacements(mesh, LinearField{0.0, 0.0, 0.0, 0.0, 0.0, eyy});
    return pr;
}

}  // namespace

// ESEMPIO 181. A mano, un triangolo. Il triangolo 0 del quadrato unitario ha i nodi (0, 0), (1, 0), (0, 1) e area
// 1/2; con E = 1 e nu = 1/4 e' D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. I sei spostamenti sono
// (u0, v0, u1, v1, u2, v2) e, con 2 A = 1, la deformazione B u_e e' eps_xx = u1 - u0, eps_yy = v2 - v0, gamma_xy =
// (u2 - u0) + (v1 - v0).
//   (a) Trazione. u_e = (0, 0, 15/16, 0, 0, -5/16) ha eps = (15/16, -5/16, 0) e sigma = D eps =
//       (1.2 15/16 - 0.4 5/16, 0.4 15/16 - 1.2 5/16, 0) = (1, 0, 0): e' il campo dell'ESEMPIO 168 (a), tensione 1 in x.
//   (b) Dilatazione libera. Con eps_th = (2.5, 2.5, 0) il triangolo si sposta di u_e = (0, 0, 2.5, 0, 0, 2.5), cioe' di
//       B u_e = (2.5, 2.5, 0) = eps_th: tutta la deformazione e' dovuta al calore e la tensione e' 0.
//   (c) Trattenuto. u_e = 0 con lo stesso eps_th: sigma = D (0 - eps_th) = -(1.2 2.5 + 0.4 2.5, 0.4 2.5 + 1.2 2.5, 0) =
//       (-4, -4, 0): compressione.
//   (d) Taglio. u_e = (0, 0, 0, 0, 1, 0) (u = y) ha gamma_xy = 1 e sigma = (0, 0, 0.4 1) = (0, 0, 0.4).
//   (e) Un po' e un po'. u_e = (0, 0, 0.5, 0, 0, 0.5) (eps = (0.5, 0.5, 0)) con eps_th = (2.5, 2.5, 0): conta la
//       DIFFERENZA, eps - eps_th = (-2, -2, 0), e sigma = (1.2 (-2) + 0.4 (-2), 0.4 (-2) + 1.2 (-2), 0) =
//       (-3.2, -3.2, 0).
//   (f) Il triangolo 1. Ha i nodi (1, 0), (1, 1), (0, 1), con b = (0, 1, -1) e c = (-1, 1, 0) (2 A = 1). Se si sposta
//       in x di 0.2 il suo solo nodo 1 (u_e = (0, 0, 0.2, 0, 0, 0)), eps = (b_1 0.2, 0, c_1 0.2) = (0.2, 0, 0.2) e
//       sigma = (1.2 0.2, 0.4 0.2, 0.4 0.2) = (0.24, 0.08, 0.08).
TEST(ElasticityStress2d, ElementStressOfASingleTriangleByHand) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const Vector3 no_heat = {0.0, 0.0, 0.0};
    const Vector3 heat = {2.5, 2.5, 0.0};

    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.9375, 0.0, 0.0, -0.3125}, no_heat), {1.0, 0.0, 0.0},
                       1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 2.5, 0.0, 0.0, 2.5}, heat), {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, heat), {-4.0, -4.0, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, no_heat), {0.0, 0.0, 0.4}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.5, 0.0, 0.0, 0.5}, heat), {-3.2, -3.2, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 1, D, {0.0, 0.0, 0.2, 0.0, 0.0, 0.0}, no_heat), {0.24, 0.08, 0.08},
                       1e-12);
}

// ESEMPIO 182. La tensione di un triangolo qualunque, con un campo lineare qualunque. Un campo u = a0 + a1 x + a2 y,
// v = b0 + b1 x + b2 y ha in ogni punto la deformazione (a1, b2, a2 + b1), e si calcola senza B: la tensione e' D
// ((a1, b2, a2 + b1) - eps_th), con D usata riga per colonna (sigma[i] e' la somma su k di D[i][k] (...)[k]). Si
// prova su ogni triangolo di nove mesh (sei rettangoli, il triangolo con un nodo dentro, il quadrilatero irregolare,
// due triangoli di area diversa), con quattro D (la matrice di Hooke con E = 1, una qualunque, l'acciaio e una NON
// simmetrica, che distingue le righe dalle colonne), tre deformazioni termiche (anche con il taglio, che la
// dilatazione termica vera non ha ma la funzione deve sottrarre) e undici campi: i moti rigidi, che hanno tensione 0
// qualunque sia D, le deformazioni "pure" e un campo con deformazioni di 1e-3 e una traslazione di 0.3.
TEST(ElasticityStress2d, ElementStressOfAnyLinearFieldOnAnyTriangle) {
    const std::vector<Matrix3> materials = {hooke_quarter(), general(), steel(), nonsymmetric()};
    const std::vector<Vector3> heats = {{0.0, 0.0, 0.0}, {2.5e-3, 2.5e-3, 0.0}, {3.0e-3, -1.0e-3, 2.0e-3}};
    const std::vector<LinearField> fields = all_fields();

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Matrix3& D = materials[m];
            SCOPED_TRACE("mesh " + std::to_string(k) + ", material " + std::to_string(m));
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                for (const LinearField& f : fields) {
                    const Vector6 u_e = local_values(mesh, e, f);
                    const Vector3 strain = strain_of(f);
                    double u_max = 0.0;
                    for (double a : u_e) {
                        u_max = std::max(u_max, std::fabs(a));
                    }
                    for (const Vector3& eps_th : heats) {
                        SCOPED_TRACE("triangolo " + std::to_string(e));
                        const Vector3 want = times(D, minus(strain, eps_th));
                        const double tolerance = stress_tolerance(D, largest_entry(strain), largest_entry(eps_th),
                                                                  u_max, size);
                        expect_stress_near(element_stress_2d(mesh, e, D, u_e, eps_th), want, tolerance);
                    }
                }
            }
            if (HasFailure()) {
                return;
            }
        }
    }
}

// ESEMPIO 183. Un triangolo che non esiste lancia std::out_of_range (come strain_displacement_matrix, da cui passa):
// il quadrato unitario ha i triangoli 0 e 1, la mesh di 3 x 3 celle i triangoli da 0 a 17.
TEST(ElasticityStress2d, AMissingTriangleThrowsOutOfRange) {
    const Matrix3 D = hooke_quarter();
    const Vector6 u_e = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const Vector3 eps_th = {0.5, 0.5, 0.0};

    const Mesh2D square = unit_square();
    EXPECT_NO_THROW(element_stress_2d(square, 0, D, u_e, eps_th));
    EXPECT_NO_THROW(element_stress_2d(square, 1, D, u_e, eps_th));
    EXPECT_THROW(element_stress_2d(square, 2, D, u_e, eps_th), std::out_of_range);
    EXPECT_THROW(element_stress_2d(square, 1000000000, D, u_e, eps_th), std::out_of_range);
    EXPECT_THROW(element_stress_2d(square, std::numeric_limits<std::size_t>::max(), D, u_e, eps_th),
                 std::out_of_range);

    const Mesh2D grid = rectangle_mesh(1.0, 1.0, 3, 3);
    ASSERT_EQ(grid.num_triangles(), 18u);
    EXPECT_NO_THROW(element_stress_2d(grid, 17, D, u_e, eps_th));
    EXPECT_THROW(element_stress_2d(grid, 18, D, u_e, eps_th), std::out_of_range);
}

// ESEMPIO 184. A mano, tutta la mesh. Il quadrato unitario ha 4 nodi, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), e
// due triangoli, (0, 1, 2) e (1, 3, 2). Il materiale ha E = 1 e nu = 1/4 (come nell'ESEMPIO 168) e alpha = 1, con
// T_ref = 5. Gli spostamenti sono (u0, v0, u1, v1, u2, v2, u3, v3) e le temperature (T0, T1, T2, T3). Ogni problema
// ha due tensioni, una per triangolo.
//   (a) La trazione dell'ESEMPIO 168 (a): u = (0, 0, 15/16, 0, 0, -5/16, 15/16, -5/16), T = 5 dappertutto (dT = 0).
//       Tutti e due i triangoli hanno tensione (1, 0, 0).
//   (b) Il corpo libero scaldato (b): u = (0, 0, 2.5, 0, 0, 2.5, 2.5, 2.5), T = 7 (dT = 2, eps_th = 1.25 2 (1, 1, 0) =
//       2.5 (1, 1, 0)): B u = eps_th e tutte e due le tensioni sono 0.
//   (c) Lo spostamento imposto (c): u = (0, 0, 0.1, 0, 0, -1/30, 0.1, -1/30), T = 5: eps = (0.1, -1/30, 0) e sigma =
//       (1.2 0.1 - 0.4 / 30, 0.4 0.1 - 1.2 / 30, 0) = (8/75, 0, 0), cioe' (0.10666..., 0, 0) in tutti e due.
//   (d) Scaldato e incastrato: u = 0, T = 7: sigma = -D eps_th = (-4, -4, 0) in tutti e due.
//   (e) Si sposta un nodo solo: u = 0 tranne u3 = 0.2 (il numero 6), T = 5. Il triangolo 0, che ha i nodi 0, 1, 2,
//       non sente il nodo 3: tensione (0, 0, 0). Il triangolo 1, che ha i nodi 1, 3, 2, ha lo spostamento
//       (0, 0, 0.2, 0, 0, 0) dell'ESEMPIO 181 (f): tensione (0.24, 0.08, 0.08).
//   (f) La temperatura cambia da un nodo all'altro: u = 0, T = (5, 5, 5, 11). Il triangolo 0 ha i nodi 0, 1, 2 e
//       dT = (5 + 5 + 5) / 3 - 5 = 0: tensione 0. Il triangolo 1 ha i nodi 1, 3, 2 e dT = (5 + 11 + 5) / 3 - 5 = 2:
//       tensione -D eps_th = (-4, -4, 0).
TEST(ElasticityStress2d, StressesOfTheUnitSquareByHand) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const double nu = 0.25;
    const double alpha = 1.0;
    const double T_ref = 5.0;
    const std::vector<double> rest(8, 0.0);

    const auto solve = [&](const std::vector<double>& u, const std::vector<double>& T) {
        return stress_field_2d(mesh, D, nu, alpha, u, T, T_ref);
    };

    std::vector<Vector3> sigma =
        solve({0.0, 0.0, 15.0 / 16.0, 0.0, 0.0, -5.0 / 16.0, 15.0 / 16.0, -5.0 / 16.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {1.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {1.0, 0.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 2.5, 0.0, 0.0, 2.5, 2.5, 2.5}, {7.0, 7.0, 7.0, 7.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {0.0, 0.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 0.1, 0.0, 0.0, -1.0 / 30.0, 0.1, -1.0 / 30.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {8.0 / 75.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {8.0 / 75.0, 0.0, 0.0}, 1e-12);

    sigma = solve(rest, {7.0, 7.0, 7.0, 7.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {-4.0, -4.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {-4.0, -4.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {0.24, 0.08, 0.08}, 1e-12);

    sigma = solve(rest, {5.0, 5.0, 5.0, 11.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {-4.0, -4.0, 0.0}, 1e-12);
}

// ESEMPIO 185. Il test della toppa per le tensioni. Con un campo di spostamento LINEARE u = a0 + a1 x + a2 y, v =
// b0 + b1 x + b2 y la deformazione e' la stessa in ogni triangolo, (a1, b2, a2 + b1), per qualunque forma dei
// triangoli; con una temperatura uniforme T = T_ref + dT lo e' anche eps_th = c (1, 1, 0), con c = (1 + nu) alpha dT.
// La tensione D ((a1, b2, a2 + b1) - eps_th) e' allora la stessa in tutti i triangoli. Si prova su nove mesh, tre
// materiali (anche non fisici, e l'acciaio), tre temperature (dT = 0, +100 e -60 kelvin) e undici campi: i moti
// rigidi (che hanno tensione -D eps_th, qualunque sia lo spostamento), le deformazioni "pure" e un campo con
// deformazioni di 1e-3 e una traslazione di 0.3.
TEST(ElasticityStress2d, ALinearDisplacementFieldGivesTheSameStressInEveryTriangle) {
    const std::vector<LinearField> fields = all_fields();
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (const Material& m : sample_materials()) {
            for (double dT : {0.0, 100.0, -60.0}) {
                const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
                const double c = (1.0 + m.nu) * m.alpha * dT;
                for (const LinearField& f : fields) {
                    SCOPED_TRACE("mesh " + std::to_string(k) + ", dT = " + std::to_string(dT));
                    const std::vector<double> u = nodal_displacements(mesh, f);
                    const Vector3 strain = strain_of(f);
                    const Vector3 want = times(m.D, minus(strain, {c, c, 0.0}));
                    const double tolerance =
                        stress_tolerance(m.D, largest_entry(strain), std::fabs(c), largest_entry(u), size);

                    const std::vector<Vector3> sigma =
                        stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
                    ASSERT_EQ(sigma.size(), mesh.num_triangles());
                    for (std::size_t e = 0; e < sigma.size(); ++e) {
                        expect_stress_near(sigma[e], want, tolerance);
                    }
                }
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 186. I due estremi della lastra scaldata in modo uniforme (T = T_ref + dT in tutti i nodi, quindi
// eps_th = c (1, 1, 0) con c = (1 + nu) alpha dT).
//   - INCASTRATA (u = 0 in tutti i nodi): il calore vorrebbe dilatarla e non puo'. La tensione e' -D eps_th in ogni
//     triangolo, una compressione se si scalda: e' la tensione che nel passo prima nasceva insieme allo spostamento 0
//     (ESEMPIO 173). Tutto il calore diventa tensione.
//   - LIBERA (u = (a0 + c x, b0 + c y), con una traslazione (a0, b0) = (0.3, -0.2) che non cambia niente): il corpo si
//     dilata di c in ogni direzione, B u = eps_th e la tensione e' 0. Tutto il calore diventa spostamento.
// Nove mesh, tre materiali e due riscaldamenti (+100 e -60 kelvin).
TEST(ElasticityStress2d, AHeatedPlateIsStressedWhenClampedAndStressFreeWhenFree) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                SCOPED_TRACE("mesh " + std::to_string(k) + ", dT = " + std::to_string(dT));
                const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
                const double c = (1.0 + m.nu) * m.alpha * dT;

                const std::vector<Vector3> clamped = stress_field_2d(
                    mesh, m.D, m.nu, m.alpha, std::vector<double>(2 * mesh.num_nodes(), 0.0), T, reference_temperature);
                ASSERT_EQ(clamped.size(), mesh.num_triangles());
                const Vector3 compression = times(m.D, {-c, -c, 0.0});
                for (const Vector3& s : clamped) {
                    expect_stress_near(s, compression, stress_tolerance(m.D, 0.0, std::fabs(c), 0.0, size));
                }

                const std::vector<double> u = nodal_displacements(mesh, LinearField{0.3, c, 0.0, -0.2, 0.0, c});
                const std::vector<Vector3> free_body =
                    stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
                ASSERT_EQ(free_body.size(), mesh.num_triangles());
                for (const Vector3& s : free_body) {
                    expect_stress_near(s, {0.0, 0.0, 0.0},
                                       stress_tolerance(m.D, std::fabs(c), std::fabs(c), largest_entry(u), size));
                }
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 187. La temperatura di un triangolo e' la MEDIA delle temperature dei suoi tre nodi, come per le forze
// termiche (assemble_thermal_load_2d): dT_e = (T[t[0]] + T[t[1]] + T[t[2]]) / 3 - T_ref, e eps_th e' (1 + nu) alpha
// dT_e (1, 1, 0). Con u = 0 la tensione e' -D eps_th, diversa da un triangolo all'altro
// (L = la dimensione maggiore della mesh).
//   - Con una temperatura LINEARE, T = T_ref + g0 + gx x + gy y (g0 = 20 K, gx = 80 K / L, gy = -50 K / L), la media
//     dei nodi e' il valore nel baricentro: dT_e = g0 + gx cx + gy cy, con (cx, cy) il baricentro del triangolo.
//   - Con una temperatura che NON e' lineare, T = T_ref + 30 (x / L)^2 - 20 (y / L)^2 + 10 (x / L) (y / L), la media
//     dei nodi non e' il valore nel baricentro, e la funzione deve usare la media: dT_e = (f0 + f1 + f2) / 3, con f_i
//     il valore di 30 (x / L)^2 - ... nel nodo i.
// Nove mesh, tre materiali.
TEST(ElasticityStress2d, TheTemperatureOfATriangleIsTheMeanOfItsThreeNodes) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double L = mesh_size(mesh);
        const double g0 = 20.0;
        const double gx = 80.0 / L;
        const double gy = -50.0 / L;
        const auto bump = [&](std::size_t I) {
            const double X = mesh.x(I) / L;
            const double Y = mesh.y(I) / L;
            return 30.0 * X * X - 20.0 * Y * Y + 10.0 * X * Y;
        };
        std::vector<double> linear(mesh.num_nodes());
        std::vector<double> curved(mesh.num_nodes());
        for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
            linear[I] = reference_temperature + g0 + gx * mesh.x(I) + gy * mesh.y(I);
            curved[I] = reference_temperature + bump(I);
        }
        const std::vector<double> rest(2 * mesh.num_nodes(), 0.0);

        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<Vector3> from_linear =
                stress_field_2d(mesh, m.D, m.nu, m.alpha, rest, linear, reference_temperature);
            const std::vector<Vector3> from_curved =
                stress_field_2d(mesh, m.D, m.nu, m.alpha, rest, curved, reference_temperature);
            ASSERT_EQ(from_linear.size(), mesh.num_triangles());
            ASSERT_EQ(from_curved.size(), mesh.num_triangles());

            // Gli scarti di temperatura sono al piu' 150 kelvin (20 + 80 + 50 nel caso lineare, 60 in quello curvo).
            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 150.0;
            const double tolerance = stress_tolerance(m.D, 0.0, thermal, 0.0, 1.0);
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangolo " + std::to_string(e));
                const Triangle& t = mesh.triangle(e);
                const double cx = (mesh.x(t[0]) + mesh.x(t[1]) + mesh.x(t[2])) / 3.0;
                const double cy = (mesh.y(t[0]) + mesh.y(t[1]) + mesh.y(t[2])) / 3.0;

                const double dT_linear = g0 + gx * cx + gy * cy;
                const double c_linear = (1.0 + m.nu) * m.alpha * dT_linear;
                expect_stress_near(from_linear[e], times(m.D, {-c_linear, -c_linear, 0.0}), tolerance);

                const double dT_curved = (bump(t[0]) + bump(t[1]) + bump(t[2])) / 3.0;
                const double c_curved = (1.0 + m.nu) * m.alpha * dT_curved;
                expect_stress_near(from_curved[e], times(m.D, {-c_curved, -c_curved, 0.0}), tolerance);
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 188. Ogni triangolo usa i SUOI nodi. I campi dei test prima sono lineari o uniformi, e danno la stessa
// tensione in tutti i triangoli: uno scambio di nodi non si vede. Qui spostamenti e temperature cambiano da un nodo
// all'altro senza regole semplici (wavy_vector), quindi ogni triangolo ha una tensione sua, e quella del triangolo e
// deve essere quella di element_stress_2d con i suoi spostamenti u[2 t[i / 2] + i % 2], la media delle temperature
// dei suoi nodi meno T_ref e la sua deformazione termica plane_strain_thermal_strain(nu, alpha, dT_e). Nove mesh,
// quattro materiali (anche con la D non simmetrica). Ci si assicura anche che le tensioni non siano tutte uguali.
TEST(ElasticityStress2d, EveryTriangleUsesItsOwnNodes) {
    std::vector<Material> materials = sample_materials();
    materials.push_back({nonsymmetric(), 0.2, 5.0e-4});

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        const std::vector<double> u = wavy_vector(2 * mesh.num_nodes(), 0.01 * mesh_size(mesh));
        std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
        for (double& a : T) {
            a += reference_temperature;
        }
        for (const Material& m : materials) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<Vector3> sigma = stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
            ASSERT_EQ(sigma.size(), mesh.num_triangles());

            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 80.0;
            const double tolerance = stress_tolerance(m.D, 0.02, thermal, largest_entry(u), size);
            double spread = 0.0;
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangolo " + std::to_string(e));
                const Vector3 eps_th =
                    plane_strain_thermal_strain(m.nu, m.alpha, mean_excess(mesh, e, T, reference_temperature));
                expect_stress_near(sigma[e], element_stress_2d(mesh, e, m.D, gather(mesh, e, u), eps_th), tolerance);
                for (std::size_t i = 0; i < 3; ++i) {
                    spread = std::max(spread, std::fabs(sigma[e][i] - sigma[0][i]));
                }
            }
            EXPECT_GT(spread, 1e-3 * largest_entry(sigma)) << "le tensioni dei triangoli sono tutte uguali";
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 189. Le forze sui nodi delle tensioni sono K u - F. Per ogni triangolo A B^T sigma_e sono sei forze sui suoi
// nodi (come le forze termiche A B^T D eps_th), e sommate come in K danno, grado di liberta' per grado di liberta',
//     somma sui triangoli di A B^T D (B u_e - eps_th) = K u - F,
// con K = assemble_elasticity_stiffness_2d e F = assemble_thermal_load_2d: e' l'equazione di equilibrio, scritta con
// le tensioni. Vale per QUALUNQUE u e T (non solo per la soluzione): il sinistro e' calcolato con la tensione della
// funzione, il destro con K ed F. Si prova con spostamenti e temperature qualunque (wavy_vector), su nove mesh e
// quattro materiali (anche con la D non simmetrica: K ed F la usano come viene data).
TEST(ElasticityStress2d, TheNodalForcesOfTheStressesAreKuMinusF) {
    std::vector<Material> materials = sample_materials();
    materials.push_back({nonsymmetric(), 0.2, 5.0e-4});

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const std::vector<double> u = wavy_vector(2 * mesh.num_nodes(), 0.01 * mesh_size(mesh));
        std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
        for (double& a : T) {
            a += reference_temperature;
        }
        for (const Material& m : materials) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<double> F =
                assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, reference_temperature);
            const std::vector<double> Ku = assemble_elasticity_stiffness_2d(mesh, m.D) * u;
            const std::vector<double> f =
                nodal_forces(mesh, stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature));
            ASSERT_EQ(f.size(), F.size());
            ASSERT_EQ(Ku.size(), F.size());

            const double scale = largest_entry(Ku) + largest_entry(F);
            for (std::size_t i = 0; i < f.size(); ++i) {
                EXPECT_NEAR(f[i], Ku[i] - F[i], 1e-10 * scale) << "grado di liberta' " << i;
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 190. In un corpo libero scaldato le tensioni sono in equilibrio da sole. Con una temperatura qualunque
// (wavy_vector, di ampiezza 40 kelvin) e i soli tre vincoli isostatici (nessuna forza esterna, quindi nessuna
// reazione) si calcola u con solve_displacement_2d: il corpo si deforma in modo irregolare e ha tensioni diverse da 0
// (almeno un decimo della tensione termica D eps_th che avrebbe incastrato: senza questo controllo un vettore di
// tensioni tutte 0 avrebbe forze 0 e passerebbe), e le forze sui nodi A B^T sigma sommate sono 0 in TUTTI i gradi di
// liberta' (K u = F nei liberi, reazione 0 nei tre fissati). Mesh di 4 x 3 celle, tre materiali.
TEST(ElasticityStress2d, TheStressesOfAFreeHeatedBodyAreInEquilibrium) {
    const Mesh2D mesh = rectangle_mesh(3.0, 1.5, 4, 3);
    std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
    for (double& a : T) {
        a += reference_temperature;
    }
    const std::vector<std::size_t> dofs = isostatic_dofs(mesh, 0);

    for (const Material& m : sample_materials()) {
        const std::vector<double> F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, reference_temperature);
        const std::vector<double> u = solve_displacement_2d(mesh, m.D, F, dofs, {0.0, 0.0, 0.0});
        const std::vector<Vector3> sigma = stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
        ASSERT_EQ(sigma.size(), mesh.num_triangles());

        const double thermal_stress = largest_entry(m.D) * (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 40.0;
        EXPECT_GT(largest_entry(sigma), 0.1 * thermal_stress) << "le tensioni sono troppo piccole per provare qualcosa";

        const std::vector<double> f = nodal_forces(mesh, sigma);
        const double scale = largest_entry(F);
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_NEAR(f[i], 0.0, 1e-8 * scale) << "grado di liberta' " << i;
        }
    }
}

// ESEMPIO 191. Le tensioni sono LINEARI in spostamenti, temperatura e materiale. Con lo stesso materiale e due coppie
// (u1, T1) e (u2, T2), con dT = T - T_ref:
//   - la somma: sigma(u1 + u2, T_ref + dT1 + dT2) = sigma(u1, T1) + sigma(u2, T2);
//   - un fattore s = -2.5 (anche negativo) su tutti e due: sigma(s u1, T_ref + s dT1) = s sigma(u1, T1);
//   - il materiale: con 3 D al posto di D (e gli stessi nu e alpha) tutte le tensioni sono 3 volte quelle di D.
// E due cose sul calore: con alpha = 0 la temperatura non conta (le tensioni sono quelle della sola B u_e, con T
// qualunque o con T = T_ref), e con T = T_ref non conta alpha. Nove mesh, tre materiali.
TEST(ElasticityStress2d, TheStressIsLinearInDisplacementTemperatureAndMaterial) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const std::size_t N = 2 * mesh.num_nodes();
        const double size = smallest_size(mesh);
        const std::vector<double> u1 = wavy_vector(N, 0.01 * mesh_size(mesh));
        const std::vector<double> u2 = shifted(u1, 7);
        const std::vector<double> dT1 = wavy_vector(mesh.num_nodes(), 40.0);
        const std::vector<double> dT2 = shifted(dT1, 3);

        const auto temperature = [&](double s1, double s2) {
            std::vector<double> T(mesh.num_nodes());
            for (std::size_t I = 0; I < T.size(); ++I) {
                T[I] = reference_temperature + s1 * dT1[I] + s2 * dT2[I];
            }
            return T;
        };
        const auto combine = [&](const std::vector<double>& a, double sa, const std::vector<double>& b, double sb) {
            std::vector<double> w(a.size());
            for (std::size_t i = 0; i < w.size(); ++i) {
                w[i] = sa * a[i] + sb * b[i];
            }
            return w;
        };

        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 250.0;
            const double tolerance = stress_tolerance(m.D, 0.05, thermal, 2.5 * largest_entry(u1), size);
            const auto stress = [&](const Matrix3& D, double alpha, const std::vector<double>& u,
                                    const std::vector<double>& T) {
                return stress_field_2d(mesh, D, m.nu, alpha, u, T, reference_temperature);
            };

            const std::vector<Vector3> s1 = stress(m.D, m.alpha, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> s2 = stress(m.D, m.alpha, u2, temperature(0.0, 1.0));
            const std::vector<Vector3> sum = stress(m.D, m.alpha, combine(u1, 1.0, u2, 1.0), temperature(1.0, 1.0));
            const std::vector<Vector3> scaled =
                stress(m.D, m.alpha, combine(u1, -2.5, u2, 0.0), temperature(-2.5, 0.0));
            Matrix3 triple = m.D;
            for (auto& row : triple) {
                for (double& a : row) {
                    a *= 3.0;
                }
            }
            const std::vector<Vector3> tripled = stress(triple, m.alpha, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> no_alpha_hot = stress(m.D, 0.0, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> no_alpha_cold = stress(m.D, 0.0, u1, temperature(0.0, 0.0));
            const std::vector<Vector3> cold = stress(m.D, m.alpha, u1, temperature(0.0, 0.0));
            ASSERT_EQ(s1.size(), mesh.num_triangles());

            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangolo " + std::to_string(e));
                Vector3 want_sum{};
                Vector3 want_scaled{};
                Vector3 want_tripled{};
                for (std::size_t i = 0; i < 3; ++i) {
                    want_sum[i] = s1[e][i] + s2[e][i];
                    want_scaled[i] = -2.5 * s1[e][i];
                    want_tripled[i] = 3.0 * s1[e][i];
                }
                expect_stress_near(sum[e], want_sum, tolerance);
                expect_stress_near(scaled[e], want_scaled, tolerance);
                expect_stress_near(tripled[e], want_tripled, 3.0 * tolerance);
                expect_stress_near(no_alpha_hot[e], no_alpha_cold[e], tolerance);
                expect_stress_near(cold[e], no_alpha_cold[e], tolerance);
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// ESEMPIO 192. I nodi che non stanno in nessun triangolo (la mesh ha i nodi 0 = (5, 5) e 5 = (7, 7) fuori dai due
// triangoli del quadrato 1, 2, 3, 4) non vengono letti: ne' il loro spostamento (i numeri 0, 1, 10 e 11 di u) ne' la
// loro temperatura. Con dei NaN al loro posto le tensioni dei due triangoli sono IDENTICHE a quelle calcolate con dei
// numeri qualunque, e senza NaN.
TEST(ElasticityStress2d, NodesOutsideEveryTriangleAreNotRead) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const Matrix3 D = hooke_quarter();
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> u = wavy_vector(12, 0.1);
    std::vector<double> T = wavy_vector(6, 30.0);
    for (double& a : T) {
        a += reference_temperature;
    }
    std::vector<double> u_nan = u;
    std::vector<double> T_nan = T;
    for (std::size_t d : {0u, 1u, 10u, 11u}) {
        u[d] = 123.0;  // numeri qualunque
        u_nan[d] = not_a_number;
    }
    T[0] = -5000.0;
    T[5] = 1.0e9;
    T_nan[0] = not_a_number;
    T_nan[5] = not_a_number;

    const std::vector<Vector3> clean = stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, reference_temperature);
    const std::vector<Vector3> with_nan = stress_field_2d(mesh, D, 0.25, 1.0e-3, u_nan, T_nan, reference_temperature);
    ASSERT_EQ(clean.size(), 2u);
    ASSERT_EQ(with_nan.size(), 2u);
    for (std::size_t e = 0; e < 2; ++e) {
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_FALSE(std::isnan(with_nan[e][i])) << "triangolo " << e << ", componente " << i;
            EXPECT_EQ(with_nan[e][i], clean[e][i]) << "triangolo " << e << ", componente " << i;
        }
    }
}

// ESEMPIO 193. Dati sbagliati: la funzione lancia std::invalid_argument (e non legge fuori da un vettore). Con la
// mesh di 2 x 1 celle (6 nodi, 12 spostamenti) sono sbagliati:
//   - un u che non ha 12 numeri (11, 13, 6, 24, 0);
//   - una T che non ha 6 numeri (5, 7, 12, 0);
//   - un coefficiente di Poisson che non sta fra -1 e 1/2, estremi esclusi (-1, 0.5, -2, 1, +-infinito, NaN).
// Vanno bene invece nu = -0.99 e 0.49 (quasi gli estremi), alpha = 0 e negativo, e qualunque T_ref (anche sotto lo zero
// assoluto): D, alpha, u, T e T_ref non vengono controllati.
TEST(ElasticityStress2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 2, 1);
    ASSERT_EQ(mesh.num_nodes(), 6u);
    const Matrix3 D = hooke_quarter();
    const std::vector<double> u(12, 0.01);
    const std::vector<double> T(6, 300.0);
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, 293.15));

    for (std::size_t size : {11u, 13u, 6u, 24u, 0u}) {
        EXPECT_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, std::vector<double>(size, 0.01), T, 293.15),
                     std::invalid_argument)
            << "u di " << size << " numeri";
    }
    for (std::size_t size : {5u, 7u, 12u, 0u}) {
        EXPECT_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, std::vector<double>(size, 300.0), 293.15),
                     std::invalid_argument)
            << "T di " << size << " numeri";
    }
    for (double nu : {-1.0, 0.5, -2.0, 1.0, -infinity, infinity, not_a_number}) {
        EXPECT_THROW(stress_field_2d(mesh, D, nu, 1.0e-3, u, T, 293.15), std::invalid_argument) << "nu = " << nu;
    }

    EXPECT_NO_THROW(stress_field_2d(mesh, D, -0.99, 1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.49, 1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 0.0, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, -1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, -300.0));
}

// ESEMPIO 194. Insieme a solve_displacement_2d: la trazione di un rettangolo. Come nell'ESEMPIO 169 (lato sinistro con
// u = 0, un nodo con v = 0, sul lato destro la tensione sigma = 1e-3 E) si calcolano gli spostamenti, e da questi le
// tensioni: in ogni triangolo sigma_xx = sigma, sigma_yy = 0 e sigma_xy = 0, e T = T_ref non scalda niente. Si prova su
// sei rettangoli di forma diversa (anche molto schiacciati), con E = 1 e con E = 210e9 e con nu = 0, 1/4, 0.3, 0.4: gli
// errori degli spostamenti (1e-12 e meno) si moltiplicano per 1 / h nella deformazione, e le tensioni coincidono con
// sigma entro 1e-6 sigma.
TEST(ElasticityStress2d, TensionOfARectangleThroughTheSolver) {
    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (double E : {1.0, 210e9}) {
            for (double nu : {0.0, 0.25, 0.3, 0.4}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(E) + ", nu = " + std::to_string(nu));
                const double sigma = 1e-3 * E;
                const Problem pr = stretched_rectangle(mesh, g.lx, E, nu, sigma);
                const Matrix3 D = plane_strain_matrix(E, nu);
                const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);

                const std::vector<Vector3> stress = stress_field_2d(
                    mesh, D, nu, 1.0e-5, u, std::vector<double>(mesh.num_nodes(), reference_temperature),
                    reference_temperature);
                ASSERT_EQ(stress.size(), mesh.num_triangles());
                for (const Vector3& s : stress) {
                    expect_stress_near(s, {sigma, 0.0, 0.0}, 1e-6 * sigma);
                }
            }
        }
    }
}

// ESEMPIO 195. La lastra scaldata fra due pareti, con la formula di libro (walls_problem). Si scalda di dT in modo
// uniforme una lastra che non puo' allungarsi in x (u = 0 sui lati sinistro e destro) e che in y e' libera. Il calore
// che non puo' dilatarla in x diventa compressione:
//     sigma_xx = -E alpha dT / (1 - nu),        sigma_yy = 0,        sigma_xy = 0,
// e in y la lastra si allunga di epsilon_yy = (1 + nu) alpha dT / (1 - nu) (spostamenti u = 0, v = epsilon_yy y). Con
// E = 1 e nu = 1/4 e dT = 2 e alpha = 1 sono sigma_xx = -(4/3) 2 = -8/3 e epsilon_yy = 1.25 2 / 0.75 = 10/3. Si prova
// su sei rettangoli, con quattro materiali (nu = 0 dove sigma_xx = -E alpha dT, l'acciaio e l'alluminio) e due
// riscaldamenti (+50 e -30 kelvin, quindi trazione se si raffredda): gli spostamenti coincidono con quelli di libro
// entro 1e-8 volte il piu' grande (il CG si ferma con un residuo di 1e-12, e sulle celle molto schiacciate, 10 x 0.014,
// l'errore sugli spostamenti arriva a 1e-10), le tensioni entro 1e-6 volte sigma_xx.
TEST(ElasticityStress2d, AHeatedPlateBetweenTwoWallsMatchesTheTextbook) {
    struct Plate {
        double E, nu, alpha;
    };
    const std::vector<Plate> plates = {{1.0, 0.25, 1.0e-3},     // E = 1, nu = 1/4
                                       {1.0, 0.0, 2.0e-3},      // nu = 0: sigma_xx = -E alpha dT
                                       {210e9, 0.3, 12.0e-6},   // acciaio
                                       {70e9, 0.33, 23.0e-6}};  // alluminio

    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (const Plate& p : plates) {
            for (double dT : {50.0, -30.0}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(p.E) + ", nu = " + std::to_string(p.nu) + ", dT = " + std::to_string(dT));
                const Problem pr = walls_problem(mesh, g.lx, p.E, p.nu, p.alpha, dT);
                const Matrix3 D = plane_strain_matrix(p.E, p.nu);
                const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);
                expect_vectors_near(u, pr.exact, 1e-8 * largest_entry(pr.exact));

                const double sigma_xx = -p.E * p.alpha * dT / (1.0 - p.nu);
                const std::vector<Vector3> stress =
                    stress_field_2d(mesh, D, p.nu, p.alpha, u,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
                ASSERT_EQ(stress.size(), mesh.num_triangles());
                for (const Vector3& s : stress) {
                    expect_stress_near(s, {sigma_xx, 0.0, 0.0}, 1e-6 * std::fabs(sigma_xx));
                }
            }
        }
    }
}

// ESEMPIO 196. La stessa lastra, in grande: un quadrato di acciaio di 1 m di lato con 40 x 40 celle
// (1681 nodi, 3362 gradi di liberta'), E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, scaldato di 100 kelvin e
// trattenuto fra due pareti. La tensione di libro e' sigma_xx = -E alpha dT / (1 - nu) = -360 MPa (-3.6e8 Pa) in
// tutti i 3200 triangoli, con sigma_yy = sigma_xy = 0: la funzione deve ritrovarla entro 1e-6 volte il suo valore (il
// CG fa circa 540 iterazioni, con numeri di K di ordine 1e11). Il tempo e' quello di solve_displacement_2d: una
// frazione di secondo.
TEST(ElasticityStress2d, ALargeSteelPlateBetweenTwoWalls) {
    const double E = 210e9;
    const double nu = 0.3;
    const double alpha = 12.0e-6;
    const double dT = 100.0;
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 40, 40);
    ASSERT_EQ(mesh.num_triangles(), 3200u);

    const Problem pr = walls_problem(mesh, 1.0, E, nu, alpha, dT);
    const Matrix3 D = plane_strain_matrix(E, nu);
    const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);

    const double sigma_xx = -E * alpha * dT / (1.0 - nu);
    EXPECT_NEAR(sigma_xx, -3.6e8, 1.0);
    const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
    const std::vector<Vector3> stress = stress_field_2d(mesh, D, nu, alpha, u, T, reference_temperature);
    ASSERT_EQ(stress.size(), 3200u);
    for (std::size_t e = 0; e < stress.size(); ++e) {
        expect_stress_near(stress[e], {sigma_xx, 0.0, 0.0}, 1e-6 * std::fabs(sigma_xx));
        if (HasFailure()) {
            return;
        }
    }
}