#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Triplets.h"

// ---- Dalle matrici dei triangoli alla matrice di tutta la mesh, con due spostamenti per nodo -----------------------
//
// element_elasticity_stiffness(mesh, e, D) da' la matrice 6 x 6 di UN triangolo, con i gradi di liberta' LOCALI
// (u0, v0, u1, v1, u2, v2). La matrice di tutta la mesh, K, e' 2 n x 2 n (n = numero dei nodi) e i suoi indici sono i
// gradi di liberta' GLOBALI: il nodo I ha lo spostamento in x al numero 2 I e quello in y al numero 2 I + 1.
// K[P][Q] e' la forza sul grado di liberta' P quando si sposta di 1 il solo grado di liberta' Q. Come per il calore, e'
// la somma dei contributi di tutti i triangoli che hanno sia P sia Q:
//
//     K[P][Q] = somma, sui triangoli e che contengono P e Q, di Ke[i][j]
//               dove i e j sono i gradi di liberta' locali di P e Q, cioe' con t = mesh.triangle(e):
//               P = 2 t[i / 2] + i % 2   e   Q = 2 t[j / 2] + j % 2
//
// (il grado di liberta' locale i e' la componente i % 2 del nodo locale i / 2). Ogni triangolo produce 36 terne
// (riga, colonna, valore) = (2 t[i / 2] + i % 2, 2 t[j / 2] + j % 2, Ke[i][j]) con i e j da 0 a 5; le terne di tutti i
// triangoli vanno a csr_from_triplets, che somma quelle con la stessa posizione. Due nodi che non stanno mai nello
// stesso triangolo non si influenzano: il blocco 2 x 2 fra loro e' 0 e non occupa memoria.
//
// elasticity_stiffness_triplets_2d(mesh, D) restituisce le terne e assemble_elasticity_stiffness_2d(mesh, D) la CSR
// (2 n x 2 n). Non ci sono errori da segnalare: la mesh e' gia' stata controllata dal costruttore di Mesh2D e D non
// viene controllata (come per element_elasticity_stiffness).
//
// Cosa deve avere una vera matrice di rigidezza, ed e' quello che i test controllano:
//   - e' simmetrica;
//   - un moto rigido (una traslazione, una rotazione) non deforma niente, quindi non produce nessuna forza: K u = 0;
//   - per uno spostamento lineare (deformazione costante eps) l'energia vale  u^T K u = eps^T D eps * (area totale),
//     e le forze di una tensione costante si annullano su ogni nodo interno (restano solo sul bordo);
//   - e' la somma dei contributi dei triangoli, anche per le mesh con numeri di nodo in disordine e con nodi che non
//     stanno in nessun triangolo (due righe vuote per ciascuno);
//   - occupa 4 posti per ogni posto della matrice del calore (un blocco 2 x 2 per ogni coppia di nodi nello stesso
//     triangolo) e resta sparsa anche per mesh grandi.

namespace {

using Dense = std::vector<std::vector<double>>;

// Un rettangolo lx x ly diviso in nx x ny celle, come in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Sei rettangoli di forme molto diverse (gli stessi dei test dell'assemblaggio del calore).
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

// Materiali a mano (3 x 3, indici xx, yy, xy), simmetrici: la matrice di rigidezza non ha bisogno che siano fisici.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 general() {  // tutti i numeri diversi, anche l'accoppiamento con il taglio
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}
Matrix3 steel() {  // un materiale vero, rigido: numeri di ordine 1e11
    return plane_strain_matrix(210e9, 0.3);
}

std::vector<Matrix3> sample_materials() { return {hooke_quarter(), general(), steel()}; }

// Il piu' grande valore assoluto sulla diagonale di A: la "scala" della matrice, per fissare le tolleranze.
double largest_diagonal(const CSR& A) {
    double m = 0.0;
    for (double d : A.diag()) {
        m = std::max(m, std::fabs(d));
    }
    return m;
}

// Il piu' grande valore assoluto di un vettore.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
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
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},      // traslazione in x
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},      // traslazione in y
            {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},     // rotazione
            {0.3, 0.0, -0.7, -1.2, 0.7, 0.0}};   // traslazione e rotazione insieme
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

// L'area totale: la somma delle aree di tutti i triangoli.
double total_area(const Mesh2D& mesh) {
    double a = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        a += mesh.area(e);
    }
    return a;
}

// eps^T D eps, con eps = (a1, b2, a2 + b1) la deformazione del campo f.
double strain_energy_density_times_two(const Matrix3& D, const LinearField& f) {
    const double eps[3] = {f.a1, f.b2, f.a2 + f.b1};
    double s = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            s += eps[i] * D[i][j] * eps[j];
        }
    }
    return s;
}

// Il "conto lento": K in una tabella piena 2 n x 2 n, in cui ogni triangolo somma a mano i suoi 36 numeri nelle caselle
// (2 t[a] + c, 2 t[b] + d), con a e b i nodi locali e c e d le componenti (il grado di liberta' locale e' 2 a + c).
// Serve solo come confronto, e solo per mesh piccole.
Dense slow_stiffness(const Mesh2D& mesh, const Matrix3& D) {
    const std::size_t n = 2 * mesh.num_nodes();
    Dense K(n, std::vector<double>(n, 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix6 Ke = element_elasticity_stiffness(mesh, e, D);
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t c = 0; c < 2; ++c) {
                for (std::size_t b = 0; b < 3; ++b) {
                    for (std::size_t d = 0; d < 2; ++d) {
                        K[2 * t[a] + c][2 * t[b] + d] += Ke[2 * a + c][2 * b + d];
                    }
                }
            }
        }
    }
    return K;
}

// Confronta A con una tabella piena, casella per casella (con una tolleranza relativa alla casella piu' grande).
void expect_same_as_dense(const CSR& A, const Dense& table) {
    ASSERT_EQ(A.size(), table.size());
    double scale = 0.0;
    for (const auto& row : table) {
        for (double v : row) {
            scale = std::max(scale, std::fabs(v));
        }
    }
    for (std::size_t i = 0; i < table.size(); ++i) {
        for (std::size_t j = 0; j < table.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), table[i][j], 1e-13 * scale) << "elemento (" << i << ", " << j << ")";
        }
    }
}

// Controlla che A sia la matrice `want`, elemento per elemento, e che occupi esattamente `places` posti.
void expect_matrix(const CSR& A, const Dense& want, std::size_t places, double tolerance) {
    ASSERT_EQ(A.size(), want.size()) << "dimensione";
    EXPECT_EQ(A.get_val_size(), places) << "posti occupati";
    for (std::size_t i = 0; i < want.size(); ++i) {
        for (std::size_t j = 0; j < want.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), want[i][j], tolerance) << "elemento (" << i << ", " << j << ")";
        }
    }
}

// I valori delle terne che stanno in posizione (row, col), in ordine crescente.
std::vector<double> values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col) {
    std::vector<double> out;
    for (const Triplet& t : triplets) {
        if (t.row == row && t.col == col) {
            out.push_back(t.val);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Confronta i valori delle terne in posizione (row, col) con `want` (in ordine crescente).
void expect_values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col,
                      const std::vector<double>& want) {
    const std::vector<double> got = values_at(triplets, row, col);
    ASSERT_EQ(got.size(), want.size()) << "posizione (" << row << ", " << col << ")";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-14) << "posizione (" << row << ", " << col << "), valore numero " << i;
    }
}

// Quante posizioni (riga, colonna) DIVERSE compaiono nella lista, in una matrice n x n.
std::size_t distinct_places(const std::vector<Triplet>& triplets, std::size_t n) {
    std::vector<std::size_t> places;
    for (const Triplet& t : triplets) {
        places.push_back(t.row * n + t.col);
    }
    std::sort(places.begin(), places.end());
    places.erase(std::unique(places.begin(), places.end()), places.end());
    return places.size();
}

}  // namespace

// ESEMPIO 149. A mano. Il quadrato unitario ha 4 nodi, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), e due
// triangoli, T0 = (0, 1, 2) e T1 = (1, 3, 2), tutti e due di area 1/2. Con E = 1 e nu = 1/4 (lambda = mu = 0.4) il
// materiale e' D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]] e i gradi di liberta' globali sono 8: il nodo I ha u al
// numero 2 I e v al 2 I + 1. Le matrici dei triangoli (A B^T D B, passo di Ke), in unita' di 0.1:
//     Ke0 = [[ 8,  4, -6, -2, -2, -2],        Ke1 = [[ 2,  0, -2, -2,  0,  2],
//            [ 4,  8, -2, -2, -2, -6],               [ 0,  6, -2, -6,  2,  0],
//            [-6, -2,  6,  0,  0,  2],               [-2, -2,  8,  4, -6, -2],
//            [-2, -2,  0,  2,  2,  0],               [-2, -6,  4,  8, -2, -2],
//            [-2, -2,  0,  2,  2,  0],               [ 0,  2, -6, -2,  6,  0],
//            [-2, -6,  2,  0,  0,  6]]               [ 2,  0, -2, -2,  0,  2]]
// T0 ha i nodi (0, 1, 2), quindi i suoi 6 gradi di liberta' locali sono i globali (0, 1, 2, 3, 4, 5). T1 ha i nodi
// (1, 3, 2): i suoi gradi di liberta' locali (0, 1, 2, 3, 4, 5) sono i globali (2, 3, 6, 7, 4, 5). Si somma ogni numero
// di Ke0 e di Ke1 nella casella (globale della riga, globale della colonna). Per esempio
//     K[2][2] = Ke0[2][2] + Ke1[0][0] = 0.6 + 0.2 = 0.8      (il nodo 1 sta in tutti e due i triangoli)
//     K[2][5] = Ke0[2][5] + Ke1[0][5] = 0.2 + 0.2 = 0.4
//     K[6][4] = Ke1[2][4] = -0.6                              (il nodo 3 sta solo in T1)
//     K[0][6] = 0                                             (i nodi 0 e 3 non stanno mai nello stesso triangolo)
// e tutta la matrice e' (in unita' di 0.1) quella scritta sotto. Occupa 56 posti: 14 blocchi 2 x 2 (i 4 nodi con se'
// stessi e i 5 lati, nei due versi), e fra i posti ce ne sono 8 che valgono 0 (come K[2][3]: il contributo c'e', e' 0).
TEST(ElasticityAssembly2d, StiffnessOfTwoRightTrianglesMatchesHandComputation) {
    const Dense want = {{0.8, 0.4, -0.6, -0.2, -0.2, -0.2, 0.0, 0.0},
                        {0.4, 0.8, -0.2, -0.2, -0.2, -0.6, 0.0, 0.0},
                        {-0.6, -0.2, 0.8, 0.0, 0.0, 0.4, -0.2, -0.2},
                        {-0.2, -0.2, 0.0, 0.8, 0.4, 0.0, -0.2, -0.6},
                        {-0.2, -0.2, 0.0, 0.4, 0.8, 0.0, -0.6, -0.2},
                        {-0.2, -0.6, 0.4, 0.0, 0.0, 0.8, -0.2, -0.2},
                        {0.0, 0.0, -0.2, -0.2, -0.6, -0.2, 0.8, 0.4},
                        {0.0, 0.0, -0.2, -0.6, -0.2, -0.2, 0.4, 0.8}};
    expect_matrix(assemble_elasticity_stiffness_2d(unit_square(), hooke_quarter()), want, 56, 1e-14);
}

// ESEMPIO 150. Simmetria e moti rigidi. K e' simmetrica (le forze di reazione sono uguali e contrarie: la forza su P
// per uno spostamento di Q e' quella su Q per uno spostamento di P) e, nelle mesh in cui ogni nodo sta in un
// triangolo, la sua diagonale e' positiva (spostare un solo grado di liberta' deforma qualche triangolo). Un moto
// rigido, una traslazione o una rotazione, non deforma nessun triangolo, quindi le forze sono tutte 0:
//     K u = 0   per   u = (1, 0, 1, 0, ...),   (0, 1, 0, 1, ...),   (-y_I, x_I, ...)   e le loro combinazioni
// (i numeri di u sono nell'ordine dei gradi di liberta' globali: u del nodo I al 2 I, v al 2 I + 1, quindi una
// traslazione in x ha 1 ai posti pari e 0 ai dispari, e una numerazione sbagliata dei gradi di liberta' si vede).
// Provato su nove mesh e su tre materiali.
TEST(ElasticityAssembly2d, StiffnessIsSymmetricAndRigidMotionsProduceNoForce) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const CSR K = assemble_elasticity_stiffness_2d(mesh, materials[m]);
            ASSERT_EQ(K.size(), 2 * mesh.num_nodes());
            const double scale = largest_diagonal(K);
            ASSERT_GT(scale, 0.0);

            EXPECT_TRUE(K.is_symmetric(1e-12 * scale));
            const std::vector<double> diagonal = K.diag();
            for (std::size_t i = 0; i < diagonal.size(); ++i) {
                EXPECT_GT(diagonal[i], 0.0) << "diagonale " << i;
            }

            const std::vector<LinearField> motions = rigid_motions();
            for (std::size_t r = 0; r < motions.size(); ++r) {
                const std::vector<double> u = nodal_displacements(mesh, motions[r]);
                const std::vector<double> force = K * u;
                ASSERT_EQ(force.size(), u.size());
                const double tolerance = 1e-12 * scale * largest_entry(u);
                for (std::size_t i = 0; i < force.size(); ++i) {
                    ASSERT_NEAR(force[i], 0.0, tolerance) << "moto rigido numero " << r << ", forza " << i;
                }
            }
        }
    }
}

// ESEMPIO 151. Energia e nodi interni. Se lo spostamento e' lineare, u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y, la
// deformazione e' la stessa in tutta la mesh, eps = (a1, b2, a2 + b1), e l'energia di deformazione e' (1/2) eps^T D eps
// per l'area totale, quindi
//     u^T K u = eps^T D eps * (area totale).
// Per u = x (eps = (1, 0, 0)) e' D[0][0] per l'area, per u = y (eps = (0, 0, 1)) e' D[2][2] per l'area, e cosi' via.
// In piu' la tensione e' costante, D eps, e le forze di una tensione costante su un nodo interno (circondato da
// triangoli) si annullano: (K u)[2 I] = (K u)[2 I + 1] = 0 per ogni nodo I che non sta sul bordo. Le forze restano
// solo sul bordo, dove la tensione "spinge" il contorno.
// Provato con le due traslazioni, i quattro campi puri e uno generale, su nove mesh e tre materiali.
TEST(ElasticityAssembly2d, StiffnessGivesTheEnergyOfLinearFieldsAndBalancesTheInteriorNodes) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<LinearField> fields = strain_fields();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        std::vector<bool> on_boundary(mesh.num_nodes(), false);
        for (std::size_t I : boundary_nodes(mesh)) {
            on_boundary[I] = true;
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const CSR K = assemble_elasticity_stiffness_2d(mesh, materials[m]);
            const double scale = largest_diagonal(K);
            for (std::size_t f = 0; f < fields.size(); ++f) {
                SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m) +
                             ", campo numero " + std::to_string(f));
                const std::vector<double> u = nodal_displacements(mesh, fields[f]);
                const std::vector<double> force = K * u;
                const double umax = largest_entry(u);

                double energy = 0.0;
                for (std::size_t i = 0; i < u.size(); ++i) {
                    energy += u[i] * force[i];
                }
                const double want = strain_energy_density_times_two(materials[m], fields[f]) * total_area(mesh);
                EXPECT_NEAR(energy, want, 1e-12 * scale * umax * umax);

                for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
                    if (!on_boundary[I]) {
                        EXPECT_NEAR(force[2 * I], 0.0, 1e-12 * scale * umax) << "nodo interno " << I << ", x";
                        EXPECT_NEAR(force[2 * I + 1], 0.0, 1e-12 * scale * umax) << "nodo interno " << I << ", y";
                    }
                }
            }
        }
    }
}

// ESEMPIO 152. Il "conto lento": la tabella piena 2 n x 2 n in cui ogni triangolo somma a mano i suoi 36 numeri nelle
// caselle (2 t[a] + c, 2 t[b] + d), con a e b i nodi locali e c e d le componenti (il grado di liberta' locale e' 2 a +
// c: e' lo stesso conto dei gradi di liberta' globali di prima, scritto con quattro cicli invece di due). La K
// assemblata e le terne sommate con csr_from_triplets devono dare le stesse caselle. Le terne sono 36 per triangolo e
// tutti i loro indici sono minori di 2 n (altrimenti csr_from_triplets lancia std::invalid_argument).
// Le mesh: sei rettangoli, il triangolo con un nodo dentro, il quadrilatero irregolare, due triangoli di area diversa
// e la mesh con due nodi fuori da ogni triangolo; i materiali sono tre.
TEST(ElasticityAssembly2d, StiffnessMatchesTheSlowDenseAssemblyOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh numero " + std::to_string(a) + ", materiale numero " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Dense table = slow_stiffness(mesh, materials[m]);

            expect_same_as_dense(assemble_elasticity_stiffness_2d(mesh, materials[m]), table);

            const std::vector<Triplet> triplets = elasticity_stiffness_triplets_2d(mesh, materials[m]);
            ASSERT_EQ(triplets.size(), 36 * mesh.num_triangles());
            expect_same_as_dense(csr_from_triplets(2 * mesh.num_nodes(), triplets), table);
        }
    }
}

// ESEMPIO 153. I posti occupati. Due nodi che stanno nello stesso triangolo (anche un nodo con se' stesso) hanno un
// blocco 2 x 2 di posti in K: quattro posti, anche se qualcuno vale 0. Nella matrice del calore (assemble_stiffness_2d)
// la stessa coppia ha un posto solo, quindi K occupa esattamente 4 volte i posti della matrice del calore, su ogni
// mesh. Per un rettangolo di nx x ny celle i nodi sono (nx + 1) (ny + 1), i lati sono
// nx (ny + 1) + (nx + 1) ny + nx ny (orizzontali, verticali, diagonali) e i posti della matrice del calore sono
// nodi + 2 lati, quindi K ne occupa
//     4 (nodi + 2 lati).
// Per il quadrato unitario dell'ESEMPIO 149 sono 4 (4 + 2 * 5) = 56.
TEST(ElasticityAssembly2d, StiffnessOccupiesFourPlacesForEveryPlaceOfTheHeatMatrix) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        SCOPED_TRACE("mesh numero " + std::to_string(a));
        const CSR K = assemble_elasticity_stiffness_2d(meshes[a], hooke_quarter());
        const CSR heat = assemble_stiffness_2d(meshes[a], 1.0);
        EXPECT_EQ(K.get_val_size(), 4 * heat.get_val_size());
    }

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE("rettangolo " + std::to_string(g.lx) + " x " + std::to_string(g.ly) + " con " +
                     std::to_string(g.nx) + " x " + std::to_string(g.ny) + " celle");
        const std::size_t nodes = (g.nx + 1) * (g.ny + 1);
        const std::size_t edges = g.nx * (g.ny + 1) + (g.nx + 1) * g.ny + g.nx * g.ny;
        const CSR K = assemble_elasticity_stiffness_2d(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), steel());
        EXPECT_EQ(K.size(), 2 * nodes);
        EXPECT_EQ(K.get_val_size(), 4 * (nodes + 2 * edges));
    }
}

// ESEMPIO 154. Nodi fuori da ogni triangolo. La mesh ha 6 nodi, ma solo (1, 2, 3) e (1, 3, 4) stanno in un triangolo:
// i nodi 0 e 5 non hanno niente da assemblare. K ha comunque 12 righe, due per nodo, e le righe e le colonne 0, 1 (nodo
// 0) e 10, 11 (nodo 5) sono vuote: valgono 0 e non occupano memoria. I posti sono quelli dei 4 nodi collegati: 4 nodi e
// 5 lati (il quadrato diviso lungo una diagonale), cioe' 4 (4 + 2 * 5) = 56.
TEST(ElasticityAssembly2d, NodesOutsideEveryTriangleGetEmptyRowsAndColumns) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const CSR K = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());
    ASSERT_EQ(K.size(), 12u);
    EXPECT_EQ(K.get_val_size(), 56u);

    const std::size_t empty[4] = {0, 1, 10, 11};
    for (std::size_t p : empty) {
        for (std::size_t q = 0; q < 12; ++q) {
            EXPECT_EQ(K.get(p, q), 0.0) << "riga " << p << ", colonna " << q;
            EXPECT_EQ(K.get(q, p), 0.0) << "riga " << q << ", colonna " << p;
        }
    }
    for (std::size_t p = 2; p < 10; ++p) {
        EXPECT_GT(K.get(p, p), 0.0) << "diagonale " << p;
    }
}

// ESEMPIO 155. Le terne prima di sommarle (le stesse dell'ESEMPIO 149). I due triangoli ne danno 36 ciascuno, 72 in
// tutto, ma le posizioni diverse sono solo 56 (i 56 posti di K): le altre 16 terne cadono su posti gia' presi
// dall'altro triangolo. Con le matrici Ke0 e Ke1 dell'ESEMPIO 149 (in unita' di 0.1 sono quelle scritte li'):
//     (2, 2): T0 da' 0.6 e T1 da' 0.2                0.6 + 0.2 = 0.8 = K[2][2]
//     (2, 5): T0 da' 0.2 e T1 da' 0.2                0.4 = K[2][5], e lo stesso in (5, 2)
//     (2, 3): T0 e T1 danno 0 tutti e due            il posto c'e' lo stesso e le terne vanno conservate
//     (6, 4): solo T1, che da' -0.6                  (il nodo 3 sta solo in T1)
//     (0, 0): solo T0, che da' 0.8
//     (0, 6): nessuna                                (i nodi 0 e 3 non stanno mai nello stesso triangolo)
// Sommate con csr_from_triplets, le 72 terne danno la K dell'ESEMPIO 149.
TEST(ElasticityAssembly2d, StiffnessTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D mesh = unit_square();
    const std::vector<Triplet> triplets = elasticity_stiffness_triplets_2d(mesh, hooke_quarter());

    ASSERT_EQ(triplets.size(), 72u);
    EXPECT_EQ(distinct_places(triplets, 8), 56u);
    expect_values_at(triplets, 2, 2, {0.2, 0.6});
    expect_values_at(triplets, 2, 5, {0.2, 0.2});
    expect_values_at(triplets, 5, 2, {0.2, 0.2});
    expect_values_at(triplets, 2, 3, {0.0, 0.0});
    expect_values_at(triplets, 6, 4, {-0.6});
    expect_values_at(triplets, 4, 6, {-0.6});
    expect_values_at(triplets, 0, 0, {0.8});
    expect_values_at(triplets, 0, 6, {});

    const CSR K = csr_from_triplets(mesh.num_nodes() * 2, triplets);
    const CSR want = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());
    ASSERT_EQ(K.size(), want.size());
    EXPECT_EQ(K.get_val_size(), want.get_val_size());
    for (std::size_t i = 0; i < K.size(); ++i) {
        for (std::size_t j = 0; j < K.size(); ++j) {
            EXPECT_DOUBLE_EQ(K.get(i, j), want.get(i, j)) << "elemento (" << i << ", " << j << ")";
        }
    }
}

// ESEMPIO 156. Una mesh grande resta sparsa. Il quadrato unitario con 100 x 100 celle ha 10201 nodi e 20402 gradi di
// liberta': la matrice piena avrebbe 20402 x 20402 numeri (oltre 3 GB), ma K occupa 4 (nodi + 2 lati) posti, cioe'
// 282404 invece di 416 milioni. Come per il calore, una traslazione non da' forze e l'energia di u = x e'
// u^T K u = D[0][0] * (area) = 1.2 (con il materiale dell'ESEMPIO 149).
TEST(ElasticityAssembly2d, AssemblyOfALargeMeshStaysSparse) {
    const std::size_t n = 100;
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, n, n);
    const CSR K = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());

    const std::size_t nodes = (n + 1) * (n + 1);
    const std::size_t edges = n * (n + 1) + (n + 1) * n + n * n;
    ASSERT_EQ(K.size(), 2 * nodes);
    EXPECT_EQ(K.get_val_size(), 4 * (nodes + 2 * edges));

    const std::vector<double> shift = nodal_displacements(mesh, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const std::vector<double> force = K * shift;
    for (std::size_t i = 0; i < force.size(); ++i) {
        ASSERT_NEAR(force[i], 0.0, 1e-12) << "riga " << i;
    }

    const std::vector<double> stretch = nodal_displacements(mesh, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    const std::vector<double> Ku = K * stretch;
    double energy = 0.0;
    for (std::size_t i = 0; i < stretch.size(); ++i) {
        energy += stretch[i] * Ku[i];
    }
    EXPECT_NEAR(energy, 1.2, 1e-9);
}