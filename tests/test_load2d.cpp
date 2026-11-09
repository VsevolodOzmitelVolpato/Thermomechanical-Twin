#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Fem2D.h"
#include "Mesh2D.h"

// ---- Il carico di una sorgente che cambia da punto a punto ------------------------------------------------
//
// Finora la sorgente di calore era un numero q, lo stesso in ogni punto della lastra. Ora e' una funzione q(x, y):
// la potenza per unita' di volume nel punto (x, y). Il carico del triangolo e e' ancora
//
//     f[i] = integrale sul triangolo di  q(x, y) N_i  dA        (i = 0, 1, 2, nell'ordine di mesh.triangle(e))
//
// ma ora l'integrale non e' piu' q A / 3: dipende da come q cambia dentro il triangolo, e in generale non si sa
// calcolare a mano. Si usa una formula: si calcola q in pochi punti e si sommano i valori con dei pesi. Qui la
// REGOLA DEI PUNTI MEDI DEI LATI: per una funzione g qualunque,
//
//     integrale di g dA  =  A / 3 * ( g(M01) + g(M12) + g(M20) )      (circa)
//
// dove M01 e' il punto medio del lato fra i nodi locali 0 e 1, M12 quello fra 1 e 2, M20 quello fra 2 e 0. La
// formula e' ESATTA per ogni polinomio di grado 0, 1 e 2. Nel triangolo (0, 0), (1, 0), (0, 1), per esempio, con
// g = x^2 l'integrale esatto e' 1/12 e la formula da' 1/6 * (1/4 + 1/4 + 0) = 1/12.
//
// Per avere f[i] si prende g = q N_i. Lungo un lato che parte dal nodo i, N_i scende da 1 a 0 in modo lineare,
// quindi nel punto medio vale 1/2; lungo il lato opposto a i vale 0 dappertutto. Percio' (A / 3 per 1/2 = A / 6)
//
//     f[0] = A / 6 * (q(M01) + q(M20))
//     f[1] = A / 6 * (q(M01) + q(M12))
//     f[2] = A / 6 * (q(M12) + q(M20))
//
// cioe' ogni nodo riceve A / 6 per la somma di q nei punti medi dei DUE lati che lo toccano.
//   - Se q e' costante vale q A / 3 per ogni nodo: e' la funzione di prima (element_load con q numero).
//   - Se q e' lineare, q N_i ha grado 2 e i tre numeri sono gli integrali esatti.
//   - Se q ha grado 2, q N_i ha grado 3 e i tre numeri non sono piu' esatti uno per uno. Ma N_0 + N_1 + N_2 = 1,
//     quindi la loro SOMMA e' l'integrale di q, che ha grado 2: quella e' ancora esatta.
//
// assemble_load_2d(mesh, q) somma, nei nodi, i carichi dei triangoli, come la funzione di prima.

namespace {

using Source = std::function<double(double, double)>;

// Confronta due vettori di tre numeri, elemento per elemento.
void expect_vector3_near(const Vector3& got, const Vector3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << " elemento " << i;
    }
}

// Confronta due vettori elemento per elemento. La tolleranza e' relativa: rel_tol * (1 + |valore atteso|).
void expect_vector_near(const std::vector<double>& got, const std::vector<double>& want, double rel_tol,
                        const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], rel_tol * (1.0 + std::fabs(want[i]))) << what << ": elemento " << i;
    }
}

// La somma di tutti gli elementi.
double total_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double value : v) {
        s += value;
    }
    return s;
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

// Il triangolo (0, 0), (1, 0), (0, 1), di area 1/2.
Mesh2D right_triangle() {
    return Mesh2D({0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {{0, 1, 2}});
}

// Il triangolo grande con un nodo dentro: (0, 0), (3, 0), (0, 3) e il nodo 3 = (1, 1). Non e' un rettangolo.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// Il quadrilatero irregolare con un nodo dentro: quattro triangoli di area diversa.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Una mesh in cui i nodi 0 = (5, 5) e 5 = (7, 7) non appartengono a nessun triangolo.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// Mesh di tutte le forme: un rettangolo, il triangolo con un nodo dentro, il quadrilatero irregolare, due mesh
// storte, il quadrato di due triangoli e la mesh con due nodi fuori da ogni triangolo.
std::vector<Mesh2D> varied_meshes() {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(perturbed_rectangle(2.0, 1.0, 8, 6, 5));
    meshes.push_back(perturbed_rectangle(1.0, 1.0, 12, 12, 31));
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    return meshes;
}

}  // namespace

// ESEMPIO 80. Il triangolo (0, 0), (1, 0), (0, 1): A = 1/2, quindi A / 6 = 1/12. I punti medi dei lati sono
// M01 = (1/2, 0), M12 = (1/2, 1/2) e M20 = (0, 1/2). Con la sorgente lineare q = x:
//     q(M01) = 1/2,  q(M12) = 1/2,  q(M20) = 0
//     f[0] = 1/12 * (1/2 + 0) = 1/24      f[1] = 1/12 * (1/2 + 1/2) = 1/12      f[2] = 1/12 * (1/2 + 0) = 1/24
// Sono gli integrali esatti: l'integrale di x N_1 = x^2 e' 1/12, quello di x N_0 e quello di x N_2 = x y sono 1/24,
// e la somma 1/6 e' l'integrale di x, cioe' A per l'ascissa del baricentro (1/2 * 1/3).
// Con q = y si scambiano i nodi 1 e 2: (1/24, 1/24, 1/12). Con q = 3 + x - y si sommano le due cose a 3 volte il
// carico costante (1/6 per nodo): f[0] = 1/2 + 1/24 - 1/24 = 1/2, f[1] = 1/2 + 1/12 - 1/24 = 13/24 e
// f[2] = 1/2 + 1/24 - 1/12 = 11/24.
TEST(Load2d, LinearSourceOnTheRightTriangleIsIntegratedExactly) {
    const Mesh2D m = right_triangle();
    const Source qx = [](double x, double) { return x; };
    const Source qy = [](double, double y) { return y; };
    const Source mixed = [](double x, double y) { return 3.0 + x - y; };

    expect_vector3_near(element_load(m, 0, qx), {{1.0 / 24.0, 1.0 / 12.0, 1.0 / 24.0}}, 1e-15, "q = x:");
    expect_vector3_near(element_load(m, 0, qy), {{1.0 / 24.0, 1.0 / 24.0, 1.0 / 12.0}}, 1e-15, "q = y:");
    expect_vector3_near(element_load(m, 0, mixed), {{1.0 / 2.0, 13.0 / 24.0, 11.0 / 24.0}}, 1e-15, "q = 3 + x - y:");
}

// ESEMPIO 81. Lo stesso triangolo, con sorgenti di grado 2: la regola dei punti medi non e' esatta per ogni nodo,
// ma lo e' per la somma.
//     q = x^2:  q(M01) = 1/4, q(M12) = 1/4, q(M20) = 0    ->  f = (1/48, 1/24, 1/48)
//     q = x y:  q(M01) = 0,   q(M12) = 1/4, q(M20) = 0    ->  f = (0, 1/48, 1/48)
// Gli integrali esatti di x^2 N_i sarebbero (1/60, 1/20, 1/60): sono diversi, ed e' normale. Le somme invece sono
// l'integrale di q: 1/12 per x^2 e 1/24 per x y. Nel secondo caso il nodo 0, che e' l'origine, riceve 0: x y fa 0
// nei punti medi dei suoi due lati.
// Questo test fissa la REGOLA: una formula piu' precisa darebbe (1/60, 1/20, 1/60) e verrebbe bocciata. E' voluto:
// il passo dopo misura errori che dipendono da questa regola, e con la regola scritta qui chiunque puo' rifare
// i numeri.
TEST(Load2d, QuadraticSourceFollowsTheMidpointRuleAndKeepsTheTotal) {
    const Mesh2D m = right_triangle();
    const Source qxx = [](double x, double) { return x * x; };
    const Source qxy = [](double x, double y) { return x * y; };

    const Vector3 a = element_load(m, 0, qxx);
    expect_vector3_near(a, {{1.0 / 48.0, 1.0 / 24.0, 1.0 / 48.0}}, 1e-15, "q = x^2:");
    EXPECT_NEAR(a[0] + a[1] + a[2], 1.0 / 12.0, 1e-15) << "q = x^2: somma";

    const Vector3 b = element_load(m, 0, qxy);
    expect_vector3_near(b, {{0.0, 1.0 / 48.0, 1.0 / 48.0}}, 1e-15, "q = x y:");
    EXPECT_NEAR(b[0] + b[1] + b[2], 1.0 / 24.0, 1e-15) << "q = x y: somma";
}

// ESEMPIO 82. Triangoli qualunque, con la sorgente lineare q = 2x + 3y - 1. Quattro nodi: 0 = (1, 1), 1 = (5, 1),
// 2 = (2, 4), 3 = (6, 5); T0 = (0, 1, 2) e T1 = (1, 3, 2).
//   T0: A = 1/2 * (4 * 3 - 1 * 0) = 6, quindi A / 6 = 1. I punti medi sono M01 = (3, 1), M12 = (7/2, 5/2) e
//       M20 = (3/2, 5/2), dove q vale 8, 27/2 e 19/2:
//           f = (8 + 19/2, 8 + 27/2, 27/2 + 19/2) = (35/2, 43/2, 23)
//   T1: i nodi locali sono 1 = (5, 1), 3 = (6, 5), 2 = (2, 4). A = 1/2 * (1 * 3 - (-3) * 4) = 15/2, quindi
//       A / 6 = 5/4. M01 = (11/2, 3), M12 = (4, 9/2), M20 = (7/2, 5/2), dove q vale 19, 41/2 e 27/2:
//           f = 5/4 * (19 + 27/2, 19 + 41/2, 41/2 + 27/2) = (325/8, 395/8, 85/2)
// Qui servono le coordinate dei nodi VERI del triangolo (non quelle del triangolo di riferimento), la sua area e
// l'ordine locale dei nodi: T1 elenca i nodi 1, 3, 2, non 0, 1, 2. Con q = 2x + 3y - 1, che e' lineare, sono
// gli integrali esatti: per esempio la somma di T0, 62, e' A per il valore di q nel baricentro (8/3, 2): 6 * 31/3.
TEST(Load2d, GeneralTrianglesUseTheirOwnNodesAndArea) {
    const Mesh2D m({1.0, 5.0, 2.0, 6.0}, {1.0, 1.0, 4.0, 5.0}, {{0, 1, 2}, {1, 3, 2}});
    const Source q = [](double x, double y) { return 2.0 * x + 3.0 * y - 1.0; };

    expect_vector3_near(element_load(m, 0, q), {{35.0 / 2.0, 43.0 / 2.0, 23.0}}, 1e-13, "triangolo 0:");
    expect_vector3_near(element_load(m, 1, q), {{325.0 / 8.0, 395.0 / 8.0, 85.0 / 2.0}}, 1e-13, "triangolo 1:");
}

// ESEMPIO 83. Le due celle quadrate di sempre, rectangle_mesh(2, 1, 2, 1): i nodi 0, 1, 2 stanno in basso (x = 0, 1,
// 2) e 3, 4, 5 in alto. I triangoli sono T0 = (0, 1, 4), T1 = (0, 4, 3), T2 = (1, 2, 5) e T3 = (1, 5, 4), tutti di
// area 1/2, quindi A / 6 = 1/12. Sorgente q = x (lineare). In un triangolo ogni nodo riceve 1/12 per la somma delle
// ascisse dei punti medi dei suoi due lati. Le ascisse dei punti medi dei lati 0-1, 1-2, 2-0 di ogni triangolo:
//     T0: 1/2, 1, 1/2        T1: 1/2, 1/2, 0        T2: 3/2, 2, 3/2        T3: 3/2, 3/2, 1
// Quindi, in dodicesimi (somma delle due ascisse):
//     nodo 0:  T0 (1/2 + 1/2) = 1,   T1 (1/2 + 0) = 1/2                             totale 3/2  ->  1/8
//     nodo 1:  T0 (1/2 + 1) = 3/2,   T2 (3/2 + 3/2) = 3,   T3 (3/2 + 1) = 5/2       totale 7    ->  7/12
//     nodo 2:  T2 (3/2 + 2) = 7/2                                                    totale 7/2  ->  7/24
//     nodo 3:  T1 (1/2 + 0) = 1/2                                                    totale 1/2  ->  1/24
//     nodo 4:  T0 (1 + 1/2) = 3/2,   T1 (1/2 + 1/2) = 1,   T3 (3/2 + 1) = 5/2       totale 5    ->  5/12
//     nodo 5:  T2 (2 + 3/2) = 7/2,   T3 (3/2 + 3/2) = 3                             totale 13/2 ->  13/24
// Il carico totale e' 2, l'integrale di x sul rettangolo 2 x 1: l'area 2 per l'ascissa del baricentro, 1.
TEST(Load2d, LoadOfTwoSquareCellsWithLinearSourceMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const Source q = [](double x, double) { return x; };

    expect_vector_near(assemble_load_2d(m, q),
                       {1.0 / 8.0, 7.0 / 12.0, 7.0 / 24.0, 1.0 / 24.0, 5.0 / 12.0, 13.0 / 24.0}, 1e-14, "carico");
}

// ESEMPIO 84. Una sorgente costante e' un caso particolare: la funzione nuova con q(x, y) = c deve dare lo stesso
// risultato della funzione di prima con il numero c, su ogni mesh e per ogni triangolo. Anche con c negativo (un
// pozzo di calore) e con c = 0.
TEST(Load2d, ConstantSourceGivesTheUniformLoad) {
    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh numero " + std::to_string(number++));
        for (double c : {7.0, -2.5, 0.0}) {
            const Source constant = [c](double, double) { return c; };
            expect_vector_near(assemble_load_2d(m, constant), assemble_load_2d(m, c), 1e-13,
                               "c = " + std::to_string(c) + ", mesh intera");
            for (std::size_t e = 0; e < m.num_triangles(); ++e) {
                const double tol = 1e-13 * (1.0 + std::fabs(c) * m.area(e));
                expect_vector3_near(element_load(m, e, constant), element_load(m, e, c), tol,
                                    "c = " + std::to_string(c) + ", triangolo " + std::to_string(e) + ":");
            }
        }
    }
}

// ESEMPIO 85. Se q e' lineare, il calcolo si puo' fare per un'altra strada, che non usa i punti medi. Sia q_i il
// valore di q nel nodo i. Su ogni triangolo la funzione lineare q e' q_0 N_0 + q_1 N_1 + q_2 N_2, quindi
//     f[i] = integrale di q N_i = somma sui nodi j del triangolo di  q_j * (integrale di N_i N_j)
//          = somma sui nodi j di  M[i][j] q_j
// con M la matrice di massa (rho_c = 1) di sempre. Sommando i triangoli: il carico di tutta la mesh e' M q_nodi, il
// prodotto della matrice di massa per il vettore dei valori di q nei nodi. Il test lo controlla su sette mesh di
// forme diverse (rettangoli, triangoli ottusi, due nodi fuori da ogni triangolo, che devono avere carico 0) con
// q = 3 - 2x + y / 2. Il carico ha un numero per nodo, anche per i nodi che non stanno in nessun triangolo.
TEST(Load2d, LinearSourceGivesTheMassMatrixTimesTheNodalValues) {
    const Source q = [](double x, double y) { return 3.0 - 2.0 * x + 0.5 * y; };

    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh numero " + std::to_string(number++));
        std::vector<double> nodal(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            nodal[i] = q(m.x(i), m.y(i));
        }
        const std::vector<double> want = assemble_mass_2d(m, 1.0) * nodal;
        expect_vector_near(assemble_load_2d(m, q), want, 1e-12, "carico");
    }
}

// ESEMPIO 86. Il calore totale: la somma dei carichi di tutti i nodi e' l'integrale di q su tutta la lastra. Per
// ogni triangolo la somma dei tre numeri e' l'integrale esatto se q ha grado <= 2 (ESEMPIO 81), quindi lo e' anche
// su tutta la mesh, per qualunque forma dei triangoli. Si prende
//     q = 3/2 - 2 x + 3/4 y + 3 x^2 - 4 x y + 5/2 y^2         (con numeri positivi e negativi)
// e l'integrale esatto, calcolato con le formule dei monomi:
//   - sul rettangolo [0, a] x [0, b]:  integrale di x^i y^j = a^(i+1) b^(j+1) / ((i + 1) (j + 1));
//   - sul triangolo di vertici (0, 0), (3, 0), (0, 3): integrale di x^i y^j = 3^(i+j+2) i! j! / (i + j + 2)!, cioe'
//     9/2 per 1, 9/2 per x e per y (A per il baricentro (1, 1)), 27/4 per x^2 e y^2, 27/8 per x y.
// Sui rettangoli anche le mesh con i nodi interni spostati: il bordo non cambia, quindi la lastra e' sempre quella.
TEST(Load2d, QuadraticSourceAddsUpToTheExactTotalHeat) {
    const double c0 = 1.5;
    const double cx = -2.0;
    const double cy = 0.75;
    const double cxx = 3.0;
    const double cxy = -4.0;
    const double cyy = 2.5;
    const Source q = [=](double x, double y) {
        return c0 + cx * x + cy * y + cxx * x * x + cxy * x * y + cyy * y * y;
    };

    struct Shape {
        double a;
        double b;
        std::size_t nx;
        std::size_t ny;
        unsigned long long seed;   // 0: rettangolo regolare
    };
    const std::vector<Shape> shapes = {
        {3.0, 1.5, 4, 3, 0}, {2.0, 1.0, 8, 6, 5}, {1.0, 1.0, 12, 12, 31}, {0.5, 4.0, 3, 9, 7}};
    for (const Shape& s : shapes) {
        SCOPED_TRACE("rettangolo " + std::to_string(s.a) + " x " + std::to_string(s.b) + ", seme " +
                     std::to_string(s.seed));
        const Mesh2D m = s.seed == 0 ? rectangle_mesh(s.a, s.b, s.nx, s.ny)
                                     : perturbed_rectangle(s.a, s.b, s.nx, s.ny, s.seed);
        const double exact = c0 * s.a * s.b + cx * s.a * s.a * s.b / 2.0 + cy * s.a * s.b * s.b / 2.0 +
                             cxx * s.a * s.a * s.a * s.b / 3.0 + cxy * s.a * s.a * s.b * s.b / 4.0 +
                             cyy * s.a * s.b * s.b * s.b / 3.0;
        EXPECT_NEAR(total_of(assemble_load_2d(m, q)), exact, 1e-12 * (1.0 + std::fabs(exact)));
    }

    const double exact_triangle = c0 * 4.5 + cx * 4.5 + cy * 4.5 + cxx * 6.75 + cxy * 3.375 + cyy * 6.75;
    EXPECT_NEAR(total_of(assemble_load_2d(triangle_with_interior_node(), q)), exact_triangle,
                1e-12 * (1.0 + std::fabs(exact_triangle)))
        << "triangolo con un nodo dentro";
}

// ESEMPIO 87. Un triangolo che non esiste: std::out_of_range, come per la funzione di prima.
TEST(Load2d, ElementLoadOfAnUnknownTriangleThrows) {
    const Mesh2D m = right_triangle();   // ha solo il triangolo 0
    const Source q = [](double x, double y) { return x + y; };

    EXPECT_NO_THROW(element_load(m, 0, q));
    EXPECT_THROW(element_load(m, 1, q), std::out_of_range);
    EXPECT_THROW(element_load(m, 1000, q), std::out_of_range);
}