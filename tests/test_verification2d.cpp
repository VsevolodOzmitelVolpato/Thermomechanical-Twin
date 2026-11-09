#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Fem1D.h"
#include "Mesh2D.h"
#include "Verification2D.h"

// ---- Verifica in 2D: l'errore L2 e la soluzione fabbricata ------------------------------------------------
//
// In 1D abbiamo misurato se il codice converge come deve confrontandolo con una soluzione esatta: l2_error,
// observed_order, decaying_sine_error. Qui facciamo lo stesso per la lastra 2D in regime stazionario.
//
// L'ERRORE L2. Il metodo da' una temperatura per nodo, T_num; la soluzione esatta, presa nei nodi, e' un'altra,
// T_ref. Fra i nodi tutte e due sono lineari a tratti. Con e = T_num - T_ref, in un triangolo
//
//     e(x, y) = e0 N0 + e1 N1 + e2 N2         (e_i = valore di e nel nodo locale i)
//
// quindi l'integrale di e^2 sul triangolo e' la somma su i, j di e_i e_j * (integrale di N_i N_j) = e^T M e, con M
// la matrice di massa con rho_c = 1: A / 6 sulla diagonale, A / 12 fuori. Sommando i triangoli (come
// nell'assemblaggio)
//
//     || e ||_L2 = sqrt( integrale di e^2 dA ) = sqrt( e^T M e )        con M la matrice di massa di tutta la mesh.
//
// Su un solo triangolo diventa   integrale di e^2 dA = A / 12 * ( e0^2 + e1^2 + e2^2 + (e0 + e1 + e2)^2 ).
//
// LA SOLUZIONE FABBRICATA. Sul quadrato [0, 1] x [0, 1] si sceglie la temperatura T(x, y) = sin(pi x) sin(pi y),
// che e' 0 su tutto il bordo. Le derivate seconde sono T_xx = T_yy = -pi^2 T, quindi la sorgente che la produce e'
//
//     q(x, y) = -k (T_xx + T_yy) = 2 pi^2 k sin(pi x) sin(pi y)
//
// e T non dipende da k. Si risolve il problema con questa sorgente e con T = 0 sul bordo, e si confronta con la T
// scelta: se il codice e' giusto l'errore scende come deve quando la mesh si infittisce.

namespace {

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

// ESEMPIO 88. Il triangolo (0, 0), (1, 0), (0, 1): A = 1/2, quindi M ha 1/12 sulla diagonale e 1/24 fuori. Per un
// errore e = (e0, e1, e2) nei tre nodi
//     || e ||^2 = 1/12 (e0^2 + e1^2 + e2^2) + 1/12 (e0 e1 + e1 e2 + e0 e2) = 1/24 ( somma e_i^2 + (somma e_i)^2 )
//   e = (1, 1, 1):    1/24 (3 + 9)  = 1/2      ->  || e || = sqrt(1/2)    (e costante: l'integrale e' l'area, 1/2)
//   e = (1, 0, 0):    1/24 (1 + 1)  = 1/12     ->  sqrt(1/12)             (e = N_0: l'integrale di N_0^2 e' A / 6)
//   e = (1, -1, 0):   1/24 (2 + 0)  = 1/12     ->  sqrt(1/12)
//   e = (1, 2, 3):    1/24 (14 + 36) = 25/12   ->  5 / sqrt(12)
//   e = (2, 0, -1):   1/24 (5 + 1)  = 1/4      ->  1/2
// Contano solo le differenze: T_num e T_ref sono scelti diversi da e e da 0, e scambiandoli e cambia segno ma la
// norma no.
TEST(Verification2D, L2ErrorOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = right_triangle();

    EXPECT_NEAR(l2_error_2d(m, {4.0, 3.0, 7.0}, {3.0, 2.0, 6.0}), std::sqrt(0.5), 1e-14) << "e = (1, 1, 1)";
    EXPECT_NEAR(l2_error_2d(m, {3.0, 2.0, 6.0}, {4.0, 3.0, 7.0}), std::sqrt(0.5), 1e-14) << "e = (-1, -1, -1)";
    EXPECT_NEAR(l2_error_2d(m, {5.0, 1.0, 2.0}, {4.0, 1.0, 2.0}), std::sqrt(1.0 / 12.0), 1e-14) << "e = (1, 0, 0)";
    EXPECT_NEAR(l2_error_2d(m, {2.0, 0.0, 1.0}, {1.0, 1.0, 1.0}), std::sqrt(1.0 / 12.0), 1e-14) << "e = (1, -1, 0)";
    EXPECT_NEAR(l2_error_2d(m, {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}), 5.0 / std::sqrt(12.0), 1e-14) << "e = (1, 2, 3)";
    EXPECT_NEAR(l2_error_2d(m, {2.0, 0.0, -1.0}, {0.0, 0.0, 0.0}), 0.5, 1e-14) << "e = (2, 0, -1)";
    EXPECT_NEAR(l2_error_2d(m, {7.0, -1.5, 3.0}, {7.0, -1.5, 3.0}), 0.0, 1e-14) << "e = 0";
}

// ESEMPIO 89. Se e e' una funzione lineare, e(x, y) = c0 + cx x + cy y, i suoi valori nei nodi la descrivono
// ESATTAMENTE, anche su triangoli di forme qualunque, e la norma si calcola a mano su tutta la lastra [0, a] x [0, b]
// con le formule dei monomi (integrale di x^i y^j = a^(i+1) b^(j+1) / ((i + 1) (j + 1))):
//     integrale di (c0 + cx x + cy y)^2 = c0^2 a b + cx^2 a^3 b / 3 + cy^2 a b^3 / 3
//                                         + c0 cx a^2 b + c0 cy a b^2 + cx cy a^2 b^2 / 2
// Si prende T_num = 3/2 - 2 x + 3/4 y e T_ref = 1/2 + x - y / 4, cioe' e = 1 - 3 x + y (c0 = 1, cx = -3, cy = 1).
// Per esempio sul quadrato unitario: 1 + 9/3 + 1/3 - 3 + 1 - 3/2 = 5/6. Anche sulle mesh con i nodi interni
// spostati: il bordo non cambia, quindi la lastra e' sempre quella.
TEST(Verification2D, L2ErrorOfALinearFieldIsTheExactIntegral) {
    const double c0 = 1.0;
    const double cx = -3.0;
    const double cy = 1.0;

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
        DVector t_num(m.num_nodes());
        DVector t_ref(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            t_num[i] = 1.5 - 2.0 * m.x(i) + 0.75 * m.y(i);
            t_ref[i] = 0.5 + m.x(i) - 0.25 * m.y(i);
        }
        const double exact = c0 * c0 * s.a * s.b + cx * cx * s.a * s.a * s.a * s.b / 3.0 +
                             cy * cy * s.a * s.b * s.b * s.b / 3.0 + c0 * cx * s.a * s.a * s.b +
                             c0 * cy * s.a * s.b * s.b + cx * cy * s.a * s.a * s.b * s.b / 2.0;
        EXPECT_NEAR(l2_error_2d(m, t_num, t_ref), std::sqrt(exact), 1e-12 * (1.0 + std::sqrt(exact)));
    }
}

// ESEMPIO 90. Una seconda strada per lo stesso numero, che non assembla nessuna matrice: la formula su un solo
// triangolo, A / 12 * ( e0^2 + e1^2 + e2^2 + (e0 + e1 + e2)^2 ), sommata sui triangoli. Si confronta su sette mesh
// di forme diverse (rettangoli, triangoli ottusi, e una con due nodi che non stanno in nessun triangolo: il loro
// valore non deve contare) con vettori "a caso" ma sempre uguali, di valori fra -4 e 6.
TEST(Verification2D, L2ErrorAgreesWithTheTriangleByTriangleFormula) {
    unsigned long long state = 12345;
    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh numero " + std::to_string(number++));
        DVector t_num(m.num_nodes());
        DVector t_ref(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            t_num[i] = 10.0 * next_random(state) - 4.0;
            t_ref[i] = 10.0 * next_random(state) - 4.0;
        }
        double sum = 0.0;
        for (std::size_t e = 0; e < m.num_triangles(); ++e) {
            const Triangle& t = m.triangle(e);
            double s1 = 0.0;   // e0 + e1 + e2
            double s2 = 0.0;   // e0^2 + e1^2 + e2^2
            for (std::size_t i = 0; i < 3; ++i) {
                const double d = t_num[t[i]] - t_ref[t[i]];
                s1 += d;
                s2 += d * d;
            }
            sum += m.area(e) / 12.0 * (s2 + s1 * s1);
        }
        EXPECT_NEAR(l2_error_2d(m, t_num, t_ref), std::sqrt(sum), 1e-12 * (1.0 + std::sqrt(sum)));
    }
}

// ESEMPIO 91. T_num e T_ref hanno un valore per nodo: altrimenti std::invalid_argument, anche se solo uno dei due e'
// sbagliato o se tutti e due lo sono (stessa lunghezza, ma non quella dei nodi).
TEST(Verification2D, L2ErrorRejectsVectorsOfTheWrongSize) {
    const Mesh2D m = right_triangle();   // 3 nodi
    const DVector ok(3, 1.0);
    const DVector too_short(2, 1.0);
    const DVector too_long(4, 1.0);
    const DVector empty;

    EXPECT_THROW(l2_error_2d(m, too_short, ok), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, ok, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_long, ok), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, ok, too_long), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_short, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_long, too_long), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, empty, empty), std::invalid_argument);
    EXPECT_NO_THROW(l2_error_2d(m, ok, ok));
}

// ESEMPIO 92. n = 2: la mesh 2 x 2 ha 9 nodi, 8 sul bordo (T = 0) e uno solo al centro, c = (1/2, 1/2). C'e' una sola
// incognita, T_c, e si calcola a mano.
//   - Il centro sta in 6 triangoli, tutti rettangoli isosceli con i cateti lunghi h = 1/2: area A = h^2 / 2 = 1/8.
//   - Riga del centro di K: in un triangolo K[i][i] = k (b_i^2 + c_i^2) / (4 A). Nell'angolo retto (il lato opposto
//     e' l'ipotenusa: b^2 + c^2 = 2 h^2) vale k; negli altri due angoli (il lato opposto e' un cateto: b^2 + c^2 = h^2)
//     vale k / 2. In due celle il centro e' l'angolo retto di un triangolo (k ciascuna), nelle altre due e' un angolo
//     acuto di tutti e due i triangoli (k/2 + k/2 = k ciascuna): K[c][c] = 4 k. Gli altri nodi hanno T = 0, quindi
//     l'equazione e'  4 k T_c = f_c.
//   - Il carico: f_c = somma sui 6 triangoli di A / 6 * (q(M) + q(M')), con M e M' i punti medi dei due lati che
//     escono dal centro (ESEMPIO 80). Ognuno dei 6 lati che escono dal centro sta in due triangoli, quindi
//         f_c = 2 * 1/48 * (somma di q nei 6 punti medi).
//     I lati vanno verso (1/2, 0), (1, 1/2), (1/2, 1), (0, 1/2): punti medi (1/2, 1/4), (3/4, 1/2), (1/2, 3/4), (1/4, 1/2),
//     dove sin(pi x) sin(pi y) vale sqrt(2) / 2; e verso (1, 1) e (0, 0): punti medi (3/4, 3/4) e (1/4, 1/4), dove vale 1/2.
//     Somma: 4 sqrt(2) / 2 + 1/2 + 1/2 = 2 sqrt(2) + 1, quindi f_c = 2 pi^2 k (2 sqrt(2) + 1) / 24.
//   - T_c = f_c / (4 k) = pi^2 (1 + 2 sqrt(2)) / 48 = 0.78719..., mentre la soluzione esatta nel centro e' 1. k non
//     compare: il risultato e' lo stesso per ogni k.
//   - L'errore e' e_c = T_c - 1 nel centro e (quasi) 0 negli altri nodi; M[c][c] = 6 * A / 6 = 1/8, quindi
//         || e ||_L2 = |e_c| / sqrt(8) = (1 - T_c) / (2 sqrt(2)) = 0.0752...
// Con n = 1 i quattro nodi stanno tutti sul bordo: T = 0 dappertutto, come la soluzione esatta, e l'errore e' 0.
TEST(Verification2D, SingleUnknownCaseMatchesHandComputation) {
    const double pi = std::acos(-1.0);
    const double sqrt2 = std::sqrt(2.0);
    const double center = pi * pi * (1.0 + 2.0 * sqrt2) / 48.0;
    const double want = (1.0 - center) / (2.0 * sqrt2);

    for (double k : {1.0, 3.5, 0.25}) {
        EXPECT_NEAR(manufactured_sine_error_2d(2, k), want, 1e-12) << "k = " << k;
    }
    EXPECT_NEAR(manufactured_sine_error_2d(1, 1.0), 0.0, 1e-15);
}

// ESEMPIO 93. I numeri della tabella del README, per la soluzione fabbricata con k = 1, misurati con questo metodo:
//     n = 4:  2.342328e-02      n = 8:  6.338199e-03      n = 16:  1.620920e-03
// Il test li fissa: se cambia il metodo (la regola del carico, il solutore, la norma) cambiano i numeri, e il test
// dice che la tabella del README va rifatta. Con k diverso da 1 l'errore e' lo stesso (T non dipende da k).
TEST(Verification2D, ErrorsMatchTheTableInTheReadme) {
    const double want_4 = 2.342328e-02;
    const double want_8 = 6.338199e-03;
    const double want_16 = 1.620920e-03;

    EXPECT_NEAR(manufactured_sine_error_2d(4, 1.0), want_4, 1e-6 * want_4);
    EXPECT_NEAR(manufactured_sine_error_2d(8, 1.0), want_8, 1e-6 * want_8);
    EXPECT_NEAR(manufactured_sine_error_2d(16, 1.0), want_16, 1e-6 * want_16);

    EXPECT_NEAR(manufactured_sine_error_2d(8, 7.5), want_8, 1e-6 * want_8) << "k = 7.5";
    EXPECT_NEAR(manufactured_sine_error_2d(8, 0.02), want_8, 1e-6 * want_8) << "k = 0.02";
}

// ESEMPIO 94. Con elementi lineari la teoria dice che l'errore L2 va come h^2: quando n raddoppia (h dimezza)
// l'errore si divide per 4, cioe' l'ordine osservato (observed_order, quello del 1D) e' 2. Misurato: 1.97 da 8 a 16,
// 1.99 da 16 a 32, 2.00 da 32 a 64. Da 4 a 8 e' 1.89: con una mesh cosi' rada non siamo ancora nell'andamento
// vero, per questo si parte da 8.
TEST(Verification2D, ErrorGoesDownWithObservedOrderTwo) {
    double previous = manufactured_sine_error_2d(8, 1.0);
    for (std::size_t n : {16, 32, 64}) {
        const double current = manufactured_sine_error_2d(n, 1.0);
        EXPECT_LT(current, previous) << "n = " << n / 2 << " -> " << n;
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "n = " << n / 2 << " -> " << n;
        previous = current;
    }
}

// ESEMPIO 95. Dati non validi: nessuna cella (n = 0), conducibilita nulla, negativa o NaN: std::invalid_argument.
TEST(Verification2D, ManufacturedErrorRejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(manufactured_sine_error_2d(0, 1.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, 0.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, -1.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, not_a_number), std::invalid_argument);
    EXPECT_NO_THROW(manufactured_sine_error_2d(4, 1.0));
}