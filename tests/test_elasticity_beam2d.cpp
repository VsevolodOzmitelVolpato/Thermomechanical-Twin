#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Elasticity2D.h"
#include "Fem1D.h"
#include "Mesh2D.h"
#include "Verification2D.h"

// ---- La trave che si incurva: una soluzione esatta in deformazione piana -------------------------------------------
//
// Fin qui ogni funzione dell'elasticita' e' stata provata con numeri fatti a mano o con casi limite. Qui si prova tutta
// la catena insieme (mesh, matrice, forza termica, solutore, tensioni) su un problema di cui si conosce la soluzione
// ESATTA, come in 2D per il calore con il seno fabbricato (ESEMPI 93 e 94): si calcola l'errore del metodo e si guarda
// se scende come deve quando la mesh si infittisce.
//
// IL PROBLEMA. Una trave libera, il rettangolo [0, L] x [0, H], in deformazione piana, con la temperatura che cresce
// linearmente con l'altezza (G e' il gradiente: kelvin per ogni unita' di y):
//     T(x, y) = T_ref + G y~,        y~ = y - H / 2.
// Sopra il centro e' piu' caldo, sotto piu' freddo. La deformazione termica di un punto e'
//     eps_th = (1 + nu) alpha G y~ (1, 1, 0) = kappa y~ (1, 1, 0),        kappa = (1 + nu) alpha G.
//
// LA SOLUZIONE ESATTA. Cerchiamo uno spostamento (u, v) la cui deformazione sia proprio eps_th: allora B u - eps_th = 0
// e la tensione sigma = D (B u - eps_th) e' zero dappertutto. Le condizioni sono
//     du/dx = kappa y~,        dv/dy = kappa y~,        du/dy + dv/dx = 0.
// La prima dice u = kappa x y~ (piu' una funzione di y), la seconda v = kappa y~^2 / 2 (piu' una funzione di x). Con
// u = kappa x y~ si ha du/dy = kappa x, quindi la terza chiede dv/dx = -kappa x: la funzione di x in v deve essere
// -kappa x^2 / 2. Si ottiene
//     u(x, y) = kappa x y~,        v(x, y) = kappa (y~^2 - x^2) / 2.
// La trave non ha nessuna tensione: si incurva e basta. La linea media (y~ = 0) diventa u = 0, v = -kappa x^2 / 2, una
// parabola, che per kappa piccolo e' un arco di cerchio di raggio 1 / kappa: il lato caldo si allunga, quello freddo si
// accorcia. (In tensione piana sarebbe kappa = alpha G; in deformazione piana c'e' il fattore (1 + nu) in piu'.)
//
// LE TRE CONDIZIONI. La trave e' libera, quindi spostarla o ruotarla non cambia niente: servono tre condizioni che
// tolgano i tre moti rigidi (due traslazioni e una rotazione). Si prendono dalla soluzione esatta, nell'angolo (0, 0)
// e in quello (L, 0):
//     u(0, 0) = 0,        v(0, 0) = kappa H^2 / 8,        v(L, 0) = kappa (H^2 / 4 - L^2) / 2.
// Con queste il problema ha una sola soluzione e, se il metodo e' giusto, e' quella esatta.
//
// L'ERRORE DEL METODO. I triangoli sono lineari: lo spostamento e' lineare in ogni triangolo, quello esatto e' di
// secondo grado. E la temperatura di un triangolo e' la media dei suoi tre nodi, quindi la deformazione termica e'
// costante in ogni triangolo, mentre quella esatta cresce con y. Il metodo sbaglia, e la teoria dice come l'errore
// scende con la dimensione h delle celle: lo spostamento con h^2, la tensione con h (e' una derivata, e qui dovrebbe
// essere ZERO). Gli errori sono due numeri:
//     displacement = il massimo, sui nodi, della distanza fra il nodo calcolato e quello esatto,
//                    sqrt((u_FEM - u_esatto)^2 + (v_FEM - v_esatto)^2);
//     stress       = il massimo, sui triangoli, del modulo sqrt(sigma_xx^2 + sigma_yy^2 + sigma_xy^2) calcolato.
// Con poche celle in altezza i triangoli lineari sono troppo rigidi in flessione (con una cella sola quasi non si
// incurvano): l'andamento teorico compare dai livelli con 4 celle in altezza in poi.
//
// Le funzioni sono due, in Verification2D come quelle del seno.
//   - bending_beam_exact_displacement_2d(mesh, beam) da' gli spostamenti esatti nei nodi della mesh (2 numeri per
//     nodo);
//   - bending_beam_errors_2d(beam, nx, ny) risolve la trave con nx x ny celle (rectangle_mesh) e da' i due errori.
// Cosa devono fare, ed e' quello che i test controllano:
//   - gli spostamenti esatti si calcolano a mano, sul quadrato (ESEMPIO 197) e su una mesh irregolare con H = 3
//     (ESEMPIO 198); si controllano i valori dei tre vincoli e la linea media dell'acciaio (ESEMPIO 199); la loro
//     deformazione e' quella termica, provata con le differenze finite (ESEMPIO 200); dipendono solo da altezza, nu,
//     alpha e gradiente, e sono lineari in alpha G (ESEMPIO 201);
//   - senza gradiente di temperatura la trave non si deforma e non ha tensioni (ESEMPIO 202);
//   - gli errori hanno i numeri della tabella del README (ESEMPIO 203) e scendono con ordine 2 per lo spostamento e 1
//     per la tensione (ESEMPI 204 e 205);
//   - leggi di scala che il metodo deve rispettare: gli errori sono lineari in G e in alpha (ESEMPIO 206), solo la
//     tensione dipende da E (ESEMPIO 207), T_ref non conta (ESEMPIO 208), e se la trave si rimpicciolisce di un
//     fattore s lo spostamento scala come s^2 e la tensione come s (ESEMPIO 210);
//   - altri coefficienti di Poisson (ESEMPIO 209) e dati sbagliati (ESEMPIO 211).

namespace {

// Una trave con numeri comodi per i conti a mano: E = 1, nu = 1/4, alpha = 1/100 e gradient = 160, quindi
// kappa = (1 + 1/4) (1/100) 160 = 2. Lunghezza e altezza si scelgono caso per caso.
BendingBeam hand_beam(double length, double height) {
    BendingBeam beam;
    beam.length = length;
    beam.height = height;
    beam.E = 1.0;
    beam.nu = 0.25;
    beam.alpha = 0.01;
    beam.gradient = 160.0;
    beam.T_ref = 0.0;
    return beam;
}

// La trave d'acciaio: 4 m per 1 m, E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, la temperatura sale di 100 K per
// ogni metro in y (50 K in piu' sul lato alto, 50 K in meno su quello basso) e il riferimento e' 20 gradi centigradi.
// kappa = (1 + 0.3) 12e-6 100 = 1.56e-3 per metro: il raggio di curvatura e' 1 / kappa = 641 m.
BendingBeam steel_beam() {
    BendingBeam beam;
    beam.length = 4.0;
    beam.height = 1.0;
    beam.E = 210.0e9;
    beam.nu = 0.3;
    beam.alpha = 12.0e-6;
    beam.gradient = 100.0;
    beam.T_ref = 293.15;
    return beam;
}

// Gli errori della trave d'acciaio con 4 k x k celle (celle quadrate di lato H / k). Ogni livello si calcola una volta
// sola: i test che ne usano piu' d'uno non rifanno i conti.
const BendingBeamErrors& steel_errors(std::size_t k) {
    static std::map<std::size_t, BendingBeamErrors> computed;
    auto found = computed.find(k);
    if (found == computed.end()) {
        found = computed.emplace(k, bending_beam_errors_2d(steel_beam(), 4 * k, k)).first;
    }
    return found->second;
}

// Una coppia di errori, assegnata per nome (non conta l'ordine dei membri della struttura).
BendingBeamErrors errors_of(double displacement, double stress) {
    BendingBeamErrors errors;
    errors.displacement = displacement;
    errors.stress = stress;
    return errors;
}

// Due vettori uguali a meno di una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "componente " << i;
    }
}

// Due errori uguali a meno di una tolleranza relativa: displacement con displacement, stress con stress.
void expect_errors_near(const BendingBeamErrors& got, const BendingBeamErrors& want, double relative,
                        const std::string& label) {
    EXPECT_NEAR(got.displacement, want.displacement, relative * std::fabs(want.displacement))
        << label << ", spostamento";
    EXPECT_NEAR(got.stress, want.stress, relative * std::fabs(want.stress)) << label << ", tensione";
}

}  // namespace

// ESEMPIO 197. Il quadrato unitario, con kappa = 2 e H = 1 (quindi y~ = y - 1/2): u = 2 x y~ e v = (y~^2 - x^2).
//     nodo 0 = (0, 0): y~ = -1/2:  u = 0,               v = 1/4 - 0 = 1/4
//     nodo 1 = (1, 0): y~ = -1/2:  u = 2 (1) (-1/2) = -1,   v = 1/4 - 1 = -3/4
//     nodo 2 = (1, 1): y~ = +1/2:  u = 2 (1) (1/2) = 1,     v = 1/4 - 1 = -3/4
//     nodo 3 = (0, 1): y~ = +1/2:  u = 0,               v = 1/4
// Il vettore e' (u0, v0, u1, v1, u2, v2, u3, v3).
TEST(ElasticityBeam2d, ExactDisplacementOfTheUnitSquareByHand) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});

    const std::vector<double> got = bending_beam_exact_displacement_2d(mesh, hand_beam(1.0, 1.0));
    const std::vector<double> want = {0.0, 0.25, -1.0, -0.75, 1.0, -0.75, 0.0, 0.25};
    expect_vectors_near(got, want, 1e-14);
}

// ESEMPIO 198. Una mesh qualunque, non una griglia, e un'altezza diversa da 1: i cinque nodi della mesh irregolare
// (0, 0), (4, 0), (3, 3), (-1, 2), (1, 0.7) con kappa = 2 e H = 3, quindi y~ = y - 3/2. La funzione deve usare le
// coordinate dei nodi e la VERA altezza (H / 2 = 3/2, non 1/2).
//     nodo 0 (0, 0):     y~ = -3/2:  u = 2 (0) (-3/2) = 0,       v = (9/4 - 0) = 9/4
//     nodo 1 (4, 0):     y~ = -3/2:  u = 2 (4) (-3/2) = -12,     v = 9/4 - 16 = -55/4
//     nodo 2 (3, 3):     y~ = +3/2:  u = 2 (3) (3/2) = 9,        v = 9/4 - 9 = -27/4
//     nodo 3 (-1, 2):    y~ = +1/2:  u = 2 (-1) (1/2) = -1,      v = 1/4 - 1 = -3/4
//     nodo 4 (1, 0.7):   y~ = -4/5:  u = 2 (1) (-4/5) = -1.6,    v = 16/25 - 1 = -0.36
TEST(ElasticityBeam2d, ExactDisplacementOnAnIrregularMeshByHand) {
    const Mesh2D mesh({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                      {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});

    const std::vector<double> got = bending_beam_exact_displacement_2d(mesh, hand_beam(4.0, 3.0));
    const std::vector<double> want = {0.0, 2.25, -12.0, -13.75, 9.0, -6.75, -1.0, -0.75, -1.6, -0.36};
    expect_vectors_near(got, want, 1e-13);
}

// ESEMPIO 199. La trave d'acciaio, kappa = 1.56e-3, sulla mesh 8 x 2 (9 nodi per riga: la riga y = 0 ha i nodi 0..8, la
// riga y = 0.5 i nodi 9..17, la riga y = 1 i nodi 18..26; i nodi di una riga stanno in x = 0, 0.5, ..., 4).
//   - I tre vincoli (i valori che bending_beam_errors_2d impone nel nodo 0 = (0, 0) e nel nodo 8 = (4, 0)):
//         u(0, 0) = 0,    v(0, 0) = kappa H^2 / 8 = 1.56e-3 / 8 = 1.95e-4,
//         v(L, 0) = kappa (H^2 / 4 - L^2) / 2 = 1.56e-3 (0.25 - 16) / 2 = -0.012285.
//   - La linea media y = H / 2 = 0.5 (nodi 9..17): y~ = 0, quindi u = 0 e v = -kappa x^2 / 2. In punta (x = 4) la
//     freccia e' kappa L^2 / 2 = 1.56e-3 (16) / 2 = 0.01248, cioe' 12.48 mm verso il basso: una trave d'acciaio di
//     quattro metri, 100 K fra sopra e sotto un metro di altezza.
TEST(ElasticityBeam2d, TheSupportValuesAndTheCenterLineOfTheSteelBeam) {
    const double kappa = 1.56e-3;
    const Mesh2D mesh = rectangle_mesh(4.0, 1.0, 8, 2);

    const std::vector<double> u = bending_beam_exact_displacement_2d(mesh, steel_beam());
    ASSERT_EQ(u.size(), 54u);

    EXPECT_NEAR(u[0], 0.0, 1e-18);
    EXPECT_NEAR(u[1], 1.95e-4, 1e-15);
    EXPECT_NEAR(u[2 * 8 + 1], -0.012285, 1e-14);

    for (std::size_t i = 0; i <= 8; ++i) {
        const double x = 0.5 * static_cast<double>(i);
        EXPECT_NEAR(u[2 * (9 + i)], 0.0, 1e-18) << "x = " << x;
        EXPECT_NEAR(u[2 * (9 + i) + 1], -kappa * x * x / 2.0, 1e-15) << "x = " << x;
    }
    EXPECT_NEAR(u[2 * 17 + 1], -0.01248, 1e-14);
}

// ESEMPIO 200. La soluzione esatta ha come deformazione proprio quella termica. Lo si controlla con le differenze
// finite: in un punto (x0, y0) si prendono quattro nodi a distanza d (a destra, sopra, a sinistra, sotto) e
//     eps_xx = du/dx = (u_destra - u_sinistra) / (2 d),        eps_yy = dv/dy = (v_sopra - v_sotto) / (2 d),
//     gamma_xy = du/dy + dv/dx = (u_sopra - u_sotto) / (2 d) + (v_destra - v_sinistra) / (2 d).
// Gli spostamenti sono di secondo grado (u e' lineare in x e in y, v e' di secondo grado in x e in y): le differenze
// centrali sono ESATTE, quindi i tre numeri devono essere (a meno di arrotondamenti) quelli della deformazione
// termica di plane_strain_thermal_strain con dT = G y~ = G (y0 - H / 2), cioe' kappa y~ (1, 1, 0), e il taglio 0. Si
// prova con tre travi (kappa = 2 e H = 1, kappa = 2 e H = 3, l'acciaio) e quattro punti, anche fuori dalla trave.
TEST(ElasticityBeam2d, TheExactDisplacementCancelsTheThermalStrain) {
    const double d = 0.25;
    const std::vector<std::pair<double, double>> points = {{0.3, 0.2}, {0.9, 0.55}, {3.5, 0.95}, {-1.0, 2.0}};

    for (const BendingBeam& beam : {hand_beam(1.0, 1.0), hand_beam(4.0, 3.0), steel_beam()}) {
        const double kappa = (1.0 + beam.nu) * beam.alpha * beam.gradient;
        for (const auto& point : points) {
            const double x0 = point.first;
            const double y0 = point.second;
            // nodi: 0 = centro, 1 = destra, 2 = sopra, 3 = sinistra, 4 = sotto; un solo triangolo (0, 1, 2)
            const Mesh2D mesh({x0, x0 + d, x0, x0 - d, x0}, {y0, y0, y0 + d, y0, y0 - d}, {{0, 1, 2}});

            const std::vector<double> u = bending_beam_exact_displacement_2d(mesh, beam);
            ASSERT_EQ(u.size(), 10u);
            const double exx = (u[2 * 1] - u[2 * 3]) / (2.0 * d);
            const double eyy = (u[2 * 2 + 1] - u[2 * 4 + 1]) / (2.0 * d);
            const double gxy = (u[2 * 2] - u[2 * 4]) / (2.0 * d) + (u[2 * 1 + 1] - u[2 * 3 + 1]) / (2.0 * d);

            const Vector3 eps_th =
                plane_strain_thermal_strain(beam.nu, beam.alpha, beam.gradient * (y0 - beam.height / 2.0));
            const double tolerance = 1e-12 * std::fabs(kappa);
            EXPECT_NEAR(exx, eps_th[0], tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_NEAR(eyy, eps_th[1], tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_NEAR(gxy, 0.0, tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_GT(std::fabs(eps_th[0]), 1e-4 * std::fabs(kappa)) << "il punto deve stare fuori dalla linea media";
        }
    }
}

// ESEMPIO 201. Gli spostamenti esatti dipendono solo da altezza, nu, alpha e gradiente, e solo attraverso
// kappa = (1 + nu) alpha G, in modo lineare. Sulla mesh 8 x 2 della trave d'acciaio:
//   - un nodo preciso: il nodo 26 = (4, 1) ha y~ = 1/2, quindi u = kappa (4) (1/2) = 2 kappa = 3.12e-3 e
//     v = kappa (1/4 - 16) / 2 = -7.875 kappa = -0.012285;
//   - length, E e T_ref non c'entrano: cambiarli lascia il vettore identico;
//   - G doppio o alpha doppio raddoppia ogni numero, G cambiato di segno li cambia tutti di segno, G = 0 li azzera.
TEST(ElasticityBeam2d, TheExactDisplacementUsesOnlyHeightPoissonRatioAlphaAndGradient) {
    const Mesh2D mesh = rectangle_mesh(4.0, 1.0, 8, 2);
    const BendingBeam beam = steel_beam();
    const std::vector<double> base = bending_beam_exact_displacement_2d(mesh, beam);
    ASSERT_EQ(base.size(), 54u);

    EXPECT_NEAR(base[2 * 26], 3.12e-3, 1e-15);
    EXPECT_NEAR(base[2 * 26 + 1], -0.012285, 1e-14);

    BendingBeam other = beam;
    other.length = 17.0;
    other.E = 1.0;
    other.T_ref = -40.0;
    EXPECT_EQ(bending_beam_exact_displacement_2d(mesh, other), base);

    other = beam;
    other.gradient = 2.0 * beam.gradient;
    std::vector<double> got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], 2.0 * base[i]) << "gradient doppio, componente " << i;
    }

    other = beam;
    other.alpha = 2.0 * beam.alpha;
    got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], 2.0 * base[i]) << "alpha doppio, componente " << i;
    }

    other = beam;
    other.gradient = -beam.gradient;
    got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], -base[i]) << "gradient opposto, componente " << i;
    }

    other = beam;
    other.gradient = 0.0;
    for (double value : bending_beam_exact_displacement_2d(mesh, other)) {
        EXPECT_EQ(value, 0.0);
    }
}

// ESEMPIO 202. Senza gradiente di temperatura (G = 0) la temperatura e' T_ref dappertutto, la trave non si dilata e
// non si piega: la forza termica e' 0, lo spostamento esatto e' 0, quello calcolato e' 0 e le tensioni sono 0. Quindi
// tutti e due gli errori sono 0, su ogni mesh (si provano quattro).
TEST(ElasticityBeam2d, ABeamWithoutTemperatureGradientIsNeitherBentNorStressed) {
    BendingBeam beam = steel_beam();
    beam.gradient = 0.0;

    const std::vector<std::pair<std::size_t, std::size_t>> cells = {{1, 1}, {4, 1}, {8, 2}, {3, 5}};
    for (const auto& c : cells) {
        const BendingBeamErrors errors = bending_beam_errors_2d(beam, c.first, c.second);
        EXPECT_NEAR(errors.displacement, 0.0, 1e-15) << c.first << " x " << c.second << " celle";
        EXPECT_NEAR(errors.stress, 0.0, 1e-3) << c.first << " x " << c.second << " celle";
    }
}

// ESEMPIO 203. I numeri della tabella del README, per la trave d'acciaio (4 m per 1 m, G = 100 K/m) con 4 k x k celle,
// misurati con questo metodo (e confermati da un secondo programma indipendente, con un risolutore diretto):
//      k     spostamento       tensione (Pa)
//      1     2.2730600e-03     8.1688961e+07
//      2     1.3743698e-03     5.9064736e+07
//      4     5.5030371e-04     4.3477999e+07
//      8     1.6650762e-04     2.1585121e+07
//     16     4.5533882e-05     1.0131672e+07
// Il test li fissa: se cambia il metodo (la regola del carico, i vincoli, la norma dell'errore) cambiano i numeri, e
// il test dice che la tabella del README va rifatta. Con una cella in altezza la tensione calcolata e' 82 MPa, dove
// la esatta e' 0: per confronto la tensione termica tipica di questa trave e' E alpha dT = 210e9 x 12e-6 x 50 = 126
// MPa. La mesh non riesce a incurvarsi e il calore diventa tensione.
TEST(ElasticityBeam2d, ErrorsMatchTheTableInTheReadme) {
    struct Row {
        std::size_t k;
        double displacement;
        double stress;
    };
    const std::vector<Row> table = {{1, 2.2730600e-03, 8.1688961e+07},
                                    {2, 1.3743698e-03, 5.9064736e+07},
                                    {4, 5.5030371e-04, 4.3477999e+07},
                                    {8, 1.6650762e-04, 2.1585121e+07},
                                    {16, 4.5533882e-05, 1.0131672e+07}};

    for (const Row& row : table) {
        const BendingBeamErrors& errors = steel_errors(row.k);
        EXPECT_NEAR(errors.displacement, row.displacement, 1e-6 * row.displacement) << "k = " << row.k;
        EXPECT_NEAR(errors.stress, row.stress, 1e-6 * row.stress) << "k = " << row.k;
    }
}

// ESEMPIO 204. Con elementi lineari lo spostamento va come h^2: quando le celle si dimezzano l'errore si divide per
// 4, cioe' l'ordine osservato (observed_order) e' 2. Misurato: 0.73 da 1 a 2 celle in altezza, 1.32 da 2 a 4, 1.72 da
// 4 a 8, 1.87 da 8 a 16 (e 1.90 da 16 a 32): l'ordine sale verso 2 man mano che la mesh e' abbastanza fine per vedere
// la curvatura. Per questo si guarda da 4 in su: fra 1.6 e 2.1. In tutti i livelli l'errore deve comunque scendere.
TEST(ElasticityBeam2d, DisplacementErrorGoesDownWithObservedOrderTwo) {
    double previous = steel_errors(1).displacement;
    for (std::size_t k : {2u, 4u, 8u, 16u}) {
        const double current = steel_errors(k).displacement;
        EXPECT_LT(current, previous) << "k = " << k / 2 << " -> " << k;
        if (k >= 8) {
            const double order = observed_order(previous, current, 2.0);
            EXPECT_GT(order, 1.6) << "k = " << k / 2 << " -> " << k;
            EXPECT_LT(order, 2.1) << "k = " << k / 2 << " -> " << k;
        }
        previous = current;
    }
}

// ESEMPIO 205. La tensione e' una derivata dello spostamento: nei triangoli lineari e' costante, e l'errore va come
// h, cioe' l'ordine osservato e' 1 (qui l'errore e' tutta la tensione calcolata, perche' quella esatta e' 0).
// Misurato: 0.47 da 1 a 2 celle in altezza, 0.44 da 2 a 4, 1.01 da 4 a 8, 1.09 da 8 a 16, 1.01 da 16 a 32. Si guarda
// da 4 in su: fra 0.9 e 1.25. Anche qui l'errore deve scendere a ogni livello.
TEST(ElasticityBeam2d, StressErrorGoesDownWithObservedOrderOne) {
    double previous = steel_errors(1).stress;
    for (std::size_t k : {2u, 4u, 8u, 16u}) {
        const double current = steel_errors(k).stress;
        EXPECT_LT(current, previous) << "k = " << k / 2 << " -> " << k;
        if (k >= 8) {
            const double order = observed_order(previous, current, 2.0);
            EXPECT_GT(order, 0.9) << "k = " << k / 2 << " -> " << k;
            EXPECT_LT(order, 1.25) << "k = " << k / 2 << " -> " << k;
        }
        previous = current;
    }
}

// ESEMPIO 206. Il problema e' lineare: se il gradiente G (o alpha) raddoppia, la soluzione esatta raddoppia e quella
// del metodo anche, quindi i due errori raddoppiano; se G cambia segno la trave si incurva dall'altra parte e gli
// errori sono gli stessi. In generale gli errori sono proporzionali a |G| e ad alpha. Sulla mesh 8 x 2 (k = 2) con
// G = 250, G = -100, G = 2.5 e alpha = 6e-6 e 24e-6 (invece di 12e-6): i fattori sono 2.5, 1, 0.025, 0.5 e 2.
TEST(ElasticityBeam2d, ErrorsAreLinearInTheGradientAndInTheExpansionCoefficient) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double gradient : {250.0, -100.0, 2.5}) {
        BendingBeam beam = steel_beam();
        beam.gradient = gradient;
        const double factor = std::fabs(gradient) / 100.0;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(factor * base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "G = " + std::to_string(gradient));
    }
    for (double alpha : {6.0e-6, 24.0e-6}) {
        BendingBeam beam = steel_beam();
        beam.alpha = alpha;
        const double factor = alpha / 12.0e-6;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(factor * base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "alpha = " + std::to_string(alpha));
    }
}

// ESEMPIO 207. Il modulo di Young E moltiplica tutta la matrice D, quindi non cambia gli spostamenti
// (K u = F: K e F crescono insieme) ma moltiplica le tensioni (sigma = D (...)). Con E = 1, 2.1e9 e 4.2e11
// (l'acciaio e' 2.1e11, cioe' 210 GPa) l'errore di spostamento resta lo stesso e quello di tensione e' proporzionale
// a E: i fattori sono 1 / 2.1e11, 0.01 e 2.
TEST(ElasticityBeam2d, OnlyTheStressDependsOnYoungModulus) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double young : {1.0, 2.1e9, 4.2e11}) {
        BendingBeam beam = steel_beam();
        beam.E = young;
        const double factor = young / 210.0e9;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "E = " + std::to_string(young));
    }
}

// ESEMPIO 208. La deformazione termica dipende da T - T_ref, non da T: se si cambia T_ref cambia anche la temperatura
// T = T_ref + G y~ della trave, e la differenza resta G y~. Con T_ref = 0, -500 e 10000 gli errori sono gli stessi
// (rispetto a T_ref = 293.15). Chi dimentica T_ref in una delle due funzioni che lo usano (il carico termico e le
// tensioni) vede una trave scaldata di centinaia di gradi.
TEST(ElasticityBeam2d, TheReferenceTemperatureDoesNotMatter) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double reference : {0.0, -500.0, 10000.0}) {
        BendingBeam beam = steel_beam();
        beam.T_ref = reference;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        expect_errors_near(got, base, 1e-9, "T_ref = " + std::to_string(reference));
    }
}

// ESEMPIO 209. Altri coefficienti di Poisson, con la mesh k = 4 (16 x 4 celle). Se nu cambia, cambiano kappa =
// (1 + nu) alpha G, la matrice D e la deformazione termica: se nu e' usato in modo diverso in un posto solo (per
// esempio un 0.3 scritto nel codice) il metodo non converge piu' alla soluzione esatta e gli errori sono molto piu'
// grandi. Misurato (e confermato dal programma indipendente):
//       nu       spostamento       tensione (Pa)
//     -0.50      3.2753386e-04     3.2214892e+07     (materiale auxetico)
//      0.00      4.9988890e-04     3.6344696e+07
//      0.45      7.2274450e-04     6.9570010e+07
// Con nu vicino a 1/2 (materiale quasi incomprimibile) i triangoli lineari si irrigidiscono (il "bloccaggio
// volumetrico") e a parita' di mesh l'errore e' piu' grande.
TEST(ElasticityBeam2d, ErrorsForOtherPoissonRatiosMatchTheTable) {
    struct Row {
        double nu;
        double displacement;
        double stress;
    };
    const std::vector<Row> table = {{-0.5, 3.2753386e-04, 3.2214892e+07},
                                    {0.0, 4.9988890e-04, 3.6344696e+07},
                                    {0.45, 7.2274450e-04, 6.9570010e+07}};

    for (const Row& row : table) {
        BendingBeam beam = steel_beam();
        beam.nu = row.nu;
        const BendingBeamErrors errors = bending_beam_errors_2d(beam, 16, 4);
        EXPECT_NEAR(errors.displacement, row.displacement, 1e-6 * row.displacement) << "nu = " << row.nu;
        EXPECT_NEAR(errors.stress, row.stress, 1e-6 * row.stress) << "nu = " << row.nu;
    }
}

// ESEMPIO 210. Se la trave si rimpicciolisce (o ingrandisce) di un fattore s, con la stessa mesh
// (stesso numero di celle, 8 x 2) e lo stesso gradiente G (kelvin per metro), tutte le lunghezze sono s volte quelle
// di prima; la differenza di temperatura fra sopra e sotto e' s volte, quindi anche la deformazione termica e' s
// volte. La soluzione e' lineare: gli spostamenti (deformazione per lunghezza) sono s^2 volte, le tensioni
// (E per deformazione) s volte. Gli errori seguono: spostamento s^2, tensione s. Con s = 0.05
// (una trave di 20 cm per 5 cm), 3 e 20. Chi scrive 0.5 al posto di H / 2 non se ne accorge con H = 1, ma qui si'.
TEST(ElasticityBeam2d, ScalingTheBeamScalesTheErrors) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double s : {0.05, 3.0, 20.0}) {
        BendingBeam beam = steel_beam();
        beam.length = 4.0 * s;
        beam.height = 1.0 * s;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(s * s * base.displacement, s * base.stress);
        expect_errors_near(got, want, 1e-9, "s = " + std::to_string(s));
    }
}

// ESEMPIO 211. Dati non validi: std::invalid_argument per nx o ny uguali a 0; per lunghezza o altezza nulle, negative o
// NaN; per E nullo, negativo o NaN; per nu fuori da (-1, 1/2) (gli estremi esclusi) o NaN. Sono i controlli delle
// funzioni che bending_beam_errors_2d chiama (rectangle_mesh, plane_strain_matrix). Una trave valida non lancia.
TEST(ElasticityBeam2d, BadInputsThrowInvalidArgument) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(bending_beam_errors_2d(steel_beam(), 4, 1));
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 0, 1), std::invalid_argument);
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 4, 0), std::invalid_argument);
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 0, 0), std::invalid_argument);

    for (double bad : {0.0, -1.0, not_a_number}) {
        BendingBeam beam = steel_beam();
        beam.length = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "length = " << bad;

        beam = steel_beam();
        beam.height = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "height = " << bad;

        beam = steel_beam();
        beam.E = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "E = " << bad;
    }
    for (double bad : {-1.0, -2.0, 0.5, 0.75, not_a_number}) {
        BendingBeam beam = steel_beam();
        beam.nu = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "nu = " << bad;
    }
}