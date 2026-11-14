#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Thermoelastic2D.h"

// ---- La tensione di von Mises: un numero solo per dire quanto e' sollecitato il materiale -------------------------
//
// Le tre tensioni del piano (xx, yy, xy) non dicono da sole se il materiale e' in pericolo. Un metallo compresso da
// tutte le parti con lo stesso valore (come un sasso in fondo al mare) non cede, anche con 1000 MPa; un metallo tirato
// da una parte sola cede a 300 MPa. A fare cedere il materiale e' la parte di tensione che cambia la FORMA, non la
// pressione uguale in tutte le direzioni. La tensione equivalente di von Mises e' un numero solo che misura proprio
// quella parte, ed e' quello che si confronta con la tensione di snervamento. Con le tre tensioni normali (xx, yy, zz)
// e il taglio xy:
//     sigma_vm = sqrt( 1/2 [ (sxx - syy)^2 + (syy - szz)^2 + (szz - sxx)^2 ] + 3 sxy^2 ).
// Si scrive con le DIFFERENZE fra le tensioni normali: sommare la stessa pressione a tutte e tre non cambia niente. Per
// la trazione semplice (solo sxx = s) da' |s|: e' cosi' che e' stata tarata.
//
// PERCHE' SERVE sigma_zz. La lastra e' in deformazione piana: non si allunga in z (epsilon_zz = 0). Ma "non si allunga"
// non vuol dire "nessuna tensione in z": il corpo e' trattenuto, e la tensione in z e' quella che lo trattiene. La
// legge di Hooke in z, con alpha dT la dilatazione termica che vorrebbe avvenire, dice (nu = coefficiente di Poisson)
//     0 = epsilon_zz = [ szz - nu (sxx + syy) ] / E + alpha dT,        cioe'        szz = nu (sxx + syy) - E alpha dT.
// La formula "da lastra sottile" (szz = 0) qui e' SBAGLIATA, e i test lo mostrano con dei numeri.
//
// TRE FATTI CHE I TEST USANO (con E = 100, nu = 1/4, alpha = 1/100 si ha E alpha = 1).
//   1. Una lastra libera e scaldata in modo uniforme non ha tensioni nel piano (ESEMPIO 216), ma ha szz = -E alpha dT:
//      vorrebbe dilatarsi anche in z e non puo'. La sua von Mises e' E alpha dT, non 0.
//   2. Una lastra incastrata e scaldata in modo uniforme ha sxx = syy = -2 dT (ESEMPIO 218) e, con la formula di szz,
//      anche szz = 0.25 (-4 dT) - dT = -2 dT: tre tensioni uguali, una compressione uguale in tutte le direzioni. La
//      von Mises e' 0, anche se le tensioni nel piano sono grandi.
//   3. A meta' di una striscia a rulli lunga la lastra non si deforma in x e in z (epsilon_xx = epsilon_zz = 0), e da'
//      szz = sxx in ogni triangolo: la von Mises e' |sxx - syy|, e vale 40 nei numeri a mano (ESEMPIO 221), cioe' la
//      formula del libro E alpha dT / (1 - nu) con dT = 30 (ESEMPIO 236). Nella lastra d'acciaio e' il numero che
//      dava la media di sxx nell'ESEMPIO 223 (ESEMPIO 237).
//
// Cosa devono fare le due funzioni, ed e' quello che i test controllano:
//   - von_mises_2d: gli stati semplici (ESEMPIO 228), un esempio a mano con tutte le componenti (229), lo stato
//     idrostatico (230), cosa non cambia (231 e 232);
//   - von_mises_field_2d: ogni triangolo usa le sue tensioni e i suoi tre nodi (233), la lastra libera (234), quella
//     incastrata (235), la striscia (236), la lastra d'acciaio (237), le leggi di scala (238) e i dati sbagliati (239);
//   - tutte e due: un NaN nei dati resta un NaN e non diventa un numero (240).

namespace {

const double kPi = 3.14159265358979323846;

// Numeri comodi per i conti a mano: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20 (E alpha = 1, D = [[120, 40, 0],
// [40, 120, 0], [0, 0, 40]]).
void use_hand_material(ThermoMechanicalProblem2D& problem) {
    problem.E = 100.0;
    problem.nu = 0.25;
    problem.alpha = 0.01;
    problem.T_ref = 20.0;
}

// La lastra a rulli di nx x ny celle, rectangle_mesh(length, height, nx, ny), senza carico di calore (f = 0): le due
// facce (y = 0 e y = H) restano a t_ref, la temperatura iniziale e' t_ref + amplitude sin(pi j / ny) nella riga j di
// nodi, u = 0 nei nodi dei due lati x = 0 e x = L e v = 0 nel nodo 0. Il materiale, i numeri del calore e il tempo li
// sceglie chi chiama (e' la stessa lastra dei test di Thermoelastic2D).
ThermoMechanicalProblem2D roller_slab_problem(std::size_t nx, std::size_t ny, double t_ref, double amplitude) {
    const std::size_t n = (nx + 1) * (ny + 1);
    ThermoMechanicalProblem2D problem;
    problem.f = DVector(n, 0.0);
    problem.T0 = DVector(n, t_ref);
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const double ratio = static_cast<double>(j) / static_cast<double>(ny);
            problem.T0[j * (nx + 1) + i] = t_ref + amplitude * std::sin(kPi * ratio);
        }
    }
    for (std::size_t i = 0; i <= nx; ++i) {
        problem.fixed_nodes.push_back(i);
        problem.fixed_temperatures.push_back(t_ref);
        problem.fixed_nodes.push_back(ny * (nx + 1) + i);
        problem.fixed_temperatures.push_back(t_ref);
    }
    for (std::size_t j = 0; j <= ny; ++j) {
        problem.fixed_dofs.push_back(2 * (j * (nx + 1)));
        problem.fixed_displacements.push_back(0.0);
        problem.fixed_dofs.push_back(2 * (j * (nx + 1) + nx));
        problem.fixed_displacements.push_back(0.0);
    }
    problem.fixed_dofs.push_back(1);
    problem.fixed_displacements.push_back(0.0);
    return problem;
}

// Il problema "generale": una mesh irregolare e tutti i numeri diversi uno dall'altro (come negli ESEMPI 214 e 224).
Mesh2D general_mesh() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

ThermoMechanicalProblem2D general_problem(const Mesh2D& mesh) {
    ThermoMechanicalProblem2D problem;
    problem.rho_c = 2.5;
    problem.k = 0.7;
    problem.f = assemble_load_2d(mesh, 3.0);
    problem.fixed_nodes = {0, 3};
    problem.fixed_temperatures = {310.0, 280.0};
    problem.T0 = {310.0, 305.0, 295.0, 280.0, 290.0};
    problem.dt = 0.2;
    problem.n_steps = 6;
    problem.theta = 0.6;
    problem.save_every = 2;
    problem.E = 70.0;
    problem.nu = 0.2;
    problem.alpha = 0.003;
    problem.T_ref = 290.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};
    return problem;
}

// La von Mises di tutti i triangoli di uno stato della catena: con il materiale e la T_ref del problema che lo ha dato.
std::vector<double> von_mises_of(const Mesh2D& mesh, const ThermoMechanicalProblem2D& problem,
                                 const ThermoMechanicalState2D& state) {
    return von_mises_field_2d(mesh, state.sigma, problem.E, problem.nu, problem.alpha, state.T, problem.T_ref);
}

// Il valore assoluto piu' grande di un vettore, e delle tensioni (tutte le componenti di tutti i triangoli).
double max_abs(const std::vector<double>& values) {
    double largest = 0.0;
    for (double value : values) {
        largest = std::max(largest, std::fabs(value));
    }
    return largest;
}

double max_abs(const std::vector<Vector3>& stresses) {
    double largest = 0.0;
    for (const Vector3& s : stresses) {
        for (std::size_t c = 0; c < 3; ++c) {
            largest = std::max(largest, std::fabs(s[c]));
        }
    }
    return largest;
}

// Tutti i numeri di un vettore uguali a want, a meno di una tolleranza assoluta.
void expect_all_near(const std::vector<double>& got, double want, double tolerance, const std::string& label) {
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want, tolerance) << label << ", triangolo " << i;
    }
}

// Due vettori uguali a meno di una tolleranza assoluta.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance,
                         const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << label << ", triangolo " << i;
    }
}

// Il problema generale con tutto il "calore" (lo scarto da T_ref della temperatura iniziale e delle temperature
// imposte, e il carico) moltiplicato per factor.
ThermoMechanicalProblem2D with_heat_times(const ThermoMechanicalProblem2D& base, double factor) {
    ThermoMechanicalProblem2D hot = base;
    for (double& value : hot.T0) {
        value = base.T_ref + factor * (value - base.T_ref);
    }
    for (double& value : hot.fixed_temperatures) {
        value = base.T_ref + factor * (value - base.T_ref);
    }
    for (double& value : hot.f) {
        value *= factor;
    }
    return hot;
}

// La von Mises di tutti e 4 gli stati del problema generale cambiato, uguale a quella del problema base per factor.
void expect_von_mises_scaled(const Mesh2D& mesh, const ThermoMechanicalProblem2D& changed,
                             const std::vector<std::vector<double>>& base_vm, double factor, const std::string& what) {
    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, changed);
    ASSERT_EQ(states.size(), base_vm.size()) << what;
    for (std::size_t s = 0; s < states.size(); ++s) {
        std::vector<double> want = base_vm[s];
        for (double& value : want) {
            value *= factor;
        }
        expect_vectors_near(von_mises_of(mesh, changed, states[s]), want, 1e-9 * max_abs(base_vm[s]) + 1e-12,
                            what + ", stato " + std::to_string(s));
    }
}

}  // namespace

// ESEMPIO 228. Gli stati semplici danno il numero che ci si aspetta. Con sigma = (xx, yy, xy) e la tensione fuori dal
// piano szz:
//   - trazione sola, (100, 0, 0) con szz = 0: le tre differenze valgono 100, 0 e -100, quindi 1/2 (100^2 + 0 + 100^2) =
//     100^2 e la von Mises e' 100: per la trazione semplice e' la tensione stessa. Lo stesso per (0, -40, 0): 40 (il
//     segno non conta);
//   - la sola szz = 25 (xx = yy = xy = 0): le differenze valgono 0, -25 e 25, quindi 25;
//   - biassiale uguale nel piano, (50, 50, 0) con szz = 0: le differenze valgono 0, 50 e -50, quindi 1/2 (0 + 2500 +
//     2500) = 2500 e la von Mises e' 50;
//   - taglio puro, (0, 0, 10): 3 x 10^2 = 300, quindi 10 sqrt(3) = 17.32. Il taglio "pesa" sqrt(3) volte una trazione;
//   - tutto 0: 0.
TEST(VonMises2d, TheSimpleStatesGiveTheNumberOneExpects) {
    EXPECT_DOUBLE_EQ(von_mises_2d({100.0, 0.0, 0.0}, 0.0), 100.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({0.0, -40.0, 0.0}, 0.0), 40.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({0.0, 0.0, 0.0}, 25.0), 25.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({50.0, 50.0, 0.0}, 0.0), 50.0);
    EXPECT_NEAR(von_mises_2d({0.0, 0.0, 10.0}, 0.0), 10.0 * std::sqrt(3.0), 1e-12);
    EXPECT_EQ(von_mises_2d({0.0, 0.0, 0.0}, 0.0), 0.0);
}

// ESEMPIO 229. Un esempio a mano con tutte le componenti: sigma = (5, 2, 3) e szz = 2. Le differenze fra le normali
// sono 5 - 2 = 3, 2 - 2 = 0 e 2 - 5 = -3, quindi
//     1/2 (3^2 + 0^2 + (-3)^2) + 3 x 3^2 = 9 + 27 = 36,        sigma_vm = 6.
// Un secondo, (4, 1, 1) con szz = 0: differenze 3, 1 e -4, quindi 1/2 (9 + 1 + 16) + 3 x 1 = 13 + 3 = 16 e la von
// Mises e' 4. E lo stato del primo esempio con szz = 0 invece di 2 (differenze 3, 2 e -5: 1/2 (9 + 4 + 25) + 27 = 46)
// da' sqrt(46) = 6.78, un altro numero: szz conta.
TEST(VonMises2d, AHandExampleWithAllTheComponents) {
    EXPECT_DOUBLE_EQ(von_mises_2d({5.0, 2.0, 3.0}, 2.0), 6.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({4.0, 1.0, 1.0}, 0.0), 4.0);
    EXPECT_NEAR(von_mises_2d({5.0, 2.0, 3.0}, 0.0), std::sqrt(46.0), 1e-12);
}

// ESEMPIO 230. Uno stato idrostatico non ha von Mises. Con le tre normali uguali, (p, p, 0) e szz = p, le tre
// differenze sono 0 e la von Mises e' 0 ESATTAMENTE (non "circa": e' una somma di quadrati di zeri). E' proprio quello
// che la forma sviluppata della formula, sxx^2 + syy^2 + szz^2 - sxx syy - syy szz - szz sxx + 3 sxy^2, non garantisce:
// per p = 0.7, 1.3, 3.3, 9.1 o -1234.5678 gli arrotondamenti danno un numero NEGATIVO piccolissimo (-1e-16, -4e-16,
// -4e-15, -3e-14, -5e-10) e la sua radice e' NaN. Con szz = 0 invece lo stesso (p, p, 0) non e' idrostatico (le
// differenze sono 0, p e -p) e vale |p|.
TEST(VonMises2d, AHydrostaticStateHasNoVonMisesStress) {
    for (double p : {0.7, 1.3, 3.3, 9.1, -1234.5678, -7.0, 5.5}) {
        EXPECT_EQ(von_mises_2d({p, p, 0.0}, p), 0.0) << "p = " << p;
        EXPECT_DOUBLE_EQ(von_mises_2d({p, p, 0.0}, 0.0), std::fabs(p)) << "p = " << p << ", szz = 0";
    }
}

// ESEMPIO 231. Cosa non cambia la von Mises. Si parte dall'esempio a mano, (5, 2, 3) con szz = 2, che vale 6:
//   - la stessa pressione p aggiunta a sxx, syy e szz (le differenze restano 3, 0, -3): 6, per p = -10, 0.5, 100, 1e6
//     e 0.1;
//   - il segno cambiato a tutte le tensioni (le differenze cambiano segno, i quadrati no): 6;
//   - sxx e syy scambiate, o il taglio cambiato di segno: 6;
//   - tutte le tensioni moltiplicate per un numero f: |f| x 6, cioe' 3, 18 e 12 per f = 0.5, 3 e -2.
TEST(VonMises2d, OnlyTheDistortionCounts) {
    EXPECT_DOUBLE_EQ(von_mises_2d({5.0, 2.0, 3.0}, 2.0), 6.0);

    for (double p : {-10.0, 0.5, 100.0, 1.0e6, 0.1}) {
        EXPECT_NEAR(von_mises_2d({5.0 + p, 2.0 + p, 3.0}, 2.0 + p), 6.0, 1e-9) << "pressione " << p;
    }
    EXPECT_NEAR(von_mises_2d({-5.0, -2.0, -3.0}, -2.0), 6.0, 1e-12);
    EXPECT_NEAR(von_mises_2d({2.0, 5.0, 3.0}, 2.0), 6.0, 1e-12);
    EXPECT_NEAR(von_mises_2d({5.0, 2.0, -3.0}, 2.0), 6.0, 1e-12);
    for (double f : {0.5, 3.0, -2.0}) {
        EXPECT_NEAR(von_mises_2d({5.0 * f, 2.0 * f, 3.0 * f}, 2.0 * f), std::fabs(f) * 6.0, 1e-12) << "f = " << f;
    }
}

// ESEMPIO 232. La von Mises non dipende dagli assi scelti nel piano. Ruotando x e y di un angolo a (c = cos a, s = sin
// a) le tensioni nel piano diventano
//     sxx' = c^2 sxx + s^2 syy + 2 c s sxy,      syy' = s^2 sxx + c^2 syy - 2 c s sxy,
//     sxy' = (syy - sxx) c s + (c^2 - s^2) sxy,
// mentre szz non cambia. Lo stato e' lo stesso, visto da assi diversi, quindi la von Mises deve dare lo stesso 6 per
// l'esempio a mano (5, 2, 3) con szz = 2. A 90 gradi sxx' = 2, syy' = 5 e sxy' = -3; a 30 gradi i numeri sono tutti
// diversi (6.85, 0.15 e 0.20) e la von Mises e' ancora 6. E' la prova del 3 davanti a sxy^2: con un altro coefficiente
// ruotando di 45 gradi (dove taglio e differenza fra le normali si mescolano) il risultato cambierebbe.
TEST(VonMises2d, TheAxesOfThePlaneDoNotMatter) {
    const double sxx = 5.0;
    const double syy = 2.0;
    const double sxy = 3.0;
    const double szz = 2.0;
    for (double degrees : {0.0, 30.0, 45.0, 60.0, 90.0, 123.0}) {
        const double angle = degrees * kPi / 180.0;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const Vector3 rotated = {c * c * sxx + s * s * syy + 2.0 * c * s * sxy,
                                 s * s * sxx + c * c * syy - 2.0 * c * s * sxy,
                                 (syy - sxx) * c * s + (c * c - s * s) * sxy};
        EXPECT_NEAR(von_mises_2d(rotated, szz), 6.0, 1e-12) << degrees << " gradi";
        if (degrees == 30.0) {
            EXPECT_GT(std::fabs(rotated[0] - sxx), 1.0) << "a 30 gradi le tensioni devono essere cambiate";
            EXPECT_GT(std::fabs(rotated[2] - sxy), 1.0) << "a 30 gradi le tensioni devono essere cambiate";
        }
    }
}

// ESEMPIO 233. Ogni triangolo usa le sue tensioni e i suoi tre nodi. Il quadrato unitario con i triangoli {0, 1, 2} e
// {0, 2, 3}, E = 100, nu = 1/4, alpha = 1/100 (E alpha = 1), T_ref = 20, e temperature tutte diverse nei quattro nodi,
// T = (19.5, 19.75, 20, 24.25).
//   - Triangolo 0, nodi 0, 1, 2: dT = (19.5 + 19.75 + 20) / 3 - 20 = 19.75 - 20 = -1/4. Con sigma = (5, 2, 3):
//     szz = nu (5 + 2) - E alpha dT = 7/4 + 1/4 = 2, e' l'ESEMPIO 229 e la von Mises e' 6.
//   - Triangolo 1, nodi 0, 2, 3: dT = (19.5 + 20 + 24.25) / 3 - 20 = 21.25 - 20 = 5/4. Con sigma = (4, 1, 1):
//     szz = nu (4 + 1) - E alpha dT = 5/4 - 5/4 = 0, e la von Mises e' 4.
// Chi usasse la temperatura di un nodo solo, la media di due, la T_ref sbagliata o i nodi di un altro triangolo
// troverebbe un szz diverso e un altro numero.
TEST(VonMises2d, TheFieldUsesTheStressAndTheNodesOfEachTriangle) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<double> T = {19.5, 19.75, 20.0, 24.25};

    const std::vector<double> vm = von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0);
    ASSERT_EQ(vm.size(), 2u);
    EXPECT_NEAR(vm[0], 6.0, 1e-12);
    EXPECT_NEAR(vm[1], 4.0, 1e-12);
}

// ESEMPIO 234. La lastra libera che si scalda in modo uniforme: von Mises = E alpha dT. E' la lastra dell'ESEMPIO 216
// (3 x 2 celle di [0, 2] x [0, 1], isolata, T0 = 25, T_ref = 20, un grado in piu' a ogni passo) con theta = 1 e
// 1/2. Nel piano non ci sono tensioni (si controlla), quindi sxx = syy = sxy = 0 e
//     szz = nu x 0 - E alpha dT = -dT,        sigma_vm = sqrt(1/2 (0 + dT^2 + dT^2)) = dT = 5 + p
// dopo p passi, in ogni triangolo: 5, 6, 7, 8, 9. Chi prendesse szz = 0 (la formula "da lastra sottile") troverebbe 0:
// una lastra che si scalda di 9 gradi e non e' sollecitata in nessun modo.
TEST(VonMises2d, AFreePlateThatIsHeatedUniformlyIsStressedAcrossThePlane) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    for (double theta : {1.0, 0.5}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(12, 25.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        problem.fixed_dofs = {0, 1, 7};
        problem.fixed_displacements = {0.0, 0.0, 0.0};

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", passo " + std::to_string(p);
            EXPECT_LT(max_abs(states[p].sigma), 1e-8) << label << ": nel piano non ci sono tensioni";
            expect_all_near(von_mises_of(mesh, problem, states[p]), 5.0 + static_cast<double>(p), 1e-7, label);
        }
    }
}

// ESEMPIO 235. La lastra incastrata che si scalda e' compressa in modo idrostatico: von Mises = 0. E' la lastra
// dell'ESEMPIO 218 (4 x 4 celle di [0, 2] x [0, 1], tutto il bordo bloccato, isolata, T0 = T_ref = 20, un grado in piu'
// a ogni passo) con theta = 1 e 1/2. Le tensioni nel piano sono grandi, sxx = syy = -2 dT, ma szz = nu (sxx + syy) -
// E alpha dT = 0.25 (-4 dT) - dT = -2 dT: le tre tensioni normali sono uguali e il taglio e' 0, la lastra e' compressa
// allo stesso modo in tutte le direzioni e la von Mises e' 0 (a meno degli errori del gradiente coniugato). Si
// controlla che le tensioni nel piano ci siano davvero (dal passo 1) e che la formula "da lastra sottile" (szz = 0)
// darebbe 2 dT: la lastra sembrerebbe a rischio quando non lo e'.
TEST(VonMises2d, AClampedPlateIsPressedFromAllSidesSoVonMisesIsZero) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 4, 4);
    for (double theta : {1.0, 0.5}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(25, 20.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        for (std::size_t node : boundary_nodes(mesh)) {
            problem.fixed_dofs.push_back(2 * node);
            problem.fixed_dofs.push_back(2 * node + 1);
            problem.fixed_displacements.push_back(0.0);
            problem.fixed_displacements.push_back(0.0);
        }

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", passo " + std::to_string(p);
            const double dT = static_cast<double>(p);
            expect_all_near(von_mises_of(mesh, problem, states[p]), 0.0, 1e-7, label);
            if (p > 0) {
                EXPECT_GT(max_abs(states[p].sigma), 1.9 * dT) << label << ": tensioni nel piano da trovare";
            }
            for (const Vector3& s : states[p].sigma) {
                EXPECT_NEAR(von_mises_2d(s, 0.0), 2.0 * dT, 1e-7) << label << ": la formula con szz = 0";
            }
        }
    }
}

// ESEMPIO 236. A meta' di una striscia lunga ogni triangolo ha la stessa von Mises. La striscia dell'ESEMPIO 221 (24
// colonne, celle quadrate di lato 1/2, Eulero all'indietro, un passo, E = 100, nu = 1/4, alpha = 1/100, righe di nodi
// a dT = 0, 60, 0). Nelle tre colonne di mezzo i quattro triangoli hanno le sigma dell'ESEMPIO 221 e, per quelli della
// cella in basso (riga 0), con sigma_zz = nu (sxx + syy) - E alpha dT_X:
//   - triangolo in basso, sigma = (-20, +20), dT_X = 20: szz = 0.25 x 0 - 20 = -20 (uguale a sxx).
//     sigma_vm = sqrt(1/2 [(-20 - 20)^2 + (20 + 20)^2 + (-20 + 20)^2]) = sqrt(1/2 (1600 + 1600)) = 40;
//   - triangolo in alto, sigma = (-60, -20), dT_X = 40: szz = 0.25 x (-80) - 40 = -60 (uguale a sxx).
//     sigma_vm = sqrt(1/2 [(-60 + 20)^2 + (-20 + 60)^2 + 0]) = 40.
// La riga 1 e' lo specchio: 40 in tutti e quattro. E' la formula del libro, E alpha dT / (1 - nu) = 30 / 0.75 = 40 con
// dT = 30 (la media di 0 e 60 della cella). Dopo un passo (g = 2/3) tutto e' moltiplicato per 2/3. Senza szz (con
// szz = 0) verrebbe sqrt(1200) = 34.6 nei triangoli in basso e sqrt(2800) = 52.9 in quelli in alto.
TEST(VonMises2d, InTheMiddleOfALongStripEveryTriangleHasTheSameVonMises) {
    const std::size_t nx = 24;
    const Mesh2D mesh = rectangle_mesh(0.5 * static_cast<double>(nx), 1.0, nx, 2);
    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, 2, 20.0, 60.0);
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 3.0;
    problem.dt = 1.0 / 36.0;
    problem.theta = 1.0;
    problem.n_steps = 1;
    problem.save_every = 1;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 2u);
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double factor = (p == 0) ? 1.0 : 2.0 / 3.0;
        const std::vector<double> vm = von_mises_of(mesh, problem, states[p]);
        for (std::size_t i = nx / 2 - 1; i <= nx / 2 + 1; ++i) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t half = 0; half < 2; ++half) {
                    EXPECT_NEAR(vm[2 * (row * nx + i) + half], factor * 40.0, 1e-3)
                        << "passo " << p << ", cella (" << i << ", " << row << "), triangolo " << half;
                }
            }
        }
    }
}

// ESEMPIO 237. Una lastra d'acciaio che si raffredda ha la von Mises "del libro". La lastra dell'ESEMPIO 223 (spessa
// H = 0.1 m e lunga 0.4 m, E = 210 GPa, nu = 0.3, alpha = 12e-6, rho_c = 7850 x 470, k = 45, le facce a T_ref =
// 293.15 K, partenza T_ref + 100 sin(pi y / H), 8 x 32 celle, passi di 5 s con Crank-Nicolson, stati a 0, 50 e 100 s).
// A meta' lunghezza, con syy che in media e' 0 (equilibrio verticale) e szz = sxx, la von Mises di ogni triangolo di
// una cella e' |sxx| della media dell'ESEMPIO 223, cioe'
//     sigma_vm = E alpha dT / (1 - nu),        dT = media dello scarto di T delle due righe di nodi della cella
// (100 K di scarto danno 360 MPa). Si controlla:
//   - tutti i numeri della lastra sono finiti e non negativi (una von Mises e' una radice di una somma di quadrati);
//   - i due triangoli di ogni cella di mezzo hanno la formula con lo scarto di T calcolato a meno di 1e-4 volte
//     360 MPa (l'effetto dei rulli a meta' lunghezza e' sotto 1e-5);
//   - con lo scarto "continuo", 100 sin(pi y / H) e^(-t / tau) preso a meta' altezza della cella (tau = 83 s), la
//     differenza e' l'errore del metodo (1.9% la media di due nodi, 1.3% il decadimento della mesh): sotto il 5% di
//     360 MPa;
//   - a meta' altezza, al tempo 0, la von Mises e' fra il 95% e il 100% di 360 MPa.
TEST(VonMises2d, ASteelSlabHasTheTextbookVonMises) {
    const std::size_t nx = 32;
    const std::size_t ny = 8;
    const double height = 0.1;
    const double length = 0.4;
    const double t_ref = 293.15;
    const double amplitude = 100.0;
    const Mesh2D mesh = rectangle_mesh(length, height, nx, ny);

    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, ny, t_ref, amplitude);
    problem.rho_c = 7850.0 * 470.0;
    problem.k = 45.0;
    problem.dt = 5.0;
    problem.n_steps = 20;
    problem.save_every = 10;
    problem.theta = 0.5;
    problem.E = 210.0e9;
    problem.nu = 0.3;
    problem.alpha = 12.0e-6;
    problem.T_ref = t_ref;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 3u);

    const double scale = problem.E * problem.alpha * amplitude / (1.0 - problem.nu);
    const double tau = height * height / (kPi * kPi * (problem.k / problem.rho_c));
    const std::size_t column = nx / 2;
    for (const ThermoMechanicalState2D& state : states) {
        const std::vector<double> vm = von_mises_of(mesh, problem, state);
        ASSERT_EQ(vm.size(), mesh.num_triangles());
        for (std::size_t e = 0; e < vm.size(); ++e) {
            EXPECT_TRUE(std::isfinite(vm[e])) << "t = " << state.time << ", triangolo " << e;
            EXPECT_GE(vm[e], 0.0) << "t = " << state.time << ", triangolo " << e;
        }
        for (std::size_t row = 0; row < ny; ++row) {
            const std::size_t cell = row * nx + column;
            const double below = state.T[row * (nx + 1) + column] - t_ref;
            const double above = state.T[(row + 1) * (nx + 1) + column] - t_ref;
            const double from_the_formula = problem.E * problem.alpha * 0.5 * (below + above) / (1.0 - problem.nu);
            const double y_middle = (static_cast<double>(row) + 0.5) * height / static_cast<double>(ny);
            const double continuum = scale * std::sin(kPi * y_middle / height) * std::exp(-state.time / tau);
            for (std::size_t half = 0; half < 2; ++half) {
                EXPECT_NEAR(vm[2 * cell + half], from_the_formula, 1e-4 * scale)
                    << "t = " << state.time << ", riga di celle " << row << ", triangolo " << half
                    << ", formula con la T calcolata";
                EXPECT_NEAR(vm[2 * cell + half], continuum, 0.05 * scale)
                    << "t = " << state.time << ", riga di celle " << row << ", triangolo " << half
                    << ", soluzione continua";
            }
        }
    }

    const std::vector<double> vm_start = von_mises_of(mesh, problem, states[0]);
    const std::size_t middle_cell = (ny / 2) * nx + column;
    EXPECT_GT(vm_start[2 * middle_cell], 0.95 * scale);
    EXPECT_LT(vm_start[2 * middle_cell], 1.0 * scale);
}

// ESEMPIO 238. Leggi di scala: la mesh irregolare e i dati dell'ESEMPIO 214 con i vincoli a zero. Nello stato di ogni
// istante, rispetto ai numeri del problema base:
//   - E doppio (sigma e E alpha raddoppiano, quindi anche szz): la von Mises e' il doppio;
//   - alpha doppio (u, sigma e E alpha dT raddoppiano): il doppio;
//   - il calore doppio (scarto di T iniziale e imposto e carico per 2): il doppio;
//   - il calore cambiato di segno (fattore -1): la STESSA von Mises. Tutte le tensioni cambiano segno (szz compresa) e
//     una von Mises e' fatta di differenze al quadrato: un pezzo che si scalda di 10 gradi o si raffredda di 10 e'
//     sollecitato allo stesso modo. Chi mettesse un valore assoluto nel posto sbagliato, o prendesse la radice di un
//     numero con segno, lo troverebbe qui;
//   - T e T_ref aumentate insieme di 100 gradi: la stessa von Mises (dipende solo da T - T_ref).
TEST(VonMises2d, VonMisesScalesWithEWithAlphaAndWithTheHeat) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);
    std::vector<std::vector<double>> base_vm;
    for (const ThermoMechanicalState2D& state : base_states) {
        base_vm.push_back(von_mises_of(mesh, base, state));
    }
    ASSERT_GT(max_abs(base_vm[3]), 1e-3) << "il test deve avere delle tensioni da confrontare";

    ThermoMechanicalProblem2D stiff = base;
    stiff.E = 2.0 * base.E;
    expect_von_mises_scaled(mesh, stiff, base_vm, 2.0, "E doppio");

    ThermoMechanicalProblem2D expanding = base;
    expanding.alpha = 2.0 * base.alpha;
    expect_von_mises_scaled(mesh, expanding, base_vm, 2.0, "alpha doppio");

    expect_von_mises_scaled(mesh, with_heat_times(base, 2.0), base_vm, 2.0, "calore doppio");
    expect_von_mises_scaled(mesh, with_heat_times(base, -1.0), base_vm, 1.0, "calore cambiato di segno");

    ThermoMechanicalProblem2D shifted = base;
    shifted.T_ref += 100.0;
    for (double& value : shifted.T0) {
        value += 100.0;
    }
    for (double& value : shifted.fixed_temperatures) {
        value += 100.0;
    }
    expect_von_mises_scaled(mesh, shifted, base_vm, 1.0, "T e T_ref + 100");
}

// ESEMPIO 239. Dati sbagliati: la funzione lancia std::invalid_argument. Il quadrato unitario con due triangoli, due
// tensioni, quattro temperature, E = 100, nu = 1/4 e alpha = 1/100 (si controlla che i dati siano validi). Si cambia un
// dato alla volta:
//   - sigma con 1 o con 3 terne invece di 2 (una per triangolo);
//   - T con 3 o con 5 numeri invece di 4 (uno per nodo);
//   - E = 0, E = -5 e E = NaN (serve E > 0, e con un NaN il confronto "E <= 0" non basta: "!(E > 0)" si);
//   - nu = 1/2, -1, 0.7, -1.5 e NaN (serve -1 < nu < 1/2, estremi esclusi, come per plane_strain_matrix).
// Gli estremi vicini ma dentro (nu = 0.49 e nu = -0.99) vanno bene.
TEST(VonMises2d, WrongDataIsReported) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<Vector3> one_sigma(1, Vector3{5.0, 2.0, 3.0});
    const std::vector<Vector3> three_sigma(3, Vector3{5.0, 2.0, 3.0});
    const std::vector<double> T(4, 20.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ASSERT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0));

    EXPECT_THROW(von_mises_field_2d(mesh, one_sigma, 100.0, 0.25, 0.01, T, 20.0), std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, three_sigma, 100.0, 0.25, 0.01, T, 20.0), std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, std::vector<double>(3, 20.0), 20.0),
                 std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, std::vector<double>(5, 20.0), 20.0),
                 std::invalid_argument);
    for (double E : {0.0, -5.0, nan}) {
        EXPECT_THROW(von_mises_field_2d(mesh, sigma, E, 0.25, 0.01, T, 20.0), std::invalid_argument) << "E = " << E;
    }
    for (double nu : {0.5, -1.0, 0.7, -1.5, nan}) {
        EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, nu, 0.01, T, 20.0), std::invalid_argument)
            << "nu = " << nu;
    }
    EXPECT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.49, 0.01, T, 20.0));
    EXPECT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, -0.99, 0.01, T, 20.0));
}

// ESEMPIO 240. Un NaN nei dati non diventa un numero. Se una tensione e' NaN (per esempio perche' il calcolo prima e'
// andato male) la von Mises deve essere NaN: un 0, o un altro numero qualunque, sembrerebbe "nessuna sollecitazione" e
// nasconderebbe il guasto. Vale per ognuna delle quattro tensioni. (Un modo di nasconderlo: proteggere la radice con
// std::max(0.0, ...) o con "se > 0": con un NaN il confronto da' falso e il risultato diventa 0.) Nel campo, una
// temperatura NaN in un nodo da' NaN nei triangoli che hanno quel nodo e solo in quelli: nel quadrato unitario con i
// triangoli {0, 1, 2} e {0, 2, 3} e il nodo 3 a NaN, il triangolo 0 ha la sua von Mises (6, come nell'ESEMPIO 233 con
// T = (19.5, 19.75, 20)) e il triangolo 1 e' NaN.
TEST(VonMises2d, ANanIsNeverHiddenAsANumber) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(von_mises_2d({nan, 2.0, 3.0}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, nan, 3.0}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, 2.0, nan}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, 2.0, 3.0}, nan)));

    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<double> T = {19.5, 19.75, 20.0, nan};
    const std::vector<double> vm = von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0);
    ASSERT_EQ(vm.size(), 2u);
    EXPECT_NEAR(vm[0], 6.0, 1e-12);
    EXPECT_TRUE(std::isnan(vm[1]));
}