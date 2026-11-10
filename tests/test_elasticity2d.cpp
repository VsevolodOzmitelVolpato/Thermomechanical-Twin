#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "Elasticity2D.h"

// ---- La legge del materiale in deformazione piana ------------------------------------------------------------
//
// Da qui in poi la lastra non e' solo calda: si DEFORMA. Un punto (x, y) si sposta di (u, v) e il materiale
// risponde con una tensione. In questo passo ci sono solo le due formule del materiale (plane_strain_matrix e
// plane_strain_thermal_strain): la mesh non c'e' ancora.
//
// LA DEFORMAZIONE PIANA. La lastra e' la sezione di un corpo lunghissimo in z, che in z non si puo' allungare:
// epsilon_zz = 0. (Per una lastra sottile e libera si usa un'altra ipotesi, la tensione piana: non e' il nostro caso.)
// I numeri che contano nel piano sono 3, nell'ordine (xx, yy, xy):
//
//     epsilon = (du/dx, dv/dy, du/dy + dv/dx),        sigma = (sigma_xx, sigma_yy, sigma_xy).
//
// L'ultimo, gamma_xy = du/dy + dv/dx, e' lo scorrimento "ingegneristico": il doppio della deformazione di taglio vera.
// Per questo nella matrice D il termine di taglio e' mu e non 2 mu.
//
// IL CALORE. Un materiale con coefficiente di dilatazione alpha, scaldato di dT = T - T_ref, vorrebbe dilatarsi di
// alpha dT in ogni direzione. La tensione nasce solo dalla parte di deformazione che NON e' dovuta al calore:
//
//     sigma = D (epsilon - epsilon_th).
//
// DA DOVE VENGONO D E epsilon_th. La legge di Hooke in 3D con la dilatazione termica e'
//
//     eps_xx = [sigma_xx - nu (sigma_yy + sigma_zz)] / E + alpha dT    (e le altre due uguali, scambiando gli indici),
//
// con lo scorrimento  gamma_xy = 2 (1 + nu) sigma_xy / E. In deformazione piana eps_zz = 0, e la terza equazione da'
//
//     sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT.
//
// Sostituendo sigma_zz nelle altre due, nel piano resta
//
//     eps_xx = [(1 - nu^2) sigma_xx - nu (1 + nu) sigma_yy] / E + (1 + nu) alpha dT          (analoga per eps_yy).
//
// Si legge cosi':
//   - la dilatazione termica che vale nel piano e' epsilon_th = (1 + nu) alpha dT (1, 1, 0), NON alpha dT: il corpo e'
//     trattenuto in z, e la dilatazione che in z non puo' avvenire si scarica nel piano (ESEMPI 122 e 123);
//   - la parte elastica e' epsilon_el = C sigma con la matrice di cedevolezza
//         C = 1 / E * [[1 - nu^2, -nu (1 + nu), 0], [-nu (1 + nu), 1 - nu^2, 0], [0, 0, 2 (1 + nu)]],
//     e D e' la sua inversa (ESEMPIO 120). Si inverte a mano con i coefficienti di Lame
//         lambda = E nu / ((1 + nu) (1 - 2 nu)),        mu = E / (2 (1 + nu)),
//     e viene  D = [[lambda + 2 mu, lambda, 0], [lambda, lambda + 2 mu, 0], [0, 0, mu]].
//
// VALIDITA'. E > 0 e -1 < nu < 1/2. Con nu = 1/2 (materiale incomprimibile) lambda e' infinito.

namespace {

// La matrice di cedevolezza C della deformazione piana, quella del commento sopra: ricavata dalla legge di Hooke in 3D
// senza passare da lambda e mu. Serve all'ESEMPIO 120 come controllo su un'altra strada.
Matrix3 plane_strain_compliance(double E, double nu) {
    Matrix3 C{};
    C[0][0] = (1.0 - nu * nu) / E;
    C[0][1] = -nu * (1.0 + nu) / E;
    C[1][0] = -nu * (1.0 + nu) / E;
    C[1][1] = (1.0 - nu * nu) / E;
    C[2][2] = 2.0 * (1.0 + nu) / E;
    return C;
}

}  // namespace

// ESEMPIO 117. E = 1 e nu = 1/4, a mano. I coefficienti di Lame:
//     lambda = E nu / ((1 + nu) (1 - 2 nu)) = 0.25 / (1.25 * 0.5) = 0.25 / 0.625 = 0.4
//     mu     = E / (2 (1 + nu))             = 1 / 2.5 = 0.4
// (con nu = 1/4 coincidono). Quindi lambda + 2 mu = 1.2 e
//     D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]].
TEST(PlaneStrainMatrix, MatchesHandComputationForPoissonOneQuarter) {
    const Matrix3 D = plane_strain_matrix(1.0, 0.25);
    const double want[3][3] = {{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}};

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(D[i][j], want[i][j], 1e-14) << "D[" << i << "][" << j << "]";
        }
    }
}

// ESEMPIO 118. Senza effetto Poisson (nu = 0) si ha lambda = 0 e mu = E / 2, quindi D = diag(E, E, E / 2): ogni
// direzione risponde per conto suo con rigidezza E, e lo scorrimento con il modulo di taglio E / 2. Con E = 3 e'
// diag(3, 3, 1.5). Poi D e' proporzionale a E: con nu = 0.3, E = 2.5 da' 2.5 volte i numeri di E = 1.
TEST(PlaneStrainMatrix, WithoutPoissonEffectIsDiagonalAndProportionalToE) {
    const Matrix3 D = plane_strain_matrix(3.0, 0.0);
    EXPECT_NEAR(D[0][0], 3.0, 1e-14);
    EXPECT_NEAR(D[1][1], 3.0, 1e-14);
    EXPECT_NEAR(D[2][2], 1.5, 1e-14);
    EXPECT_EQ(D[0][1], 0.0);
    EXPECT_EQ(D[1][0], 0.0);
    EXPECT_EQ(D[0][2], 0.0);
    EXPECT_EQ(D[2][0], 0.0);
    EXPECT_EQ(D[1][2], 0.0);
    EXPECT_EQ(D[2][1], 0.0);

    const Matrix3 unit = plane_strain_matrix(1.0, 0.3);
    const Matrix3 scaled = plane_strain_matrix(2.5, 0.3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(scaled[i][j], 2.5 * unit[i][j], 1e-13) << "D[" << i << "][" << j << "]";
        }
    }
}

// ESEMPIO 119. Un acciaio: E = 210e9 Pa, nu = 0.3. Con i coefficienti di Lame
//     lambda = 210e9 * 0.3 / (1.3 * 0.4) = 63e9 / 0.52 = 1575e9 / 13 = 121.1538...e9
//     mu     = 210e9 / 2.6                = 1050e9 / 13 =                80.7692...e9
// si ha D[0][0] = lambda + 2 mu = 3675e9 / 13 = 282.6923...e9, D[0][1] = lambda e D[2][2] = mu. Con numeri grandi
// (1e11) si confronta in valore relativo.
TEST(PlaneStrainMatrix, MatchesSteelValues) {
    const Matrix3 D = plane_strain_matrix(210e9, 0.3);
    const double lambda = 1575e9 / 13.0;
    const double mu = 1050e9 / 13.0;

    EXPECT_NEAR(D[0][0], lambda + 2.0 * mu, 1e-12 * 282.7e9);
    EXPECT_NEAR(D[1][1], lambda + 2.0 * mu, 1e-12 * 282.7e9);
    EXPECT_NEAR(D[0][1], lambda, 1e-12 * 121.2e9);
    EXPECT_NEAR(D[1][0], lambda, 1e-12 * 121.2e9);
    EXPECT_NEAR(D[2][2], mu, 1e-12 * 80.8e9);
}

// ESEMPIO 120. D e' l'inversa della matrice di cedevolezza C ricavata dalla legge di Hooke in 3D con eps_zz = 0 (vedi
// sopra): D C = I. E' un controllo su una strada diversa dalla formula con lambda e mu. A mano, con E = 1 e nu = 1/4:
// C = [[0.9375, -0.3125, 0], [-0.3125, 0.9375, 0], [0, 0, 2.5]] e
//     1.2 * 0.9375 + 0.4 * (-0.3125) = 1.125 - 0.125 = 1,
//     1.2 * (-0.3125) + 0.4 * 0.9375 = -0.375 + 0.375 = 0,
//     0.4 * 2.5 = 1.
// Provato per piu' coppie (E, nu), anche con nu negativo (materiali auxetici) e vicino al limite 1/2.
TEST(PlaneStrainMatrix, IsTheInverseOfTheCompliance) {
    struct Case {
        double E;
        double nu;
    };
    const Case cases[] = {{1.0, 0.25}, {210e9, 0.3}, {3.0, -0.5}, {0.7, 0.45}, {1e-3, 0.0}, {68.9e9, 0.33}};

    for (const Case& c : cases) {
        const Matrix3 D = plane_strain_matrix(c.E, c.nu);
        const Matrix3 C = plane_strain_compliance(c.E, c.nu);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                double product = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    product += D[i][k] * C[k][j];
                }
                EXPECT_NEAR(product, i == j ? 1.0 : 0.0, 1e-12)
                    << "(D C)[" << i << "][" << j << "], E = " << c.E << ", nu = " << c.nu;
            }
        }
    }
}

// ESEMPIO 121. La forma di D, per nu da -0.9 a 0.49:
//   - simmetrica: D[0][1] = D[1][0];
//   - normale e taglio non si accoppiano: D[0][2], D[2][0], D[1][2] e D[2][1] valgono 0 (in un materiale isotropo un
//     allungamento non produce scorrimenti);
//   - positiva definita: l'energia di deformazione 1/2 epsilon^T D epsilon e' > 0 per ogni epsilon diverso da 0.
//     Per una matrice di questa forma basta che D[2][2] > 0 e che i due minori in alto a sinistra siano > 0:
//     D[0][0] > 0 e D[0][0] D[1][1] - D[0][1]^2 > 0 (criterio di Sylvester). Qui D[0][0] D[1][1] - D[0][1]^2 =
//     (lambda + 2 mu)^2 - lambda^2 = 4 mu (lambda + mu).
TEST(PlaneStrainMatrix, IsSymmetricAndPositiveDefinite) {
    for (double nu : {-0.9, -0.5, 0.0, 0.25, 0.49}) {
        const Matrix3 D = plane_strain_matrix(2.0, nu);

        EXPECT_NEAR(D[0][1], D[1][0], 1e-14) << "nu = " << nu;
        EXPECT_EQ(D[0][2], 0.0) << "nu = " << nu;
        EXPECT_EQ(D[2][0], 0.0) << "nu = " << nu;
        EXPECT_EQ(D[1][2], 0.0) << "nu = " << nu;
        EXPECT_EQ(D[2][1], 0.0) << "nu = " << nu;
        EXPECT_NEAR(D[0][0], D[1][1], 1e-14) << "nu = " << nu;

        EXPECT_GT(D[0][0], 0.0) << "nu = " << nu;
        EXPECT_GT(D[0][0] * D[1][1] - D[0][1] * D[0][1], 0.0) << "nu = " << nu;
        EXPECT_GT(D[2][2], 0.0) << "nu = " << nu;
    }
}

// ESEMPIO 122. La dilatazione termica in deformazione piana, a mano: nu = 1/4, alpha = 1e-5, dT = 50. Il materiale
// vorrebbe dilatarsi di alpha dT = 5e-4; nel piano, trattenuto in z, diventa (1 + nu) alpha dT = 1.25 * 5e-4 = 6.25e-4,
// uguale in x e in y, senza scorrimento: epsilon_th = (6.25e-4, 6.25e-4, 0). Raffreddando (dT = -50) il segno
// cambia; con dT = 0 o alpha = 0 e' tutto 0; e' lineare in alpha e in dT.
TEST(PlaneStrainThermalStrain, MatchesHandComputation) {
    const Vector3 hot = plane_strain_thermal_strain(0.25, 1e-5, 50.0);
    EXPECT_NEAR(hot[0], 6.25e-4, 1e-17);
    EXPECT_NEAR(hot[1], 6.25e-4, 1e-17);
    EXPECT_EQ(hot[2], 0.0);

    const Vector3 cold = plane_strain_thermal_strain(0.25, 1e-5, -50.0);
    EXPECT_NEAR(cold[0], -6.25e-4, 1e-17);
    EXPECT_NEAR(cold[1], -6.25e-4, 1e-17);
    EXPECT_EQ(cold[2], 0.0);

    const Vector3 no_heat = plane_strain_thermal_strain(0.25, 1e-5, 0.0);
    const Vector3 no_expansion = plane_strain_thermal_strain(0.25, 0.0, 50.0);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(no_heat[i], 0.0) << "i = " << i;
        EXPECT_EQ(no_expansion[i], 0.0) << "i = " << i;
    }

    const Vector3 doubled = plane_strain_thermal_strain(0.25, 2e-5, 100.0);
    EXPECT_NEAR(doubled[0], 4.0 * hot[0], 1e-16);
    EXPECT_NEAR(doubled[1], 4.0 * hot[1], 1e-16);
}

// ESEMPIO 123. Il controllo fisico: una lastra bloccata da tutte le parti. Se (eps_xx, eps_yy, gamma_xy) = 0 la
// dilatazione termica non puo' manifestarsi e la tensione e' sigma = D (0 - epsilon_th) = -D epsilon_th. Il corpo e'
// bloccato anche in z (deformazione piana), quindi in tutte e 3 le direzioni, e la pressione e' idrostatica:
//     sigma_xx = sigma_yy = sigma_zz = -E alpha dT / (1 - 2 nu).
// Si ricava dalla legge 3D, senza passare da D ne dal fattore (1 + nu): con eps = 0 in ogni direzione e le tre
// tensioni uguali a s, 0 = s (1 - 2 nu) / E + alpha dT.
// A mano per E = 1, nu = 1/4, alpha = 1e-5, dT = 100: epsilon_th = 1.25e-3 e
//     sigma_xx = -(1.2 + 0.4) * 1.25e-3 = -2e-3,            -E alpha dT / (1 - 2 nu) = -1e-3 / 0.5 = -2e-3,
//     sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT = 0.25 * (-4e-3) - 1e-3 = -2e-3:  uguale alle altre due.
// Senza il fattore (1 + nu) la tensione verrebbe -1.6e-3, sbagliata. Altri casi: un acciaio scaldato di 80 K
// (E = 210e9, nu = 0.3, alpha = 12e-6: sigma = -504 MPa), un raffreddamento (dT < 0: trazione) e un nu negativo.
TEST(PlaneStrainThermalStrain, FullyConstrainedPlateIsUnderHydrostaticStress) {
    struct Case {
        double E;
        double nu;
        double alpha;
        double dT;
    };
    const Case cases[] = {{1.0, 0.25, 1e-5, 100.0}, {210e9, 0.3, 12e-6, 80.0}, {3.0, -0.4, 2e-5, -30.0},
                          {0.7, 0.45, 1e-4, 10.0}};

    for (const Case& c : cases) {
        const Matrix3 D = plane_strain_matrix(c.E, c.nu);
        const Vector3 eps_th = plane_strain_thermal_strain(c.nu, c.alpha, c.dT);
        Vector3 sigma{};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                sigma[i] -= D[i][j] * eps_th[j];
            }
        }
        const double sigma_zz = c.nu * (sigma[0] + sigma[1]) - c.E * c.alpha * c.dT;
        const double want = -c.E * c.alpha * c.dT / (1.0 - 2.0 * c.nu);
        const double tol = 1e-12 * std::fabs(want);

        EXPECT_NEAR(sigma[0], want, tol) << "E = " << c.E << ", nu = " << c.nu;
        EXPECT_NEAR(sigma[1], want, tol) << "E = " << c.E << ", nu = " << c.nu;
        EXPECT_NEAR(sigma[2], 0.0, tol) << "E = " << c.E << ", nu = " << c.nu;
        EXPECT_NEAR(sigma_zz, want, tol) << "E = " << c.E << ", nu = " << c.nu;
    }
}

// ESEMPIO 124. Dati non validi: E nullo, negativo o NaN; nu = 1/2 (incomprimibile), oltre 1/2, uguale a -1, sotto -1
// o NaN: std::invalid_argument. Valgono invece nu = 0, nu negativi (materiali auxetici) e nu vicino ai limiti. Per la
// dilatazione termica contano solo i controlli su nu: alpha e dT possono avere qualunque segno.
TEST(PlaneStrainMatrix, RejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(plane_strain_matrix(0.0, 0.3), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(-1.0, 0.3), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(not_a_number, 0.3), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(1.0, 0.5), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(1.0, 0.7), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(1.0, -1.5), std::invalid_argument);
    EXPECT_THROW(plane_strain_matrix(1.0, not_a_number), std::invalid_argument);

    EXPECT_NO_THROW(plane_strain_matrix(1.0, 0.0));
    EXPECT_NO_THROW(plane_strain_matrix(1.0, 0.499));
    EXPECT_NO_THROW(plane_strain_matrix(1.0, -0.999));
    EXPECT_NO_THROW(plane_strain_matrix(1e-9, 0.3));
}

TEST(PlaneStrainThermalStrain, RejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(plane_strain_thermal_strain(0.5, 1e-5, 10.0), std::invalid_argument);
    EXPECT_THROW(plane_strain_thermal_strain(0.7, 1e-5, 10.0), std::invalid_argument);
    EXPECT_THROW(plane_strain_thermal_strain(-1.0, 1e-5, 10.0), std::invalid_argument);
    EXPECT_THROW(plane_strain_thermal_strain(-1.5, 1e-5, 10.0), std::invalid_argument);
    EXPECT_THROW(plane_strain_thermal_strain(not_a_number, 1e-5, 10.0), std::invalid_argument);

    EXPECT_NO_THROW(plane_strain_thermal_strain(0.0, 1e-5, 10.0));
    EXPECT_NO_THROW(plane_strain_thermal_strain(0.499, 1e-5, 10.0));
    EXPECT_NO_THROW(plane_strain_thermal_strain(-0.999, 1e-5, 10.0));
    EXPECT_NO_THROW(plane_strain_thermal_strain(0.3, -1e-5, -50.0));
    EXPECT_NO_THROW(plane_strain_thermal_strain(0.3, 0.0, 0.0));
}