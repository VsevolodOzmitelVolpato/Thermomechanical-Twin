#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "Elasticity2D.h"

// ---- The material law in plane strain ------------------------------------------------------------------------
//
// From here on the plate is no longer just hot: it DEFORMS. A point (x, y) moves by (u, v) and the material
// responds with a stress. In this step there are only the two material formulas (plane_strain_matrix and
// plane_strain_thermal_strain): the mesh is not there yet.
//
// PLANE STRAIN. The plate is the cross-section of a very long body in z, which cannot stretch in z:
// epsilon_zz = 0. (For a thin, free plate a different hypothesis is used, plane stress: it is not our case.)
// The numbers that matter in the plane are 3, in the order (xx, yy, xy):
//
//     epsilon = (du/dx, dv/dy, du/dy + dv/dx),        sigma = (sigma_xx, sigma_yy, sigma_xy).
//
// The last one, gamma_xy = du/dy + dv/dx, is the "engineering" shear strain: twice the true shear strain.
// That is why in the matrix D the shear term is mu and not 2 mu.
//
// HEAT. A material with expansion coefficient alpha, heated by dT = T - T_ref, would like to expand by
// alpha dT in every direction. The stress arises only from the part of the strain that is NOT due to heat:
//
//     sigma = D (epsilon - epsilon_th).
//
// WHERE D AND epsilon_th COME FROM. Hooke's law in 3D with thermal expansion is
//
//     eps_xx = [sigma_xx - nu (sigma_yy + sigma_zz)] / E + alpha dT    (the other two are the same, with the indices swapped),
//
// with the shear strain  gamma_xy = 2 (1 + nu) sigma_xy / E. In plane strain eps_zz = 0, and the third equation gives
//
//     sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT.
//
// Substituting sigma_zz into the other two, in the plane what remains is
//
//     eps_xx = [(1 - nu^2) sigma_xx - nu (1 + nu) sigma_yy] / E + (1 + nu) alpha dT          (analogous for eps_yy).
//
// It reads as follows:
//   - the thermal expansion that applies in the plane is epsilon_th = (1 + nu) alpha dT (1, 1, 0), NOT alpha dT: the body is
//     held in z, and the expansion that cannot happen in z is released into the plane (EXAMPLES 122 and 123);
//   - the elastic part is epsilon_el = C sigma with the compliance matrix
//         C = 1 / E * [[1 - nu^2, -nu (1 + nu), 0], [-nu (1 + nu), 1 - nu^2, 0], [0, 0, 2 (1 + nu)]],
//     and D is its inverse (EXAMPLE 120). It is inverted by hand with the Lame coefficients
//         lambda = E nu / ((1 + nu) (1 - 2 nu)),        mu = E / (2 (1 + nu)),
//     and the result is  D = [[lambda + 2 mu, lambda, 0], [lambda, lambda + 2 mu, 0], [0, 0, mu]].
//
// VALIDITY. E > 0 and -1 < nu < 1/2. With nu = 1/2 (incompressible material) lambda is infinite.

namespace {

// The compliance matrix C of plane strain, the one in the comment above: derived from Hooke's law in 3D
// without going through lambda and mu. It is used by EXAMPLE 120 as a check along a different route.
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

// EXAMPLE 117. E = 1 and nu = 1/4, by hand. The Lame coefficients:
//     lambda = E nu / ((1 + nu) (1 - 2 nu)) = 0.25 / (1.25 * 0.5) = 0.25 / 0.625 = 0.4
//     mu     = E / (2 (1 + nu))             = 1 / 2.5 = 0.4
// (with nu = 1/4 they coincide). So lambda + 2 mu = 1.2 and
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

// EXAMPLE 118. Without the Poisson effect (nu = 0) we get lambda = 0 and mu = E / 2, so D = diag(E, E, E / 2): each
// direction responds on its own with stiffness E, and the shear strain with the shear modulus E / 2. With E = 3 it is
// diag(3, 3, 1.5). Then D is proportional to E: with nu = 0.3, E = 2.5 gives 2.5 times the numbers of E = 1.
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

// EXAMPLE 119. A steel: E = 210e9 Pa, nu = 0.3. With the Lame coefficients
//     lambda = 210e9 * 0.3 / (1.3 * 0.4) = 63e9 / 0.52 = 1575e9 / 13 = 121.1538...e9
//     mu     = 210e9 / 2.6                = 1050e9 / 13 =                80.7692...e9
// we get D[0][0] = lambda + 2 mu = 3675e9 / 13 = 282.6923...e9, D[0][1] = lambda and D[2][2] = mu. With large numbers
// (1e11) the comparison is made in relative terms.
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

// EXAMPLE 120. D is the inverse of the compliance matrix C derived from Hooke's law in 3D with eps_zz = 0 (see
// above): D C = I. It is a check along a route different from the formula with lambda and mu. By hand, with E = 1 and nu = 1/4:
// C = [[0.9375, -0.3125, 0], [-0.3125, 0.9375, 0], [0, 0, 2.5]] and
//     1.2 * 0.9375 + 0.4 * (-0.3125) = 1.125 - 0.125 = 1,
//     1.2 * (-0.3125) + 0.4 * 0.9375 = -0.375 + 0.375 = 0,
//     0.4 * 2.5 = 1.
// Tried for several pairs (E, nu), also with negative nu (auxetic materials) and close to the limit 1/2.
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

// EXAMPLE 121. The shape of D, for nu from -0.9 to 0.49:
//   - symmetric: D[0][1] = D[1][0];
//   - normal and shear are not coupled: D[0][2], D[2][0], D[1][2] and D[2][1] are 0 (in an isotropic material an
//     elongation produces no shear strain);
//   - positive definite: the strain energy 1/2 epsilon^T D epsilon is > 0 for every epsilon different from 0.
//     For a matrix of this form it is enough that D[2][2] > 0 and that the two upper-left minors are > 0:
//     D[0][0] > 0 and D[0][0] D[1][1] - D[0][1]^2 > 0 (Sylvester's criterion). Here D[0][0] D[1][1] - D[0][1]^2 =
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

// EXAMPLE 122. Thermal expansion in plane strain, by hand: nu = 1/4, alpha = 1e-5, dT = 50. The material
// would like to expand by alpha dT = 5e-4; in the plane, held in z, it becomes (1 + nu) alpha dT = 1.25 * 5e-4 = 6.25e-4,
// equal in x and in y, with no shear strain: epsilon_th = (6.25e-4, 6.25e-4, 0). On cooling (dT = -50) the sign
// changes; with dT = 0 or alpha = 0 everything is 0; it is linear in alpha and in dT.
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

// EXAMPLE 123. The physical check: a plate blocked on all sides. If (eps_xx, eps_yy, gamma_xy) = 0 the
// thermal expansion cannot show up and the stress is sigma = D (0 - epsilon_th) = -D epsilon_th. The body is
// blocked also in z (plane strain), so in all 3 directions, and the pressure is hydrostatic:
//     sigma_xx = sigma_yy = sigma_zz = -E alpha dT / (1 - 2 nu).
// It follows from the 3D law, without going through D or the factor (1 + nu): with eps = 0 in every direction and the three
// stresses equal to s, 0 = s (1 - 2 nu) / E + alpha dT.
// By hand for E = 1, nu = 1/4, alpha = 1e-5, dT = 100: epsilon_th = 1.25e-3 and
//     sigma_xx = -(1.2 + 0.4) * 1.25e-3 = -2e-3,            -E alpha dT / (1 - 2 nu) = -1e-3 / 0.5 = -2e-3,
//     sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT = 0.25 * (-4e-3) - 1e-3 = -2e-3:  equal to the other two.
// Without the factor (1 + nu) the stress would come out as -1.6e-3, which is wrong. Other cases: a steel heated by 80 K
// (E = 210e9, nu = 0.3, alpha = 12e-6: sigma = -504 MPa), a cooling (dT < 0: tension) and a negative nu.
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

// EXAMPLE 124. Invalid data: E zero, negative or NaN; nu = 1/2 (incompressible), above 1/2, equal to -1, below -1
// or NaN: std::invalid_argument. Valid instead: nu = 0, negative nu (auxetic materials) and nu close to the limits. For
// thermal expansion only the checks on nu matter: alpha and dT may have any sign.
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
