#include <gtest/gtest.h>

#include <vector>

#include "Fem1D.h"

TEST(Fem1d, UniformMeshHasTheRightNodes) {
    std::vector<double> x = uniform_mesh(2.0, 4);   // 4 elementi tra 0 e 2
    ASSERT_EQ(x.size(), 5u);                         // quindi 5 nodi
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i], 0.5 * i, 1e-14) << "nodo " << i;
    }
}

// Due elementi di lunghezza 1 e 2: nodi 0, 1, 3. Conducibilita k = 2.
TEST(Fem1d, StiffnessMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);

    DMatrix expected = {{ 2.0, -2.0,  0.0},
                        {-2.0,  3.0, -1.0},
                        { 0.0, -1.0,  1.0}};
    ASSERT_EQ(K.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(K[i].size(), 3u);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(K[i][j], expected[i][j], 1e-14) << "K[" << i << "][" << j << "]";
        }
    }
}

// Stessi nodi, sorgente q = 4.
TEST(Fem1d, LoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector F = assemble_load(x, 4.0);

    DVector expected = {2.0, 6.0, 4.0};
    ASSERT_EQ(F.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(F[i], expected[i], 1e-14) << "F[" << i << "]";
    }
}