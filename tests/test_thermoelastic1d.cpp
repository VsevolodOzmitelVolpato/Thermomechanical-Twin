#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Thermoelastic1D.h"

// Funzione di appoggio: confronta due vettori numero per numero e, se qualcosa non torna,
// dice quale elemento e' sbagliato.
static void expect_vector_near(const DVector& a, const DVector& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "f[" << i << "]";
    }
}

// ---- Forze nodali equivalenti alla dilatazione termica --------------------------------

// Un solo elemento, lungo 2. E = 3, alpha = 5. La temperatura e' 1 grado sopra T_ref = 0 in
// entrambi i nodi, quindi dT = 1 e la forza vale E alpha dT = 15.
// Il calore spinge il nodo sinistro verso sinistra (-15) e il destro verso destra (+15).
TEST(Thermoelastic1d, ThermalLoadOfOneElement) {
    std::vector<double> x = {0.0, 2.0};
    DVector f = assemble_thermal_load(x, 3.0, 5.0, {1.0, 1.0}, 0.0);

    expect_vector_near(f, {-15.0, 15.0}, 1e-12);
}

// Due elementi lunghi 1 e 2 (nodi 0, 1, 3). E = 2 e alpha = 0.5, quindi E alpha = 1.
// T_ref = 10 e T = {10, 20, 40}: gli scarti da T_ref sono {0, 10, 30}.
//   elemento 0: scarto medio (0 + 10) / 2 = 5   -> forza 5:  nodo 0 riceve -5,  nodo 1 riceve +5
//   elemento 1: scarto medio (10 + 30) / 2 = 20 -> forza 20: nodo 1 riceve -20, nodo 2 riceve +20
//   il nodo 1 e' in comune: +5 - 20 = -15
// Gli elementi hanno lunghezze diverse ma nella formula non compare nessuna lunghezza.
TEST(Thermoelastic1d, ThermalLoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector f = assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0}, 10.0);

    expect_vector_near(f, {-5.0, -15.0, 20.0}, 1e-12);
}

// Se la temperatura e' ovunque uguale a T_ref la barra e' a riposo: nessuna forza.
// (T_ref e' diverso da 0 apposta: chi dimenticasse di sottrarlo troverebbe delle forze.)
TEST(Thermoelastic1d, NoTemperatureChangeMeansNoForce) {
    std::vector<double> x = {0.0, 1.0, 3.0, 4.0};
    DVector f = assemble_thermal_load(x, 200.0, 12e-6, DVector(4, 300.0), 300.0);

    expect_vector_near(f, DVector(4, 0.0), 1e-12);
}

// Il calore non spinge la barra dall'esterno: le forze sono tutte interne e la loro somma e' 0.
// Prova con una temperatura qualunque su una maglia irregolare (e si controlla anche che le
// forze ci siano: una somma di zeri non dimostrerebbe niente).
TEST(Thermoelastic1d, ThermalLoadsSumToZero) {
    std::vector<double> x = {0.0, 0.1, 0.5, 0.6, 1.5, 2.0};
    DVector T = {20.0, 35.0, 18.0, 90.0, 60.0, 25.0};
    DVector f = assemble_thermal_load(x, 70.0, 2.3e-5, T, 20.0);

    ASSERT_EQ(f.size(), 6u);
    double sum = 0.0;
    for (double fi : f) {
        sum += fi;
    }
    EXPECT_NEAR(sum, 0.0, 1e-12);
    EXPECT_GT(std::fabs(f[0]), 1e-3);
    EXPECT_GT(std::fabs(f[5]), 1e-3);
}

// Input sbagliati: T con un numero di valori diverso da quello dei nodi.
TEST(Thermoelastic1d, ThermalLoadWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};   // 3 nodi

    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0}, 10.0), std::invalid_argument);               // 2 valori
    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0, 50.0}, 10.0), std::invalid_argument);   // 4 valori
}