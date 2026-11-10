#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "Fem1D.h"
#include "Verification2D.h"

// ---- Il seno che decade in 2D: verifica dello schema theta ----------------------------------------------------
//
// In 1D abbiamo controllato lo schema theta con un problema di cui si conosce la soluzione esatta, il seno che decade
// (decaying_sine_error). Qui facciamo lo stesso per la lastra: decaying_sine_error_2d(n, dt, n_steps, theta).
//
// IL PROBLEMA. Il quadrato [0, 1] x [0, 1], rho_c = 1, k = 1, nessuna sorgente, T = 0 su tutto il bordo e
// temperatura iniziale T0(x, y) = sin(pi x) sin(pi y). L'equazione del calore e'  T_t = T_xx + T_yy  e per questa T0
//
//     T_xx + T_yy = -2 pi^2 T0
//
// quindi la forma resta quella del seno e solo l'ampiezza decade:
//
//     T(x, y, t) = exp(-2 pi^2 t) sin(pi x) sin(pi y)                              (2 pi^2 = 19.74...).
//
// La funzione risolve con la mesh di n x n celle e n_steps passi di ampiezza dt, e restituisce l'errore L2
// (l2_error_2d) rispetto a questa soluzione, presa nei nodi, al tempo finale t = n_steps * dt.
//
// GLI ORACOLI. In 1D il seno era un autovettore delle matrici, e a mano si trovava la soluzione ESATTA DELLO SCHEMA
// (g^n sin(pi x), amplification_factor). Sulla mesh di triangoli la matrice di massa collega ogni nodo a due vicini
// sulla diagonale (e non agli altri due), quindi il seno non e' piu' un autovettore e per n qualunque non c'e' una
// formula. Le risposte giuste vengono da:
//   - la mesh 2 x 2, che ha una sola incognita e quindi una formula chiusa (ESEMPI 109 e 110);
//   - numeri calcolati per un'altra strada, con matrici piene e senza gradiente coniugato (ESEMPIO 111);
//   - gli ordini di convergenza, nello spazio e nel tempo (ESEMPI 112, 113 e 114).
//
// DUE ERRORI CHE SI SOMMANO. Rispetto alla soluzione esatta il metodo sbaglia per due motivi: la mesh (un errore E_h
// che va come h^2 e resta anche con dt -> 0) e il passo nel tempo (un errore E_dt che va come dt^p, con p = 1 per
// Eulero all'indietro e p = 2 per Crank-Nicolson). Con un solo seno che decade i due si sommano:
//
//     errore = E_h + E_dt,        E_h non dipende da dt.
//
// Per Crank-Nicolson E_h ed E_dt hanno lo stesso segno (la temperatura calcolata sta sotto quella esatta: decade
// troppo in fretta). Per Eulero all'indietro E_dt ha il segno opposto (sta sopra: decade troppo piano).
//
// COME SI MISURA L'ORDINE NEL TEMPO. In 1D bastava una mesh molto fine (N = 400), cosi' E_h era trascurabile. In 2D una
// mesh cosi' fine costa troppo (a n = 64, senza ottimizzazioni, un passo prende circa 0.1 s) e il rapporto
// e(dt) / e(dt / 2) verrebbe sporcato da E_h. Si fanno invece tre calcoli, con dt, dt / 2 e dt / 4, e si guardano le
// DIFFERENZE fra calcoli successivi, in cui E_h si cancella:
//
//     d1 = e(dt) - e(dt / 2) = E_dt(dt) - E_dt(dt / 2) = C dt^p (1 - 2^-p)
//     d2 = e(dt / 2) - e(dt / 4)                       = C (dt / 2)^p (1 - 2^-p)            =>   d1 / d2 = 2^p.
//
// Quindi  p = observed_order(d1, d2, 2)  (la funzione dell'ordine in spazio, applicata alle differenze).
// Attenzione: e e' una norma, sempre positiva, e non ricorda il segno di E_h + E_dt. La sottrazione vale solo se quel
// segno e' lo stesso nei tre calcoli: con Eulero all'indietro serve un dt abbastanza grande da far dominare E_dt.

namespace {

// L'errore atteso sulla mesh 2 x 2 (ESEMPIO 110): |gamma^n_steps - exp(-2 pi^2 n_steps dt)| / sqrt(8), con
// gamma = (1 - 32 (1 - theta) dt) / (1 + 32 theta dt).
double two_by_two_error(double dt, std::size_t n_steps, double theta) {
    const double pi = std::acos(-1.0);
    const double steps = static_cast<double>(n_steps);
    const double gamma = (1.0 - 32.0 * (1.0 - theta) * dt) / (1.0 + 32.0 * theta * dt);
    return std::fabs(std::pow(gamma, steps) - std::exp(-2.0 * pi * pi * dt * steps)) / std::sqrt(8.0);
}

}  // namespace

// ESEMPIO 107. Con 0 passi la temperatura finale e' quella iniziale, e T0 e' gia' la soluzione esatta al tempo 0:
// l'errore e' 0 (a meno degli arrotondamenti), qualunque siano n, dt e theta.
TEST(DecayingSine2D, ZeroStepsGiveZeroError) {
    EXPECT_NEAR(decaying_sine_error_2d(4, 0.01, 0, 0.5), 0.0, 1e-14);
    EXPECT_NEAR(decaying_sine_error_2d(9, 0.3, 0, 1.0), 0.0, 1e-14);
    EXPECT_NEAR(decaying_sine_error_2d(2, 1.0, 0, 0.0), 0.0, 1e-14);
}

// ESEMPIO 108. n = 1: una sola cella, 4 nodi tutti sul bordo, nessuna incognita. La temperatura e' 0 in tutti i nodi a
// ogni passo, e la soluzione esatta nei 4 nodi e' 0 (sin(pi) in un computer vale 1.2e-16, un arrotondamento): errore 0
// qualunque siano dt, passi e theta. E' il caso piu' piccolo accettato: n = 1 e' valido, n = 0 no (ESEMPIO 115).
TEST(DecayingSine2D, SingleCellMeshHasNothingToSolve) {
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.01, 3, 0.5), 0.0, 1e-12);
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.5, 1, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.001, 10, 0.0), 0.0, 1e-12);
}

// ESEMPIO 109. n = 2, Eulero all'indietro (theta = 1), UN passo di dt = 0.01, a mano. Come nell'ESEMPIO 92 i nodi sono
// 9: 8 sul bordo (T = 0) e uno solo al centro c = (1/2, 1/2), quindi c'e' una sola incognita, T_c.
//   - Dall'ESEMPIO 92: K[c][c] = 4 k = 4. Il centro sta in 6 triangoli di area 1/8 e in ognuno la massa M[c][c] vale
//     rho_c A / 6: M[c][c] = 6 * (1/8) / 6 = 1/8.
//   - Gli altri nodi hanno T = 0 sempre, e non c'e' sorgente: la riga del centro di (M + dt K) T_new = M T_old e'
//         (1/8 + 4 dt) T_c,new = 1/8 T_c,old.
//   - Nel centro T0 = sin(pi/2) sin(pi/2) = 1, quindi T_c,new = (1/8) / (1/8 + 0.04) = 0.125 / 0.165 = 0.75757...
//   - La soluzione esatta nel centro: exp(-2 pi^2 0.01) = exp(-0.19739...) = 0.820868...
//   - L'errore nel centro e' e_c = 0.757575... - 0.820868... = -0.063292...; negli altri 8 nodi e' (quasi) 0: T = 0 e
//     la soluzione esatta vale al massimo 1e-16. Quindi
//         || e ||_L2 = sqrt(e^T M e) = |e_c| sqrt(M[c][c]) = 0.063292... / sqrt(8) = 0.022377...
TEST(DecayingSine2D, TwoByTwoBackwardEulerMatchesHandComputation) {
    EXPECT_NEAR(decaying_sine_error_2d(2, 0.01, 1, 1.0), 2.237744055204e-02, 1e-11);
}

// ESEMPIO 110. Sempre n = 2, ma per piu' passi e per ogni theta. Un passo dello schema theta nel centro e'
//     (1/8 + theta dt 4) T_new = (1/8 - (1 - theta) dt 4) T_old,
// cioe' T_new = gamma T_old con
//     gamma = (1 - 32 (1 - theta) dt) / (1 + 32 theta dt)        (si moltiplica sopra e sotto per 8).
// Dopo n_steps passi T_c = gamma^n_steps (T_c = 1 all'inizio) e l'errore e'
//     | gamma^n_steps - exp(-2 pi^2 n_steps dt) | / sqrt(8).
// I casi (gli errori attesi, con la formula, sono scritti accanto):
//   dt = 0.01,  2 passi, theta = 1/2:   gamma =  0.7241   errore 5.2838e-02
//   dt = 0.005, 4 passi, theta = 1:     gamma =  0.8621   errore 4.2969e-02
//   dt = 0.002, 5 passi, theta = 0:     gamma =  0.9360   errore 3.6221e-02   (Eulero in avanti, stabile: dt < 1/16)
//   dt = 0.01,  3 passi, theta = 0.3:   gamma =  0.7080   errore 7.0068e-02
//   dt = 0.2,   3 passi, theta = 1/2:   gamma = -0.5238   errore 5.0816e-02
// L'ultimo e' un passo molto grande: Crank-Nicolson e' stabile (|gamma| < 1) ma gamma e' NEGATIVO, la temperatura
// calcolata cambia segno a ogni passo mentre quella vera decade senza oscillare. La formula lo riproduce, e il codice
// deve farlo uguale.
TEST(DecayingSine2D, TwoByTwoMatchesClosedFormForEveryScheme) {
    struct Case {
        double dt;
        std::size_t n_steps;
        double theta;
    };
    const Case cases[] = {{0.01, 2, 0.5}, {0.005, 4, 1.0}, {0.002, 5, 0.0}, {0.01, 3, 0.3}, {0.2, 3, 0.5}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(2, c.dt, c.n_steps, c.theta), two_by_two_error(c.dt, c.n_steps, c.theta),
                    1e-12)
            << "dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}

// ESEMPIO 111. Numeri calcolati per un'altra strada: la mesh, le matrici M e K, lo schema theta e l'errore L2 sono
// stati riscritti da zero in Python con matrici piene e l'eliminazione di Gauss (nessun gradiente coniugato).
//     n = 4, dt = 5e-4,  100 passi, theta = 1/2:   2.452250395721e-02
//     n = 8, dt = 0.01,    5 passi, theta = 1:     1.018063914618e-02
//     n = 8, dt = 0.001,  10 passi, theta = 0:     4.073834024549e-03
//     n = 6, dt = 0.002,  25 passi, theta = 0.3:   1.338984429158e-02
// Il C++ e il Python coincidono su queste cifre a circa 1e-10 (in valore relativo): il test chiede 1e-7, perche' il
// gradiente coniugato si ferma a un residuo di 1e-12 e non da' lo stesso rumore di Gauss. Se cambiano delle cifre
// significative e' cambiato il problema o lo schema, non l'arrotondamento.
TEST(DecayingSine2D, MatchesNumbersFromADenseReference) {
    struct Case {
        std::size_t n;
        double dt;
        std::size_t n_steps;
        double theta;
        double want;
    };
    const Case cases[] = {{4, 5e-4, 100, 0.5, 2.452250395721e-02},
                          {8, 0.01, 5, 1.0, 1.018063914618e-02},
                          {8, 0.001, 10, 0.0, 4.073834024549e-03},
                          {6, 0.002, 25, 0.3, 1.338984429158e-02}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(c.n, c.dt, c.n_steps, c.theta), c.want, 1e-7 * c.want)
            << "n = " << c.n << ", dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}

// ESEMPIO 112. SPAZIO. Con elementi lineari l'errore L2 va come h^2: quando n raddoppia l'errore si divide per 4, cioe'
// l'ordine osservato e' 2. Crank-Nicolson con dt = 1e-3 e 50 passi (t = 0.05): l'errore nel tempo, circa 6e-6, e' l'1%
// di quello di mesh a n = 32 (4.5e-4) e si trascura. Misurato (errori 6.90e-3, 1.78e-3, 4.53e-4): ordine 1.95 da 8 a
// 16 e 1.97 da 16 a 32. Si parte da n = 8 perche' con una mesh piu' rada (da 4 a 8 l'ordine e' 1.83) non siamo
// ancora nell'andamento vero.
TEST(DecayingSine2D, SpatialConvergenceIsSecondOrder) {
    const double dt = 1e-3;
    const std::size_t n_steps = 50;
    double previous = decaying_sine_error_2d(8, dt, n_steps, 0.5);
    for (std::size_t n : {16u, 32u}) {
        const double current = decaying_sine_error_2d(n, dt, n_steps, 0.5);
        EXPECT_LT(current, previous) << "n = " << n / 2 << " -> " << n;
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.1) << "n = " << n / 2 << " -> " << n;
        previous = current;
    }
}

// ESEMPIO 113. TEMPO, Crank-Nicolson (theta = 1/2): ordine 2. Mesh 8 x 8, fino a t = 0.05 con 4, 8 e 16 passi. Gli
// errori sono 7.876e-3, 7.139e-3, 6.956e-3: quasi uguali, perche' E_h (circa 6.9e-3) non cala. Il rapporto diretto
// darebbe un ordine di 0.14 e 0.04. Le differenze sono d1 = 7.38e-4 e d2 = 1.83e-4: rapporto 4.03, ordine 2.01.
TEST(DecayingSine2D, TemporalConvergenceOfCrankNicolsonIsSecondOrder) {
    const double t = 0.05;
    const double e1 = decaying_sine_error_2d(8, t / 4, 4, 0.5);
    const double e2 = decaying_sine_error_2d(8, t / 8, 8, 0.5);
    const double e3 = decaying_sine_error_2d(8, t / 16, 16, 0.5);
    const double d1 = e1 - e2;
    const double d2 = e2 - e3;

    ASSERT_GT(d1, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    ASSERT_GT(d2, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    EXPECT_NEAR(observed_order(d1, d2, 2.0), 2.0, 0.1) << "e = " << e1 << ", " << e2 << ", " << e3;
}

// ESEMPIO 114. TEMPO, Eulero all'indietro (theta = 1): ordine 1. Mesh 16 x 16, fino a t = 0.05 con 8, 16 e 32 passi.
// Gli errori sono 9.07e-3, 3.80e-3, 1.09e-3. Il rapporto diretto darebbe 1.26 e 1.81 (sbagliato: E_h = 1.8e-3 ha il
// segno opposto a E_dt e la norma lo dimentica). Le differenze sono d1 = 5.27e-3 e d2 = 2.71e-3: rapporto 1.94, ordine
// 0.96; per Eulero all'indietro l'ordine si avvicina a 1 da sotto quando dt cala (0.90 con 4, 8, 16 passi). Con 64
// passi E_dt (circa 1.4e-3) scenderebbe sotto E_h e l'errore cambierebbe segno: la sottrazione non varrebbe piu'.
TEST(DecayingSine2D, TemporalConvergenceOfBackwardEulerIsFirstOrder) {
    const double t = 0.05;
    const double e1 = decaying_sine_error_2d(16, t / 8, 8, 1.0);
    const double e2 = decaying_sine_error_2d(16, t / 16, 16, 1.0);
    const double e3 = decaying_sine_error_2d(16, t / 32, 32, 1.0);
    const double d1 = e1 - e2;
    const double d2 = e2 - e3;

    ASSERT_GT(d1, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    ASSERT_GT(d2, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    EXPECT_NEAR(observed_order(d1, d2, 2.0), 1.0, 0.1) << "e = " << e1 << ", " << e2 << ", " << e3;
}

// ESEMPIO 115. Dati non validi: nessuna cella (n = 0): std::invalid_argument. Passo dt nullo, negativo o NaN, theta
// fuori da [0, 1] o NaN: se ne accorge solve_transient_2d (ESEMPIO 104), e l'eccezione deve arrivare fino a chi
// chiama. I casi al limite, theta = 0 e theta = 1, sono validi.
TEST(DecayingSine2D, RejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(decaying_sine_error_2d(0, 0.01, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.0, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, -0.01, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, not_a_number, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, -0.1), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, 1.1), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, not_a_number), std::invalid_argument);
    EXPECT_NO_THROW(decaying_sine_error_2d(4, 0.01, 1, 0.0));
    EXPECT_NO_THROW(decaying_sine_error_2d(4, 0.01, 1, 1.0));
}

// ESEMPIO 116. I numeri delle tre tabelle del seno che decade in 2D, quelle che apps/convergence_study stampa per il
// README: SPAZIO con Crank-Nicolson (dt = 2.5e-4, 200 passi) e TEMPO con Eulero all'indietro e con Crank-Nicolson
// (mesh 32 x 32, dt = 0.05 / passi). Tutte arrivano a t = 0.05. Misurati con questo metodo e ricontrollati in Python
// (matrici sparse e un solutore diretto, niente gradiente coniugato): coincidono fino alle 7 cifre scritte qui.
//
//     SPAZIO (Crank-Nicolson, dt = 2.5e-4, 200 passi)
//         n = 4:  2.452116e-02      n = 8:  6.894963e-03      n = 16:  1.776746e-03
//     TEMPO (mesh 32 x 32)                       4 passi        8 passi        16 passi
//         Eulero all'indietro (theta = 1):       2.016970e-02   1.035828e-02   5.092920e-03
//         Crank-Nicolson (theta = 1/2):          1.386333e-03   6.800985e-04   5.054105e-04
//
// Il test fissa le righe poco costose: le altre (n = 32 e n = 64 nello spazio, 32 e 64 passi nel tempo) costano di
// piu' senza ottimizzazioni. Se cambia il metodo (lo schema theta, il solutore, la norma) cambiano i numeri, e il test
// dice che le tabelle del README vanno rifatte.
TEST(DecayingSine2D, ErrorsMatchTheTablesInTheReadme) {
    struct Case {
        std::size_t n;
        double dt;
        std::size_t n_steps;
        double theta;
        double want;
    };
    const Case cases[] = {{4, 2.5e-4, 200, 0.5, 2.452116e-02},  {8, 2.5e-4, 200, 0.5, 6.894963e-03},
                          {16, 2.5e-4, 200, 0.5, 1.776746e-03}, {32, 0.05 / 4, 4, 1.0, 2.016970e-02},
                          {32, 0.05 / 8, 8, 1.0, 1.035828e-02}, {32, 0.05 / 16, 16, 1.0, 5.092920e-03},
                          {32, 0.05 / 4, 4, 0.5, 1.386333e-03}, {32, 0.05 / 8, 8, 0.5, 6.800985e-04},
                          {32, 0.05 / 16, 16, 0.5, 5.054105e-04}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(c.n, c.dt, c.n_steps, c.theta), c.want, 1e-6 * c.want)
            << "n = " << c.n << ", dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}