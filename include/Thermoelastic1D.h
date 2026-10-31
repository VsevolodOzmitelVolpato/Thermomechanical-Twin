#ifndef THERMOELASTIC1D_H
#define THERMOELATIC1D_H

#include <vector>
#include "Fem1D.h" 

// Termo-meccanica 1D: una barra di sezione 1 che si dilata quando si scalda.
//
// Il materiale ha modulo di Young E (> 0) e coefficiente di dilatazione termica alpha.
// T_ref e' la temperatura alla quale la barra e' senza tensioni.
// Scaldata di dT = T - T_ref, la barra vorrebbe allungarsi di  alpha dT  per ogni unita' di lunghezza.
// Con u(x) lo spostamento, la tensione e'   sigma = E (u' - alpha dT):
// conta solo la parte di deformazione che NON e' dovuta al calore.

// Forze nodali equivalenti alla dilatazione termica (un numero per nodo).
// Si immagina ogni elemento bloccato ai due estremi: scaldandosi spinge i due nodi verso l'esterno
// con forza E alpha dT, dove dT = (T_sinistra + T_destra) / 2 - T_ref e' lo scarto medio
// dell'elemento (la lunghezza dell'elemento si semplifica). Il nodo sinistro riceve -E alpha dT,
// il destro +E alpha dT; un nodo in comune a due elementi somma i due contributi.
// T = temperatura in ogni nodo (x.size() numeri, lineare dentro l'elemento).
// Lancia std::invalid_argument se T non ha un numero per nodo.
DVector assemble_thermal_load(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref);

// Condizione sull'estremo destro della barra: libero oppure bloccato (spostamento 0).
// L'estremo sinistro e' sempre bloccato.
enum class RightEnd { Free, Fixed };

// Spostamento u in ogni nodo di una barra con nodi x (crescenti), riscaldata da T (un valore per nodo).
// Si risolve K u = f, con K la matrice di rigidezza (E/h per ogni elemento, come la K del calore
// con E al posto di k) e f le forze termiche. L'estremo sinistro e' bloccato (u = 0); l'estremo
// destro e' libero oppure bloccato (u = 0), secondo right_end.
// Lancia std::invalid_argument se x ha meno di 2 nodi, se T non ha un valore per nodo o se E non e' > 0,
// e std::runtime_error se CG non converge.
DVector solve_thermal_expansion(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, RightEnd right_end);

// Tensione in ogni elemento (x.size() - 1 numeri), dato lo spostamento u di ogni nodo:
//   sigma_e = E * ( (u[e+1] - u[e]) / h_e  -  alpha * dT_medio_e )
// h_e e' la lunghezza dell'elemento e dT_medio_e la media degli scarti (T - T_ref) dei suoi due nodi.
// Il primo termine e' la deformazione totale, il secondo quella che il calore produrrebbe comunque:
// la tensione nasce dalla differenza. Positiva = trazione, negativa = compressione.
// Una barra con meno di 2 nodi non ha elementi: il risultato e' un vettore vuoto.
// Lancia std::invalid_argument se T o u non hanno un valore per ogni nodo.
DVector element_stress(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, const DVector& u);


#endif