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


#endif