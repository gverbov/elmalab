# File: Simulazione
# Laboratorio 22/10/2026
# Docente: Matteo Franchini
# Studente: Glib Verzhbovskyi

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from scipy.stats import chi2

A = 0.9
B = 20
D = 5.2
I = 400
C = 5
S = 161

def dev(X):
    dX = (sum([(X[i] - np.mean(X))**2 for i in range(len(X))])/(len(X) - 1))**0.5
    return dX

def sim():
    P = sorted(np.random.uniform(-C, C, size=S))
    M = lambda x: B + I*np.sinc(A*x/np.pi)**2*np.cos(D*x)**2
    MP = [M(i) for i in P]
    Y = [np.random.poisson(i) for i in MP]
    sigma_y = [max(i, 1)**0.5 for i in Y]

    for i in range(S):
        assert MP[i] >= 0, "Esiste punto t.c. < 0"
        assert Y[i] >= 0, "Esiste conteggio negativo"
        assert sigma_y[i] > 0, "Incertezza non positiva"

    return P, Y, sigma_y

def exp(n=2000):
    with open("exp.txt", "w", encoding="utf-8") as f:
        f.write(f'{n}; {S}\n')
        for _ in range(n):
            P, Y, sigma_y = sim()
            f.write(f'{S}; ')
            f.write('; '.join([f'{P[i]} {Y[i]} {sigma_y[i]}' for i in range(S)]))
            f.write('\n')

# Generazione dei dati simulati
exp()
