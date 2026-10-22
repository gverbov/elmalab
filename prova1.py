python
```
# Laboratorio 14/10/2026
# Docente: Matteo Franchini
# Studente: Glib Verzhbovskyi
import numpy as np
import matplotlib.pyplot as pl

C = 5
S = 161
B = 20
I = 400
A = 0.9
D = 5.2
E = [0.25, 0.5, 1, 2, 4, 8]

def cent(X, Y):
    l = len(X)
    i = 0
    S = 0
    p = X[i]
    while (p <= 1.2 and i < l):
        p = X[i]
        S += Y[i]
        i+=1
    return S

def dev4(X, T):
    mX = np.mean(X)
    dX = (sum([(X[i] - mX)**2 for i in range(len(X))])/(len(X) - 1))**0.5
    dmX = dX/(len(X)**0.5)
    scX = abs((mX - T)/T)
    return mX, dX, dmX, scX

def sim(s=1, P = None):
    if P is None: P = sorted(np.random.uniform(-C, C, size=S))
    fE = lambda x: np.sinc(A*x/np.pi)**2
    fC = lambda x: np.cos(D*x)**2
    M = lambda x: B + I*fE(x)*fC(x)

    MP = [M(i) for i in P]
    nMP = [M(-i) for i in P]
    pP = [np.random.poisson(s*i) for i in MP]
    eP = [max(i**0.5, 1) for i in pP]
    cM = cent(P, MP)
    cP = cent(P, pP)

    for i in range(S):
        assert MP[i] >= 0, "Esiste punto t.c. < 0"
        assert np.allclose(MP[i], nMP[i]), "Funzione non è simmetrica"

    return P, pP, cM, cP, eP

def graph():
    P, pP, cM, cP, eP = sim()
    print(cP - cM, np.sqrt(cM))
    pl.errorbar(P, pP, yerr=eP, color='blue', linewidth=2)
    pl.xlabel('Punti arbitrari scelti in [-5, 5]')
    pl.ylabel('Il numero medio atteso di conteggi')
    pl.grid(True)
    pl.legend()
    pl.show()

def iter1(n = 2000):
    vM = [[] for _ in range(n)]
    vP = [0]*n
    vC = [0]*n
    for i in range(n):
        if (i % (0.1*n) == 0): print("iter1 has completed: ", i/n)
        P, pP, cM, cP, eP = sim()
        vM[i] = pP
        vP[i] = cP
        vC[i] = cM
    return vM, vP, np.mean(vP), np.mean(vC)

def hist():
    M, P, mP, mC = iter1()
    print(dev4(P, mC))

    pl.hist(P, bins=30, alpha=0.7, edgecolor='black')
    pl.axvline(mP, color='red', linestyle='--', label=f'Media simulata = {mP:.2f}')
    pl.axvline(mC, color='green', linestyle='-', label=f'Valore atteso = {mC:.2f}')
    pl.xlabel('P')
    pl.ylabel('Frequenza')
    pl.legend()
    pl.grid(True)
    pl.show()

def stat(n=1000):
    P = sorted(np.random.uniform(-5, 5, size=161))
    L = []
    for s in E:
        c = np.array([sim(s, P)[3] for _ in range(n)])
        m = np.mean(c)
        std = np.std(c, ddof=1)
        r = std / m if m != 0 else np.nan
        L.append((s, m, std, r, r*np.sqrt(s)))

    S = np.array([i[0] for i in L])
    R = np.array([i[3] for i in L])

    pl.loglog(S, R, 'o-', label='Simulazione')
    pl.loglog(S, R[0]*np.sqrt(S[0]/S), '--', label=r'$r(s)\propto 1/\sqrt{s}$')
    pl.xlabel("Fattore d'esposizione s")
    pl.ylabel("Incertezza relativa r(s)")
    pl.grid(True, which="both")
    pl.show()
    return L

graph()
hist()
stat()
```
