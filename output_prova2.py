# File: Simulazione
# Laboratorio 22/10/2026
# Docente: Matteo Franchini
# Studente: Glib Verzhbovskyi

import numpy as np
import matplotlib.pyplot as pl

L = [0.0, 0.0, 0.10, 1.0]
U = [np.inf, np.inf, 3.0, 12.0]

def spl(l):
    l = l.strip().split('; ')
    s = int(l[0])
    D = [i.split() for i in l[1:] if i.strip()]
    x = [float(i[0]) for i in D]
    y = [float(i[1]) for i in D]
    dY = [float(i[2]) for i in D]
    return s, x, y, dY

def imp(file="exp.txt"):
    with open(file, "r", encoding="utf-8") as f:
        l0 = f.readline().strip().split('; ')
        N = int(l0[0])
        S = int(l0[1])
        X = []
        Y = []
        dY = []
        for i in range(N):
            s, x, y, dy = spl(f.readline())
            assert s == S, "La lunghezza dei set non è uguale"
            assert len(x) == len(y) == S, "Lunghezze non coincidenti"
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(dy)), "Valori non finiti"
            assert np.all(np.array(dy) > 0), "Incertezza non positiva"
            assert np.all(np.diff(x) >= 0), "Le posizioni non sono ordinate"
            X.append(x)
            Y.append(y)
            dY.append(dy)
        return X, Y, dY

def modello(x, B, I0, alpha, delta):
    z = alpha*x
    return B + I0*np.sinc(z/np.pi)**2*np.cos(delta*x)**2

def modello_rid(x, B, I0, delta):
    return B + I0*np.cos(delta*x)**2

def est(x, y):
    B0 = float(np.median(y))
    I00 = float(max(y) - B0)
    alpha0 = 0.9
    delta0 = 5.2
    p0 = [max(B0, 0), max(I00, 0), alpha0, delta0]
    return p0

def fit(x, y, dy, p0):
    popt, pcov = curve_fit(
        modello, x, y,
        p0=p0,
        sigma=dy,
        absolute_sigma=True,
        bounds=(L, U),
        maxfev=100000
    )
    return popt, pcov

def diagnostica(x, y, sigma_y, popt, modello_fit=modello, npar=4):
    residui = np.array(y) - modello_fit(np.array(x), *popt)
    pull = residui/np.array(sigma_y)
    chi2_obs = float(np.sum(pull**2))
    ndf = len(x) - npar
    chi2_rid = chi2_obs/ndf
    p_value = chi2.sf(chi2_obs, ndf)
    return residui, pull, chi2_obs, ndf, chi2_rid, p_value

def grafico(x, y, sigma_y, popt, pull, nome="Fit completo", modello_fit=modello):
    fig, (ax1, ax2) = pl.subplots(2, 1, sharex=True, figsize=(9, 7),
                                  gridspec_kw={'height_ratios': [3, 1]})
    ax1.errorbar(x, y, yerr=sigma_y, fmt='.', capsize=2, label='Dati')
    xx = np.linspace(min(x), max(x), 1000)
    ax1.plot(xx, modello_fit(xx, *popt), label='Fit')
    ax1.set_ylabel('Intensità (unità arbitrarie)')
    ax1.set_title(nome)
    ax1.legend()
    ax1.grid(True)

    ax2.axhline(0, linewidth=1)
    ax2.axhline(2, linestyle='--', linewidth=1)
    ax2.axhline(-2, linestyle='--', linewidth=1)
    ax2.errorbar(x, pull, fmt='.', capsize=2)
    ax2.set_xlabel('Posizione x (mm)')
    ax2.set_ylabel('Pull')
    ax2.grid(True)
    fig.tight_layout()
    return fig

# Caricamento e controllo dei dati
X, Y, dY = imp()
x = np.array(X[0], dtype=float)
y = np.array(Y[0], dtype=float)
sigma_y = np.array(dY[0], dtype=float)

assert len(x) == len(y) == len(sigma_y), "Lunghezze non coincidenti"
assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(sigma_y)), "Valori non finiti"
assert np.all(sigma_y > 0), "Incertezza non positiva"
assert np.all(np.diff(x) >= 0), "Le posizioni non sono ordinate"

print("Numero di punti:", len(x))
print("Intervallo x:", (min(x), max(x)))
print("Intervallo y:", (min(y), max(y)))
print("Intervallo sigma_y:", (min(sigma_y), max(sigma_y)))

pl.figure(figsize=(9, 4))
pl.errorbar(x, y, yerr=sigma_y, fmt='.', capsize=2)
pl.xlabel('Posizione x (mm)')
pl.ylabel('Intensità (unità arbitrarie)')
pl.title('Dati')
pl.grid(True)
pl.tight_layout()

# Stime iniziali e fit pesato
p0 = est(x, y)
print("Parametri, stime iniziali e bounds:")
for nome, valore, basso, alto in zip(['B', 'I0', 'alpha', 'delta'], p0, L, U):
    print(f'{nome}: p0={valore}, bounds=({basso}, {alto})')

popt, pcov = fit(x, y, sigma_y, p0)
dp = np.sqrt(np.diag(pcov))
nomi = ['B', 'I0', 'alpha', 'delta']
unita = ['unità arbitrarie', 'unità arbitrarie', 'mm^-1', 'mm^-1']
print("Risultati del fit:")
for nome, valore, errore, unita_i in zip(nomi, popt, dp, unita):
    print(f'{nome} = {valore:.6g} +/- {errore:.2g} {unita_i}')
print("Parametri vicini ai bounds:")
for nome, valore, basso, alto in zip(nomi, popt, L, U):
    if np.isfinite(basso) and abs(valore-basso) < 1e-4*(1+abs(basso)):
        print(nome, "vicino al limite inferiore")
    if np.isfinite(alto) and abs(valore-alto) < 1e-4*(1+abs(alto)):
        print(nome, "vicino al limite superiore")

# Matrice di correlazione
corr = pcov/np.outer(dp, dp)
print("Matrice di correlazione:")
print(corr)
pl.figure(figsize=(6, 5))
im = pl.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
pl.colorbar(im, label='Correlazione')
pl.xticks(range(4), nomi)
pl.yticks(range(4), nomi)
pl.title('Matrice di correlazione')
for i in range(4):
    for j in range(4):
        pl.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center')
pl.tight_layout()
corr_off = corr.copy()
np.fill_diagonal(corr_off, 0)
i, j = np.unravel_index(np.argmax(np.abs(corr_off)), corr_off.shape)
print("Coppia maggiormente correlata:", nomi[i], nomi[j], "rho =", corr[i, j])

# Diagnostica del fit completo
residui, pull, chi2_obs, ndf, chi2_rid, p_value = diagnostica(x, y, sigma_y, popt)
print("Fit completo: chi2 =", chi2_obs, "ndf =", ndf, "chi2/ndf =", chi2_rid, "p-value =", p_value)
grafico(x, y, sigma_y, popt, pull)
pl.show()

# Attività 5: confronto con il modello ridotto
p0_rid = [popt[0], popt[1], popt[3]]
popt_rid, pcov_rid = curve_fit(
    modello_rid, x, y,
    p0=p0_rid,
    sigma=sigma_y,
    absolute_sigma=True,
    bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, 12.0]),
    maxfev=100000
)
residui_rid, pull_rid, chi2_rid_obs, ndf_rid, chi2_rid_red, p_value_rid = diagnostica(
    x, y, sigma_y, popt_rid, modello_fit=modello_rid, npar=3
)
print("Confronto dei modelli:")
print("Completo: Npar=4, chi2 =", chi2_obs, "ndf =", ndf, "chi2/ndf =", chi2_rid, "p-value =", p_value)
print("Ridotto:  Npar=3, chi2 =", chi2_rid_obs, "ndf =", ndf_rid, "chi2/ndf =", chi2_rid_red, "p-value =", p_value_rid)
grafico(x, y, sigma_y, popt_rid, pull_rid, nome="Fit ridotto: inviluppo omesso", modello_fit=modello_rid)
pl.show()

# Attività 6: stabilità rispetto alla stima iniziale di delta
print("Controllo di stabilità:")
for fattore in [0.8, 1.0, 1.2]:
    p0_stab = p0.copy()
    p0_stab[3] = np.clip(p0[3]*fattore, L[3]+1e-8, U[3]-1e-8)
    try:
        popt_stab, pcov_stab = fit(x, y, sigma_y, p0_stab)
        chi2_stab = np.sum(((y-modello(x, *popt_stab))/sigma_y)**2)
        print(f'fattore={fattore}: convergenza, delta={popt_stab[3]:.6g}, chi2={chi2_stab:.6g}, parametri={popt_stab}')
    except (RuntimeError, ValueError) as errore:
        print(f'fattore={fattore}: fit non convergente ({errore})')
      
# Domande finali:
# 1. k, a e d non possono essere lasciati simultaneamente liberi perché una singola curva identifica solo alpha=k*a e delta=k*d.
# 2. Il pull plot mostra gli scarti normalizzati per le incertezze e rende visibili strutture sistematiche non evidenti nel grafico dei dati.
# 3. Con absolute_sigma=False la matrice di covarianza viene riscalata in base al chi quadrato ridotto.
# 4. No: piccole incertezze formali descrivono la precisione stimata entro il modello, non garantiscono che il modello sia corretto.
# 5. La correlazione può ridursi con dati che coprano meglio l'inviluppo, includendo minimi e massimi informativi su un intervallo più ampio.
