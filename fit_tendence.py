# Programa para comprobar que los ajustes con varianza cada vez mayor siguen la tendencia de las curvas 
# sacadas directamente por integración numérica (picos cada vez más estrechos y caídas más rápidas)


import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from scipy.optimize import curve_fit
from scipy.integrate import quad
from functions import*                  # My module


def modelo_A(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * np.exp(- a2 * x) + a3 * x * np.exp(- a4 * x) + a5 * x**2 * np.exp(- a6 * x)

def modelo_B(x, b0, b1, b2, b3, b4):
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4 * x)

def modelo_C(x, c0, c1, c2, c3, c4):
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4 * x)

def pdf_non_normalized(x, var, B, C):
    lamdac =  0.328269 / var
    phic   =  0.434063 / var
    return np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)

def prob_fit(delta, var):
    rhoc   =  2.57107
    lamdac =  0.328269 / var
    phic   =  0.434063 / var

    a0, a1, a2, a3, a4, a5, a6 = 0.04695910742441713, -0.6642049938350947, 2.4353359038289777, 3.6354491307579315, 1.5181112635331262, 0.17469598137756703, 0.6214850573229408
    b0, b1, b2, b3, b4 = 0.0979931675570106, 3.1972055079415354, 3.9083535195901087, 0.35435722761935323, 0.5201525548415581
    c0, c1, c2, c3, c4 = 0.0023928609265879866, 1.7761815559139071, 3.632416222582498, 0.4986992331769517, 1.650113588148832

    B = modelo_B(var, b0, b1, b2, b3, b4)
    C = modelo_C(var, c0, c1, c2, c3, c4)

    # Normalización
    I0, _ = quad(pdf_non_normalized, 0, np.inf, args=(var, B, C))
    A = 1 / I0

    return A * np.exp(phic - lamdac * delta) * (delta + B + C / delta)**(-5/2)


plt.figure()
n_curves = 30
cmap = cm.get_cmap('inferno', n_curves)

rho_ar = np.linspace(0, 20, 10000)
variances = np.linspace(7, 20, n_curves)
for i in range(len(variances)):
    sigma2 = variances[i]
    Pfit = prob_fit(rho_ar, sigma2)
    plt.plot(rho_ar, Pfit, color=cmap(i), label=f'{sigma2:.2f}')

plt.yscale('log')
#plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.show()
