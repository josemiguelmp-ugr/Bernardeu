import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from scipy.integrate import quad
from functions import*      # My module


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



var = 14


# Región peliaguda
rho_ar_1 = np.arange(0, 0.5, 0.001)
integration_ar_1 = np.array([ complex_integration(rs, var, step_size=1e-3) for rs in rho_ar_1 ])

# Región suave
rho_ar_2 = np.arange(0.5, 20, 0.01)
integration_ar_2 = np.array([ complex_integration(rs, var, step_size=1e-3) for rs in rho_ar_2 ])

# Concatenación
rho_ar = np.concatenate((rho_ar_1[4:], rho_ar_2))
integration_ar = np.concatenate((integration_ar_1[4:], integration_ar_2))







df_fit_2 = pd.read_csv('Fits/fits_2.csv')
A = df_fit_2['A'].values[0]
B = df_fit_2['B'].values[0]
C = df_fit_2['C'].values[0]

lamdac =  0.328269 / var
phic   =  0.434063 / var
fit_original = A * np.exp(phic - lamdac * rho_ar) * (rho_ar + B + C / rho_ar)**(-5/2)


print( area_under_curve(rho_ar[integration_ar < 0.1], integration_ar[integration_ar < 0.1]) )
print( area_under_curve(rho_ar[integration_ar < 0.1], fit_original[integration_ar < 0.1]) )



prob_fit_ar = prob_fit(rho_ar, var)

print(f'Área bajo la curva (integración numérica): {area_under_curve(rho_ar, integration_ar)}')
print(f'Área bajo la curva (ajuste original):{area_under_curve(rho_ar, fit_original)}')
print(f'Área bajo la curva (ajuste):{area_under_curve(rho_ar, prob_fit_ar)}')


fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(rho_ar, integration_ar, color='b', marker='.', linestyle='None', label='Numerical integration')
ax.plot(rho_ar, prob_fit_ar, color='orange', linestyle='-.', label='Multiparameter fit')
ax.plot(rho_ar, fit_original, color='m', linestyle='-.', label='Original fit')

ax.set_title(rf'$\sigma^2$={var}')
ax.set_yscale('log')
#ax.set_ylim(1e-6, 5)
ax.set_ylabel(r'$P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()


ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

#plt.savefig('Figures/PDF_rho_more_curves.png')
plt.show()
