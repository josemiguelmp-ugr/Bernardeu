import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from functions import*      # My module



def modelo_A(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * np.exp(-a2*x) + a3 * x * np.exp(-a4*x) + a5 * x**2 * np.exp(-a6*x)

def modelo_B(x, b0, b1, b2, b3, b4):
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4*x)

def modelo_C(x, c0, c1, c2, c3, c4):
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4*x)

def prob_fit(delta, var):
    rhoc   =  2.57107
    lamdac =  0.328269 / var
    phic   =  0.434063 / var

    a0, a1, a2, a3, a4, a5, a6 = 0.1515133032572409, -0.7398474981570156, 3.7816153444693623, 4.324877339896807, 1.5356218603924487, 0.1772438178761217, 0.5989455285798162
    b0, b1, b2, b3, b4 = 0.2979027781611449, -3.412460631361204, 4.141648436938888, 0.28402315400946915, 0.5727311750174297
    c0, c1, c2, c3, c4 = -0.002389953474410354, 1.8267290015358206, 3.2804570704658347, 0.2952092122224205, 1.3164414519596497

    A = modelo_A(var, a0, a1, a2, a3, a4, a5, a6) 
    B = modelo_B(var, b0, b1, b2, b3, b4)
    C = modelo_C(var, c0, c1, c2, c3, c4)

    return A * np.exp(phic - lamdac * delta) * (delta + B + C / delta)**(-5/2)



var = 0.45
rho_ar = np.arange(0, 20, 0.05)
integration_ar = np.array([complex_integration(rs, var, step_size=1e-3) for rs in rho_ar])
prob_fit_ar = prob_fit(rho_ar, var)

print(area_under_curve(rho_ar, integration_ar))
print(area_under_curve(rho_ar, prob_fit_ar))


fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(rho_ar,  integration_ar,  color='b', linestyle='-', label='Numerical integration')
ax.plot(rho_ar, prob_fit_ar, color='orange', linestyle='-.', label='Multiparameter fit')

ax.set_yscale('log')
ax.set_ylim(1e-6, 5)
ax.set_ylabel(r'$P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()


ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

#plt.savefig('Figures/PDF_rho_more_curves.png')
plt.show()
