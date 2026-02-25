# Script para comparar una gaussiana centrada en el valor más probable con la LDF

# %%
# Celda de importaciones
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from functions import complex_integration, log_formatter

# %%

var = 0.45

# Gaussiana
def PDF_gaussian(delta, var, mean):
    PDF = 1 / np.sqrt(2 * np.pi * var) * np.exp( - (delta - mean)**2 / (2 * var) )
    return PDF


rho_ar = np.arange(0, 15, 0.01)
integration = []
for rs in rho_ar:
    integration.append(complex_integration(rs, var, 1e-3))

integration_ar = np.array(integration)

index_max = integration_ar.argmax()
r_max = rho_ar[index_max]

# Ajuste de la gaussiana
def PDF_gaussian_var(delta, var):
    mean = r_max
    return PDF_gaussian(delta, var, mean)

popt, _ = curve_fit(PDF_gaussian_var, [rho_ar[index_max]], [integration_ar[index_max]])
print("\n")
print(f"La varianza que estamos utilizando para la LDF es: var_ldf = {var}")
print(f"La varianza necesaria para tener una gaussiana que se ajuste a la LDF es: var_gaussian = {popt[0]:.4f}")



# %%
# FIGURE

fig, ax = plt.subplots()

ax.plot(rho_ar, np.real(integration_ar), '.',  label='Numerical integration')
ax.plot(rho_ar, PDF_gaussian(rho_ar, popt[0], r_max), '-.', label='Gaussian')

ax.set_yscale('log')
ax.set_ylim(1e-6, 2)
ax.set_xlim(-0.1, 15)
ax.set_ylabel(r'$P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

plt.show()
plt.close()

# %%
