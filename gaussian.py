# Script para comparar una gaussiana centrada en el valor más probable con la LDF

# %%
# Celda de importaciones
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from scipy.optimize  import curve_fit, fsolve
from scipy.integrate import quad

from functions import log_formatter


# %%
# Funciones del ajuste para la PDF

def modelo_A(x):
    a0, a1, a2, a3, a4, a5, a6 = 0.04695910742441713, -0.6642049938350947, 2.4353359038289777, 3.6354491307579315, 1.5181112635331262, 0.17469598137756703, 0.6214850573229408
    return a0 + a1 * np.exp(- a2 * x) + a3 * x * np.exp(- a4 * x) + a5 * x**2 * np.exp(- a6 * x)

def modelo_B(x):
    b0, b1, b2, b3, b4 = 0.0979931675570106, 3.1972055079415354, 3.9083535195901087, 0.35435722761935323, 0.5201525548415581
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4 * x)

def modelo_C(x):
    c0, c1, c2, c3, c4 = 0.0023928609265879866, 1.7761815559139071, 3.632416222582498, 0.4986992331769517, 1.650113588148832
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4 * x)


def integral_non_norm(var):
    
    """
    Función que integra la PDF en todo el rango de la variable aleatoria (de 0 a infinito).
    Para ejecutar esta integral numéricamente, hacemos un cambio de variable, de manera que la integral vaya entre 0 y 1.
    Si la PDF está normalizada, la integral debería dar 1 como resultado

    Parameters
    ----------
    var : float
        Varianza
    
    Returns
    -------
    float
        Valor de la probabilidad total, obtenido por la integración numérica
    """

    lamdac =  0.328269 / var
    phic   =  0.434063 / var

    B = modelo_B(var)
    C = modelo_C(var)

    def integrando_non_norm(u):
        # Cambio de variable: u = delta / (1 - delta)
        delta = u / (1 - u)
        jacobiano = 1 / (1 - u)**2
        PDF = np.exp(phic - lamdac * delta) * (delta + B + C / delta)**(-5/2)

        return PDF * jacobiano

    res, _ = quad(integrando_non_norm, 0, 1, epsrel=1e-8)
    return res


# Devuelve la PDF normalizada, ajustada a la forma de la LDF
def PDF_fit(delta, var):
    lamdac =  0.328269 / var
    phic   =  0.434063 / var

    A = 1.0 / integral_non_norm(var) if integral_non_norm(var) != 0 else 0
    B = modelo_B(var)
    C = modelo_C(var)

    return A * np.exp(phic - lamdac * delta) * (delta + B + C / delta)**(-5/2)



# %%
# Intentamos relacionar la gaussiana con la LDF

from functions import log_formatter

# Gaussiana
def PDF_gaussian(delta, var, mean):
    PDF = 1 / np.sqrt(2 * np.pi * var) * np.exp( - (delta - mean)**2 / (2 * var) )
    return PDF


def PDF_gaussian_var(delta, var):
    mean = delta_star
    return PDF_gaussian(delta, var, mean)


def maximo(delta, var):
    lamdac = 0.328269 / var
    B = modelo_B(var)
    C = modelo_C(var)

    return 2*lamdac*delta**3 + 2*lamdac*B*delta**2 + 2*lamdac*C*delta + 5*delta**2 - 5*C


def maximum_finder(var):
    initial_guess = 1.0
    function = lambda delta: maximo(delta, var)
    solution = fsolve(function, initial_guess)
    return solution[0]


def var_eff(delta_star, var):
    B = modelo_B(var)
    C = modelo_C(var)

    numerator = delta_star**2 * (delta_star**2 + B * delta_star + C )**2
    denominator = C**2 - delta_star**4 + 2 * C * delta_star * (2 * delta_star + B)

    return (2/5) * numerator / denominator



var = 0.2
delta_ar   = np.arange(0, 15, 0.0001)
delta_ar_g = np.arange(-0.1, 15, 0.0001)

PDF_fit_ar = PDF_fit(delta_ar, var)
index_max = PDF_fit_ar.argmax()
d_max = delta_ar[index_max]

delta_star = maximum_finder(var)
var_effective = var_eff(delta_star, var)
C_norm = PDF_fit(delta_star, var)


def PDF_gaussian_2(delta, var_eff, mean, norm):
    return norm * np.exp( - (delta - mean)**2 / (2 * var_eff) )

PDF_gaussian_ar = PDF_gaussian_2(delta_ar_g, var_effective, delta_star, C_norm)

popt, _ = curve_fit(PDF_gaussian_var, [d_max], [PDF_fit_ar[index_max]])
gaussian_fit_ar = PDF_gaussian(delta_ar_g, popt[0], delta_star)

print( f"\nCálculo de delta_star = {delta_star}; Pico de la PDF = {d_max}"  )
print( f"Expansión de Taylor en torno al máximo: var={var_effective}" )
print( f"Ajuste gaussiano al pico de la PDF: var={popt[0]}" )
print( "\n" )

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(delta_ar, PDF_fit_ar, label='LDF')
ax.plot(delta_ar_g, PDF_gaussian_ar, linestyle='dashed', label='Gaussian Taylor expansion')
ax.plot(delta_ar_g, gaussian_fit_ar, '-.', label=f'Gaussian fit normalized')

ax.set_yscale('log')
ax.set_ylim(4, 6)
ax.set_xlim(-0.1, 0.2)

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

ax.set_xlabel(u"$\\delta$")
ax.set_ylabel(u"$\\mathcal{P}\\,(\\delta)$")

plt.legend()
plt.show()





# %%

variances = np.arange(1e-3, 80, 1e-3)
variances_eff = []

for variance in variances:
    delta_star = maximum_finder(variance)
    var_effective = var_eff(delta_star, variance)
    variances_eff.append(var_effective)




# %%


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(variances, variances_eff, 'm-')
ax.set_xlim(0, 5)
ax.set_xlabel(u"$\\sigma^2$")
ax.set_ylabel(u"$\\sigma_{eff}^2$")
plt.show()

# %%
