import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import csv

from scipy.optimize import curve_fit
from scipy.integrate import quad
from functions import*                 # My module

"""

# =======================================================================
#              FIT TO THE NUMERICAL INTEGRATION (EXAMPLE)
# =======================================================================


var   = 2.3           # Variance at radius R


# Values in the critical point from Mathematica (maximum of Fig. 1)
rhoc   =  2.57107
lamdac =  0.328269 / var
phic   =  0.434063 / var


# Región peliaguda
rho_ar_1 = np.arange(0, 0.5, 0.001)
integration_ar_1 = np.array([complex_integration(rs, var, step_size=1e-3) for rs in rho_ar_1])
#prob_fit_ar = prob_fit(rho_ar, var)

# Región suave
rho_ar_2 = np.arange(0.5, 20, 0.01)
integration_ar_2 = np.array([complex_integration(rs, var, step_size=1e-3) for rs in rho_ar_2])

# Concatenación
rho_ar = np.concatenate((rho_ar_1[4:], rho_ar_2))
integration_ar = np.concatenate((integration_ar_1[4:], integration_ar_2))



# Fit using Eq. (46)
def pdf_non_normalized(x, B, C):
    return np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)

def pdf_model(x, A, B, C):
    return A * pdf_non_normalized(x, B, C)



popt_s2, _ = curve_fit(pdf_model, rho_ar, integration_ar)
A, B, C = popt_s2

I0, _ = quad(pdf_non_normalized, 0, np.inf, args=(B, C))
A_ = 1 / I0
fit_s2_ = pdf_model(rho_ar, A_, B, C)

print(f'A={A}, B={B}, C={C}')
print(f'Área bajo la curva (integración numérica): {area_under_curve(rho_ar, integration_ar)}')
print(f'Área bajo la curva (ajuste): {area_under_curve(rho_ar, fit_s2_)}')



# =======================================================================
#                            COMPARISON FIGURE
# =======================================================================

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(rho_ar, integration_ar, marker='.', linestyle='None', label='Numerical integration')
ax.plot(rho_ar, fit_s2_, linestyle='-', label='Pumba')

ax.set_yscale('log')
#ax.set_ylim(1e-6, 5)
ax.set_ylabel(r'$P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()


ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

#plt.savefig('Figures/PDF_rho_more_curves.png')
plt.show()

"""


"""
# Los archivos fits.csv y fits_2.csv se han hecho dejando los parámetros B y C libres y tomando A de manera que normalice la PDF
# Los archivos fits_4.csv y fits_5.csv se han hecho dejando todos los parámetros (A, B, C) libres, por lo que las PDFs no salen normalizadas

# También hay que considerar que en los nuevos fits estamos metiendo una mayor precisión alrededor del pico de la curva

# =======================================================================
#                      FITS FOR VARIANCES < 4.5
# =======================================================================


# Fit using Eq. (46)
def pdf_non_normalized(x, B, C):
    return np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)

def pdf_model(x, A, B, C):
    return A * pdf_non_normalized(x, B, C)




filename = 'Fits/fits.csv'
variances = np.arange(0.2, 4.5, 0.1)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Variance', 'A', 'B', 'C'])


for sigma2 in variances:
    lamdac =  0.328269 / sigma2
    phic   =  0.434063 / sigma2


    # Región peliaguda
    rho_ar_1 = np.arange(0, 0.5, 0.001)
    integration_ar_1 = np.array([complex_integration(rs, sigma2, step_size=1e-3) for rs in rho_ar_1])

    # Región suave
    rho_ar_2 = np.arange(0.5, 20, 0.01)
    integration_ar_2 = np.array([complex_integration(rs, sigma2, step_size=1e-3) for rs in rho_ar_2])

    # Concatenación
    rho_ar = np.concatenate((rho_ar_1[4:], rho_ar_2))
    integration_ar = np.concatenate((integration_ar_1[4:], integration_ar_2))


    popt_s2, _ = curve_fit(pdf_model, rho_ar, integration_ar)
    A, B, C = popt_s2
    I0, _ = quad(pdf_non_normalized, 0, np.inf, args=(B, C))
    A_ = 1 / I0                                                     # A normalizado
    print(f'var={sigma2}, A={A_}, B={B}, C={C}')

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([sigma2, A_, B, C])

"""

# =======================================================================
#                       FITS FOR VARIANCES >= 4.5
# =======================================================================


# Fit using Eq. (46)
def pdf_non_normalized(x, B, C):
    return np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)

def pdf_model(x, A, B, C):
    return A * pdf_non_normalized(x, B, C)



filename = 'Fits/fits_2.csv'
variances = np.arange(4.5, 20.5, 0.5)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Variance', 'A', 'B', 'C'])


for sigma2 in variances:
    lamdac =  0.328269 / sigma2
    phic   =  0.434063 / sigma2


    # Región peliaguda
    rho_ar_1 = np.arange(0, 0.5, 0.001)
    integration_ar_1 = np.array([complex_integration(rs, sigma2, step_size=1e-3) for rs in rho_ar_1])

    # Región suave
    rho_ar_2 = np.arange(0.5, 20, 0.01)
    integration_ar_2 = np.array([complex_integration(rs, sigma2, step_size=1e-3) for rs in rho_ar_2])

    # Concatenación
    rho_ar = np.concatenate((rho_ar_1[4:], rho_ar_2))
    integration_ar = np.concatenate((integration_ar_1[4:], integration_ar_2))


    popt_s2, _ = curve_fit(pdf_model, rho_ar, integration_ar)
    A, B, C = popt_s2
    I0, _ = quad(pdf_non_normalized, 0, np.inf, args=(B, C))
    A_ = 1 / I0                                                     # A normalizado
    print(f'var={sigma2}, A={A_}, B={B}, C={C}')

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([sigma2, A_, B, C])




"""

# =======================================================================
#                   VARIATION OF PARAMETERS A, B, C
# =======================================================================


df_fit_1 = pd.read_csv('Fits/fits.csv')
variances_1 = df_fit_1['Variance'].values
A_ar_1 = df_fit_1['A'].values
B_ar_1 = df_fit_1['B'].values
C_ar_1 = df_fit_1['C'].values

df_fit_2 = pd.read_csv('Fits/fits_2.csv')
variances_2 = df_fit_2['Variance'].values
A_ar_2 = df_fit_2['A'].values
B_ar_2 = df_fit_2['B'].values
C_ar_2 = df_fit_2['C'].values

variances = np.concatenate((variances_1, variances_2))
A_ar = np.concatenate((A_ar_1, A_ar_2))
B_ar = np.concatenate((B_ar_1, B_ar_2))
C_ar = np.concatenate((C_ar_1, C_ar_2))




def modelo_A(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * np.exp(- a2 * x) + a3 * x * np.exp(- a4 * x) + a5 * x**2 * np.exp(- a6 * x)

def modelo_B(x, b0, b1, b2, b3, b4):
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4 * x)

def modelo_C(x, c0, c1, c2, c3, c4):
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4 * x)


popt_A, pcov_A = curve_fit(modelo_A, variances, A_ar)
popt_B, pcov_B = curve_fit(modelo_B, variances, B_ar)
popt_C, pcov_C = curve_fit(modelo_C, variances, C_ar)

a0, a1, a2, a3, a4, a5, a6 = popt_A
b0, b1, b2, b3, b4 = popt_B
c0, c1, c2, c3, c4 = popt_C

print(f'a0={a0}, a1={a1}, a2={a2}, a3={a3}, a4={a4}, a5={a5}, a6={a6}')
print(f'b0={b0}, b1={b1}, b2={b2}, b3={b3}, b4={b4}')
print(f'c0={c0}, c1={c1}, c2={c2}, c3={c3}, c4={c4}')

print( np.sqrt(np.diag(pcov_A)) )

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(variances, A_ar, linestyle='-', color='b', label='A')
ax.plot(variances, B_ar, linestyle='-', color='m', label='B')
ax.plot(variances, C_ar, linestyle='-', color='g', label='C')

variances = np.linspace(0.2, 20, 3000)
fit_A = modelo_A(variances, *popt_A)
fit_B = modelo_B(variances, *popt_B)
fit_C = modelo_C(variances, *popt_C)

ax.plot(variances, fit_A, linestyle='-.', label='A fit')
ax.plot(variances, fit_B, linestyle='-.', label='B fit')
ax.plot(variances, fit_C, linestyle='-.', label='C fit')

ax.set_ylabel(r'Parameters')
ax.set_xlabel(r'$\sigma^2$')
ax.legend()

plt.show()

"""