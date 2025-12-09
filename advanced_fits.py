import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from functions import*      # My module


var   = 0.99           # Variance at radius R


# Values in the critical point from Mathematica (maximum of Fig. 1)
rhoc   =  2.57107
lamdac =  0.328269 / var
phic   =  0.434063 / var

"""
rho_ar = np.arange(0, 20, 0.05)
integration_ar = np.array([complex_integration(rs, var, step_size=1e-3) for rs in rho_ar])



# =======================================================================
#                 FIT FOR THE NUMERICAL INTEGRATION
# =======================================================================


# Fit using Eq. (46)
def expression_2(x, A, B, C): 
    return A * np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)

popt_s2, _ = curve_fit(expression_2, rho_ar[1:], integration_ar[1:])
A, B, C = popt_s2
fit_s2_ = expression_2(rho_ar, A, B, C)

print(f'A={A}, B={B}, C={C}')
print(area_under_curve(rho_ar, integration_ar))
print(area_under_curve(rho_ar, fit_s2_))



# =======================================================================
#                            COMPARISON FIGURE
# =======================================================================

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(rho_ar, integration_ar, label='Numerical integration')
ax.plot(rho_ar, fit_s2_, linestyle='-', label='Pumba')

ax.set_yscale('log')
ax.set_ylim(1e-6, 5)
ax.set_ylabel(r'$P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()


ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

#plt.savefig('Figures/PDF_rho_more_curves.png')
plt.show()

"""

"""
rhoc   =  2.57107
rho_ar = np.arange(0, 4.5, 0.05)

# Fit using Eq. (46)
def expression_2(x, A, B, C): 
    return A * np.exp(phic - lamdac * x) * (x + B + C / x)**(-5/2)


variances = np.arange(4.5, 20, 0.5)
A_ar, B_ar, C_ar = [], [], []
for sigma2 in variances:
    lamdac =  0.328269 / sigma2
    phic   =  0.434063 / sigma2
    integration_ar = np.array([complex_integration(rs, sigma2, step_size=1e-3) for rs in rho_ar])
    popt_s2, _ = curve_fit(expression_2, rho_ar[1:], integration_ar[1:])
    A, B, C = popt_s2
    print(f'var={sigma2}, A={A}, B={B}, C={C}')
    A_ar.append(A), B_ar.append(B), C_ar.append(C)



# We save the curves in a csv
df = pd.DataFrame()
df['Variance'] = variances
df['A'] = A_ar
df['B'] = B_ar
df['C'] = C_ar

df.to_csv('Fits/fits_5.csv', index=False)

"""





# =======================================================================
#                   VARIATION OF PARAMETERS A, B, C
# =======================================================================


df_fit_4 = pd.read_csv('Fits/fits_4.csv')
variances_4 = df_fit_4['Variance'].values
A_ar_4 = df_fit_4['A'].values
B_ar_4 = df_fit_4['B'].values
C_ar_4 = df_fit_4['C'].values

df_fit_5 = pd.read_csv('Fits/fits_5.csv')
variances_5 = df_fit_5['Variance'].values
A_ar_5 = df_fit_5['A'].values
B_ar_5 = df_fit_5['B'].values
C_ar_5 = df_fit_5['C'].values

variances = np.concatenate((variances_4, variances_5))
A_ar = np.concatenate((A_ar_4, A_ar_5))
B_ar = np.concatenate((B_ar_4, B_ar_5))
C_ar = np.concatenate((C_ar_4, C_ar_5))




def modelo_A(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * np.exp(-a2*x) + a3 * x * np.exp(-a4*x) + a5 * x**2 * np.exp(-a6*x)

def modelo_B(x, b0, b1, b2, b3, b4):
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4*x)

def modelo_C(x, c0, c1, c2, c3, c4):
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4*x)

popt_A, _ = curve_fit(modelo_A, variances, A_ar)
fit_A = modelo_A(variances, *popt_A)

popt_B, _ = curve_fit(modelo_B, variances, B_ar)
fit_B = modelo_B(variances, *popt_B)

popt_C, _ = curve_fit(modelo_C, variances, C_ar)
fit_C = modelo_C(variances, *popt_C)

a0, a1, a2, a3, a4, a5, a6 = popt_A
b0, b1, b2, b3, b4 = popt_B
c0, c1, c2, c3, c4 = popt_C

print(f'a0={a0}, a1={a1}, a2={a2}, a3={a3}, a4={a4}, a5={a5}, a6={a6}')
print(f'b0={b0}, b1={b1}, b2={b2}, b3={b3}, b4={b4}')
print(f'c0={c0}, c1={c1}, c2={c2}, c3={c3}, c4={c4}')



fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

ax.plot(variances, A_ar, linestyle='-', color='b', label='A')
ax.plot(variances, B_ar, linestyle='-', color='m', label='B')
ax.plot(variances, C_ar, linestyle='-', color='g', label='C')

ax.plot(variances, fit_A, linestyle='-.', label='A fit')
ax.plot(variances, fit_B, linestyle='-.', label='B fit')
ax.plot(variances, fit_C, linestyle='-.', label='C fit')

ax.set_ylabel(r'Parameters')
ax.set_xlabel(r'$\sigma^2$')
ax.legend()

plt.show()

