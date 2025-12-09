import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Bernardeu
df_mis_datos = pd.read_csv('Data/curvas.csv')
data_bernardeu_lo = np.loadtxt('Data/LO.csv', delimiter=',')
data_bernardeu_nnlo = np.loadtxt('Data/NNLO.csv', delimiter=',')

mi_rho = df_mis_datos['Density'].values
mi_prob_rho = df_mis_datos['Numerical_integration'].values


rhoc = 2.57107
lamdac = 0.729487
phic = 0.910978


y   = - np.log(mi_prob_rho[1:] / mi_rho[1:])
p2  = np.polyfit(mi_rho[1:] - rhoc, y, 2)
p3  = np.polyfit(mi_rho[1:] - rhoc, y, 3)
p4  = np.polyfit(mi_rho[1:] - rhoc, y, 4)
p10 = np.polyfit(mi_rho[1:] - rhoc, y, 10)

fit_2  = p2[0] * (mi_rho - rhoc)**2 + p2[1] * (mi_rho - rhoc) + p2[2]
fit_3  = p3[0] * (mi_rho - rhoc)**3 + p3[1] * (mi_rho - rhoc)**2 + p3[2] * (mi_rho - rhoc) + p3[3]
fit_4  = p4[0] * (mi_rho - rhoc)**4 + p4[1] * (mi_rho - rhoc)**3 + p4[2] * (mi_rho - rhoc)**2 + p4[3] * (mi_rho - rhoc) + p4[4]
fit_10 = p10[0] * (mi_rho - rhoc)**10 + p10[1] * (mi_rho - rhoc)**9 + p10[2] * (mi_rho - rhoc)**8 + p10[3] * (mi_rho - rhoc)**7   \
         + p10[4] * (mi_rho - rhoc)**6 + p10[5] * (mi_rho - rhoc)**5 + p10[6] * (mi_rho - rhoc)**4 + p10[7] * (mi_rho - rhoc)**3  \
         + p10[8] * (mi_rho - rhoc)**2 + p10[9] * (mi_rho - rhoc) + p10[10]


y = np.log(mi_prob_rho[1:] / mi_rho[1:])
x = mi_rho[1:] - rhoc

# Modelos
def poly3(x, a3, a1, a0):
    return -(a3 * x**3 + a1 * x + a0)

def poly4(x, a4, a3, a1, a0):
    return -(a4*x**4 + a3*x**3 + a1*x + a0)

def poly10(x, a10, a9, a8, a7, a6, a5, a4, a3, a1, a0):
    return -(a10*x**10 + a9*x**9 + a8*x**8 + a7*x**7 + a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a1*x + a0)

# Polinomios extendidos
def poly20(x, *a):
    # a = [a20, a19, ..., a1, a0] (sin término cuadrático)
    poly = 0
    for i, coef in enumerate(a):
        power = 20 - i if i < 20 else 0  # último es a0
        if power != 2:  # eliminar término cuadrático
            poly += coef * x**power
    return -poly

def poly30(x, *a):
    poly = 0
    for i, coef in enumerate(a):
        power = 30 - i if i < 30 else 0
        if power != 2:
            poly += coef * x**power
    return -poly


# Fit using Eq. (45)
def expression_1(x, A, B, C, D): 
    return np.exp(phic - lamdac * x) * ( A*(x-rhoc)**(-5/2) + B*(x-rhoc)**(-7/2) + C*(x-rhoc)**(-9/2) + D*(x-rhoc)**(-11/2)  )

# Fit using Eq. (46)
def expression_2(x, A, B, C): 
    return A * np.exp(phic -lamdac * x) * (x + B + C/x)**(-5/2)

# Ajustes
popt_1, _ = curve_fit(poly3, x, y)
popt_2, _ = curve_fit(poly4, x, y)
popt_3, _ = curve_fit(poly10, x, y)

# Para 20 y 30: inicializa parámetros con ceros o pequeños valores
popt_4, _ = curve_fit(poly20, x, y, p0=np.zeros(21))  # 21 coeficientes
popt_5, _ = curve_fit(poly30, x, y, p0=np.zeros(31))  # 31 coeficientes

# Special
popt_s1, _ = curve_fit(expression_1, mi_rho[1:], mi_prob_rho[1:] / mi_rho[1:])
popt_s2, _ = curve_fit(expression_2, mi_rho[1:], mi_prob_rho[1:] / mi_rho[1:])


# Reconstrucción de las PDF
fit3  = np.exp(poly3(mi_rho - rhoc, *popt_1)) * mi_rho
fit4  = np.exp(poly4(mi_rho - rhoc, *popt_2)) * mi_rho
fit10 = np.exp(poly10(mi_rho - rhoc, *popt_3)) * mi_rho
fit20 = np.exp(poly20(mi_rho - rhoc, *popt_4)) * mi_rho
fit30 = np.exp(poly30(mi_rho - rhoc, *popt_5)) * mi_rho

fit_s1 = expression_1(mi_rho, *popt_s1) * mi_rho
fit_s2 = expression_2(mi_rho, *popt_s2) * mi_rho


# Plots
plt.plot(mi_rho, mi_prob_rho, label = 'Numerical integration')
#plt.plot(mi_rho, fit3,  linestyle='-.', label='Grado 3')
#plt.plot(mi_rho, fit4,  linestyle='-.', label='Grado 4')
#plt.plot(mi_rho, fit10, linestyle='-.', label='Grado 10')
#plt.plot(mi_rho, fit20, linestyle='-.', label='Grado 20')
plt.plot(mi_rho, fit30, linestyle='-.', label='Grado 30')


#plt.plot(mi_rho, np.exp(- fit_2)  * mi_rho, linestyle = ':', label='Grado 2')
#plt.plot(mi_rho, np.exp(- fit_3)  * mi_rho, linestyle = ':', label='Grado 3')
#plt.plot(mi_rho, np.exp(- fit_4)  * mi_rho, linestyle = ':', label='Grado 4')
#plt.plot(mi_rho, np.exp(- fit_10) * mi_rho, linestyle = ':', label='Grado 10')

plt.plot(mi_rho, fit_s1, linestyle=':', label='Eq. (45)')
plt.plot(mi_rho, fit_s2, linestyle=':', label='Eq. (46)')


plt.yscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$P(\rho)$')
plt.tight_layout()




print(*popt_s2)

def area_under_curve(rho, prob):
    integral = 0 
    for i in np.arange(1, len(prob)):
        y_prev, y = prob[i-1], prob[i]
        drho = rho[i] - rho[i-1]
        integral += 0.5 * (y_prev + y) * drho
    
    return integral

print(area_under_curve(mi_rho[1:], mi_prob_rho[1:] / mi_rho[1:]))

print(area_under_curve(mi_rho, fit30))


A, B, C = popt_s2
print(f'A={A}, B={B}, C={C}')



print(area_under_curve(mi_rho, fit_s2))

plt.plot(mi_rho, fit_s2, linestyle='-', label='Pumba')
plt.legend()
plt.show()