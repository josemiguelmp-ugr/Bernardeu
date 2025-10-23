import numpy as np
import matplotlib.pyplot as plt

# Bernardeu
data_bernardeu_lo = np.loadtxt('Data/LO.csv', delimiter=',')
data_bernardeu_nnlo = np.loadtxt('Data/NNLO.csv', delimiter=',')

rho_lo = data_bernardeu_lo[:, 0]
prob_rho_lo = data_bernardeu_lo[:, 1]

rho_nnlo = data_bernardeu_nnlo[:, 0]
prob_rho_nnlo = data_bernardeu_nnlo[:, 1]


rho_c = 2.57107
a0, a1 = 0.964585, -0.729487
c5, c7, c9 = 1.20388, -3.80256, -15.9587  # coeficientes

# Definición de los términos
def term5(rho):
    return np.exp(a0 + a1*rho) * (c5 / (rho - rho_c)**(5/2))

def term7(rho):
    return np.exp(a0 + a1*rho) * (c7 / (rho - rho_c)**(7/2))

def term9(rho):
    return np.exp(a0 + a1*rho) * (c9 / (rho - rho_c)**(9/2))

# Sumas acumuladas
def P1(rho):  # solo 5/2
    return term5(rho)

def P2(rho):  # 5/2 + 7/2
    return term5(rho) + term7(rho)

def P3(rho):  # total
    return term5(rho) + term7(rho) + term9(rho)

# Rango de rho
rho = np.linspace(0, 14, 500)

# Evaluar
P1v, P2v, P3v = P1(rho), P2(rho), P3(rho)

# Gráfica
plt.figure(figsize=(7,5))
plt.plot(rho, P1v*rho, 'b--', label=r'$5/2$ sólo')
plt.plot(rho, P2v*rho, 'g-.', label=r'$5/2 + 7/2$')
plt.plot(rho, P3v*rho, 'k', lw=2, label=r'Total $5/2+7/2+9/2$')

plt.plot(rho_nnlo, prob_rho_nnlo, linestyle=':', label='Bernardeu NNLO')
plt.plot(rho_lo, prob_rho_lo, linestyle=':', label='Bernardeu LO')

plt.axvline(rho_c, color='gray', ls='--', label=r'$\rho_c=2.571$')
plt.yscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$P(\rho)$')
plt.title("Construcción progresiva del asintótico de $P(\\rho)$")
plt.legend()
plt.tight_layout()
plt.show()
