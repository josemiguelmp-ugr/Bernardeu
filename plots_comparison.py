import numpy as np
import pandas as pd
import matplotlib.pylab as plt


data_bernardeu_int = np.loadtxt('Data/numerical_integration.csv', delimiter=',')
data_bernardeu_sad = np.loadtxt('Data/saddle_point.csv', delimiter=',')
data_bernardeu_lo = np.loadtxt('Data/LO.csv', delimiter=',')
data_bernardeu_nnlo = np.loadtxt('Data/NNLO.csv', delimiter=',')
df_mis_datos = pd.read_csv('Data/curvas.csv')

rho = data_bernardeu_int[:, 0]
prob_rho = data_bernardeu_int[:, 1]

rho_saddle = data_bernardeu_sad[:, 0]
prob_rho_saddle = data_bernardeu_sad[:, 1]

rho_lo = data_bernardeu_lo[:, 0]
prob_rho_lo = data_bernardeu_lo[:, 1]

rho_nnlo = data_bernardeu_nnlo[:, 0]
prob_rho_nnlo = data_bernardeu_nnlo[:, 1]

mi_rho = df_mis_datos['Density'].values
mi_prob_rho = df_mis_datos['Numerical_integration'].apply(complex).values
mi_prob_rho_sad = df_mis_datos['Saddle_point'].values
mi_prob_rho_nnlo = df_mis_datos['NNLO'].values

mi_rho_nnlo = mi_rho[len(mi_rho)-len(mi_prob_rho_nnlo):]



plt.figure()
plt.plot(rho, prob_rho/rho, label= 'Bernardeu integration')
plt.plot(mi_rho, mi_prob_rho/mi_rho, label = 'My integration')
#plt.plot(rho_saddle, prob_rho_saddle/rho_saddle, label= 'Bernardeu saddle approx.')
#plt.plot(mi_rho[:len(mi_prob_rho_sad)], mi_prob_rho_sad, label='My saddle approx.')
#plt.plot(rho_nnlo, prob_rho_nnlo, label='Bernardeu NNLO')
#plt.plot(mi_rho_nnlo, mi_prob_rho_nnlo, label='My NNLO')

plt.yscale('log')
#plt.ylim(1e-5, 1.2)
#plt.xlim(0, 13)
#plt.ylabel(r'$\rho P(\rho)$')
plt.ylabel(r'$P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend(loc='best')
#plt.savefig('Figures/Comparison_numerical_int.png')
plt.show()
plt.close()



# Approximated curve

# Parámetros cosmológicos / modelo
nu = 21/13
var = 0.47
alpha = 0.5

# Equation (45) from Bernardeu
# Introducing nu = 1.4 and remaining unperturbed the other parameters in Mathematica
def prob_exact(r):
    exponential = np.exp(0.835141 - 0.651829*r)
    term1 =   1.11647 / (r - 2.33195)**(5/2)
    term2 = - 3.40557 / (r - 2.33195)**(7/2)
    term3 = - 10.0463 / (r - 2.33195)**(9/2)

    p1 = exponential * term1
    p2 = exponential * ( term1 + term2 )
    p3 = exponential * ( term1 + term2 + term3 )
    return p1, p2, p3


rho_probs = np.arange(0, 13, 0.05)
prob1, prob2, prob3 = prob_exact(rho_probs)

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.plot(rho_probs, rho_probs*prob1, label='Leading order')
ax.plot(rho_probs, rho_probs*prob2, label='Next-to-leading order')
ax.plot(rho_probs, rho_probs*prob3, label='Next-to-next-to-leading order')
ax.plot(rho_nnlo, prob_rho_nnlo, linestyle=':', label='Bernardeu NNLO')
ax.plot(rho_lo, prob_rho_lo, linestyle=':', label='Bernardeu LO')
ax.plot(rho, prob_rho, label= 'Bernardeu integration')
ax.plot(mi_rho, mi_prob_rho, label = 'My integration')


ax.set_yscale('log')
ax.set_ylim(1e-5, 1)
ax.set_ylabel(r'$\rho P(\rho)$')
ax.set_xlabel(r'$\rho$')

ax.legend()
plt.show()

