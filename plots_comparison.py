import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import csv



data_bernardeu_int = np.loadtxt('Figures/numerical_integration.csv', delimiter=',')
data_bernardeu_sad = np.loadtxt('Figures/saddle_point.csv', delimiter=',')
df_mis_datos = pd.read_csv('Figures/curvas.csv')

rho = data_bernardeu_int[:, 0]
prob_rho = data_bernardeu_int[:, 1]

rho_saddle = data_bernardeu_sad[:, 0]
prob_rho_saddle = data_bernardeu_sad[:, 1]

mi_rho = df_mis_datos['Density'].values
mi_prob_rho = df_mis_datos['Numerical_integration'].apply(complex).values
mi_prob_rho_sad = df_mis_datos['Saddle_point'].values

#plt.plot(rho, prob_rho, label= 'Bernardeu integration')
#plt.plot(mi_rho, 2.2*mi_prob_rho, label = '2.2 * My integration')
plt.plot(rho_saddle, prob_rho_saddle, label= 'Bernardeu saddle approx.')
plt.plot(mi_rho[:len(mi_prob_rho_sad)], mi_prob_rho_sad, label='My saddle approx.')

plt.yscale('log')
plt.ylim(1e-5, 1)
plt.xlim(0, 13)
plt.legend(loc='best')
plt.show()

