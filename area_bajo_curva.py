import numpy as np
import matplotlib.pylab as plt
from functions import area_under_curve

file = open('resultados.txt', 'w')
file.write('var \t Integral \t A')

def modelo_A(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * np.exp(- a2 * x) + a3 * x * np.exp(- a4 * x) + a5 * x**2 * np.exp(- a6 * x)

def modelo_B(x, b0, b1, b2, b3, b4):
    return b0 + b1 * np.exp(- b2 * x) + b3 * x * np.exp(-b4 * x)

def modelo_C(x, c0, c1, c2, c3, c4):
    return c0 + c1 * np.exp(- c2 * x) + c3 * x * np.exp(-c4 * x)


def prob_fit(delta, var):
    lamdac =  0.328269 / var
    phic   =  0.434063 / var

    a0, a1, a2, a3, a4, a5, a6 = 0.04695910742441713, -0.6642049938350947, 2.4353359038289777, 3.6354491307579315, 1.5181112635331262, 0.17469598137756703, 0.6214850573229408
    b0, b1, b2, b3, b4 = 0.0979931675570106, 3.1972055079415354, 3.9083535195901087, 0.35435722761935323, 0.5201525548415581
    c0, c1, c2, c3, c4 = 0.0023928609265879866, 1.7761815559139071, 3.632416222582498, 0.4986992331769517, 1.650113588148832

    A = modelo_A(var, a0, a1, a2, a3, a4, a5, a6)
    B = modelo_B(var, b0, b1, b2, b3, b4)
    C = modelo_C(var, c0, c1, c2, c3, c4)

    # Podemos quitar o poner la A (cuya función es normalizar la PDF)
    return np.exp(phic - lamdac * delta) * (delta + B + C / delta)**(-5/2)

var_list = np.linspace(1e-3, 20, 501)
delta_list = np.arange(1e-3, 1000, 0.1)
# Aquí la normalización no se arregla solo con poner 0.001 en vez de 0.1 en el step

areas = np.zeros_like(var_list)
for i, var in enumerate(var_list):
    # Calcular la probabilidad para todos los delta
    prob = prob_fit(delta_list, var)
    
    areas[i] = area_under_curve(delta_list, prob)
    file.write('%e %e %e\n' % (var, areas[i], 1/areas[i]))

print(var_list)
print(areas)
print(1/areas)

file.close()