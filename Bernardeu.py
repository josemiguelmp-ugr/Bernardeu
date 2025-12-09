import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

# Model parameters
nu  = 21/13.        # Fit in lambda vs rho approximately for nu=1.36
var = 0.45          # Variance at radius R
n   = -1.5
alpha = (n + 3) / 3


# Values in the critical point from Mathematica (maximum of Fig. 1)
rhoc   = 2.57107
lamdac = 0.729487
phic   = 0.910978


# Equation (32) from Bernardeu
def Psi(rho):
    tau = nu * (1 - rho**(-1/nu))             # Eq. (13)
    return tau**2 * rho**alpha / (2 * var)


# Mathematical derivatives of Psi wrt rho (from Mathematica)
def dPsi_drho(rho):
    term1 = nu * rho**(alpha - 1 - 1/nu) * (1 - rho**(-1/nu)) / var 
    term2 = nu**2 * alpha * rho**(alpha-1) * (1 - rho**(-1/nu))**2 / (2*var)
    return term1 + term2


def dPsi_drho_2(rho):
    term1 = rho**(-2 + alpha - 2/nu) / var
    term2 = alpha * nu * rho**(-2 + alpha - 1/nu) * (1 - rho**(-1/nu)) / var
    term3 = (-1 + alpha - 1/nu) * nu * rho**(-2 + alpha - 1/nu) * (1-rho**(-1/nu)) / var
    term4 = (alpha - 1) * alpha * nu**2 * rho**(alpha - 2) * (1 - rho**(-1/nu))**2 / (2*var)
    return term1 + term2 + term3 + term4



# Critical values
rho = np.arange(0.7, 30, 0.05)
lambdas = dPsi_drho(rho)
index_c = np.argmax(lambdas)
rhoc_py, lambdac_py = rho[index_c], lambdas[index_c]
phic_py = lambdac_py * rhoc_py - Psi(rhoc_py)
print('Critical values\n')
print(f'rho_c = {rhoc_py}, lambda_c = {lambdac_py}, psi_c = {Psi(rhoc_py)}, phi_c = {phic_py}')

#rhoc, lamdac, phic = rhoc_py, lambdac_py, phic_py


# Gráfica de Lambda vs rho, donde podemos ver el punto crítico (máximo)
plt.plot(rho, lambdas, label='Mine')
plt.axhline(0.4, color='r')
plt.axvline(rhoc_py, linestyle='dashed', color='gray')
plt.ylim(-0.8, 0.8)
plt.xscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Psi\'(\rho)$')

# Comparación con la curva de Bernardeu
data_bernardeu_lambda = np.loadtxt('Data/lambda_curve.csv', delimiter=',')

rho_ber = data_bernardeu_lambda[:, 0]
lambda_ber = data_bernardeu_lambda[:, 1]

plt.plot(rho_ber, lambda_ber, label='Bernardeu')
plt.legend()

#plt.savefig('Figures/Lambda_rho.png')
plt.show()
plt.close()


# ========================================================================
# 1. SADDLE POINT APPROXIMATION
# ========================================================================

# Equation (44) from Bernardeu
def prob_saddle(rho):
    """
    It is valid as long as the expression that appears in the square root is positive, i.e. rho < rhoc . When this condi-
    tion is not satisfied, the singular behavior of phi near λc dominates the integral in the complex plane.
    """    
    return np.exp( - Psi(rho) ) * np.sqrt(dPsi_drho_2(rho)) / np.sqrt(2 * np.pi)


# ========================================================================
# 2. TAYLOR EXPANSION OF THE EXPONENTIAL IN THE INTEGRAL
# ========================================================================

# Equation (45) from Bernardeu
def prob_aprox(r):
    exponential = np.exp(0.964585 - 0.729487*r)
    term1 =   1.20388 / (r - 2.57107)**(5/2)
    term2 = - 3.80256 / (r - 2.57107)**(7/2)
    term3 = - 15.9587 / (r - 2.57107)**(9/2)

    p1 = exponential * term1
    p2 = exponential * ( term1 + term2 )
    p3 = exponential * ( term1 + term2 + term3 )
    return p1, p2, p3


# Equation (46) from Bernardeu
def prob_aprox_2(r):
    exponential = np.exp(0.964585 - 0.729487*r) * 1.20388
    term1 = r
    term2 = - 1.30763
    term3 = 8.09591 / r

    p1 = exponential / (term1)**(5/2)
    p2 = exponential / (term1 + term2)**(5/2)
    p3 = exponential / (term1 + term2 + term3)**(5/2)

    return p1, p2, p3



# ====================================================================
# 3. CONSTRUCCIÓN DEL CONTORNO
# ====================================================================

# We build the integration contour for rho
def build_rho_contour(rho_hat, step_size=0.04):
    """
    Función que construye un contorno de integración en el plano complejo de rho para la integral P(rho_hat) de la Ec. (B1) de Bernardeu.
    Para construirlo, establece la condición de que Im[F(rho)] constante, donde F(rho) es el exponente de la integral, 
    i.e. F(rho) = Psi'(rho)*(rho - rho_hat) - Psi(rho).

    Parameters
    ----------
    rho_hat : float
        Densidad física para la cual integramos. Es un número real. Si rho_hat<rhoc, este será el punto de inicio del contorno en la recta real.
        En caso contrario, el punto de inicio será rhoc.
    step_size : float
        Magnitud (módulo) de cada paso utilizado para construir el contorno de integración.
    
    Return
    -------
    np.array
        Array de números complejos que componen el contorno.
    """

    rho_start = rho_hat if rho_hat < rhoc else rhoc
    rho_path = [rho_start]
    rho_curr = rho_start

    F = dPsi_drho(rho_curr) * (rho_curr - rho_hat) - Psi(rho_curr)

    while np.real(F)>-50:                     # Tenemos Im(F)=0, por lo que el integrando es exp(-Re(F)), donde Re(F) va creciendo a cada paso. Cuando el exponente sea muy negativo, la exponencial tenderá a cero
        Psi_dd = dPsi_drho_2(rho_curr)        # Psi''(rho)
        delta = rho_curr - rho_hat            # (rho - rho_hat)
        theta = - np.angle(Psi_dd * delta)
        modulus = step_size

        delta_rho = modulus * np.exp(1j * theta)
        rho_curr = rho_curr - delta_rho
        rho_path.append(rho_curr)

        F = dPsi_drho(rho_curr)*(rho_curr - rho_hat) - Psi(rho_curr)

    rho_contour = rho_path
    rho_contour = np.array(rho_contour, dtype=complex)
    return rho_contour


# ====================================================================
# 4. EJEMPLO DE UN CONTORNO PARA UN RHO_HAT DETERMINADO
# ====================================================================

rho_hat = 1.6
rho_path = build_rho_contour(rho_hat, step_size=1e-5)
lam_path = dPsi_drho(rho_path)

# We plot the contour in the complex plane for rho
real_parts, imag_parts = [], []
for i in range(0, len(rho_path)):
    real_parts.append(rho_path[i].real), imag_parts.append(rho_path[i].imag)

plt.plot(real_parts, imag_parts)
plt.xlabel(r'Re[$\rho$]')
plt.ylabel(r'Im[$\rho$]')
plt.title('Example of numerical integration contour')

#plt.savefig('Figures/Contour_complex_plane.png')
#plt.show()
plt.close()


# ====================================================================
# 5. INTEGRACIÓN NUMÉRICA PARA UNA LISTA DE RHO_HAT
# ====================================================================

def complex_integration(rho_hat, step_size=1e-3):
    """
    Integrador numérico con la regla del trapecio de la ecuación (B1) en el plano complejo de rho. 
    La integral se realiza en el contorno que te devuelve la función build_rho_contour.

    Parameters
    ----------
    rho_hat : float
        Densidad física para la cual integramos. Es un número real. Si rho_hat<rhoc, este será el punto de inicio del contorno en la recta real.
        En caso contrario, el punto de inicio será rhoc.
    step_size : float
        Magnitud (módulo) de cada paso utilizado para construir el contorno de integración.
    
    Return
    -------
    float
        Resultado numérico de la integral
    """
    rho_path = build_rho_contour(rho_hat, step_size)
    lam_path = dPsi_drho(rho_path)
    
    integral = 0
    for i in range(1, len(rho_path)):
        rho_prev, lam_prev = rho_path[i-1], lam_path[i-1]
        rho, lam = rho_path[i], lam_path[i]

        exp_prev = np.exp( lam_prev * (rho_prev - rho_hat) - Psi(rho_prev) )
        exp_curr = np.exp( lam * (rho - rho_hat) - Psi(rho) )

        dlambda = lam - lam_prev
        
        # Regla del trapecio
        integral += 0.5 * (exp_prev + exp_curr) * dlambda
    
    result = 2 * np.real(integral / (2j * np.pi))
    
    # El contorno de integración está compuesto por el contorno rho_path y por su conjugado (donde también tenemos Im[exponente]=0)
    # Como las integrales en ambos contornos son iguales, aparece un factor 2 multiplicativo
    # Ver la figura Conjugate_contour

    return result

rho_ar = np.arange(0, 15, 0.05)
integration = []
for rs in rho_ar:
    integration.append(complex_integration(rs, 1e-3))

integration_ar = np.array(integration)


# =======================================================================
# 6. GRÁFICA FINAL
# =======================================================================

rho = np.arange(0, 15, 0.05)
prob1, prob2, prob3 = prob_aprox(rho)
prob1_2, prob2_2, prob3_2 = prob_aprox_2(rho)

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.plot(rho, rho*prob1, label='Leading order')
ax.plot(rho, rho*prob2, label='Next-to-leading order')
ax.plot(rho, rho*prob3, label='Next-to-next-to-leading order')
ax.plot(rho, rho*prob1_2, ':', label='Leading order (2)')
ax.plot(rho, rho*prob2_2, ':', label='Next-to-leading order (2)')
ax.plot(rho, rho*prob3_2, ':', label='Next-to-next-to-leading order (2)')
ax.plot(rho, rho*prob_saddle(rho), label='Saddle point')
ax.plot(rho, np.real(rho*integration_ar), label='Numerical integration')


ax.set_yscale('log')
ax.set_ylim(1e-6, 1)
ax.set_ylabel(r'$\rho P(\rho)$')
ax.set_xlabel(r'$\rho$')
ax.legend()

def log_formatter(y, pos):
    """
    Formateador de etiquetas para el eje Y en escala logarítmica.

    Muestra el valor en formato decimal (ej., 1, 0.1) si y >= 10^-3.
    A partir de 10^-4, usa notación de base 10 (ej., 10^-4).

    Parameters
    ----------
    y : float
        Valor de la marca del eje (e.g., 1.0, 0.1, 0.001).
    pos : int
        Posición de la marca. Requerido por FuncFormatter, pero no usado.

    Return
    -------
    str
        Etiqueta formateada (decimal o en formato LaTeX de base 10).
    """
    if y >= 1e-3:
        return f'{y:g}'
    else:
        exponent = int(np.log10(y))
        return r'$10^{{{:d}}}$'.format(exponent)

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

#plt.savefig('Figures/PDF_rho_more_curves.png')
plt.show()
plt.close()


"""
# Representamos la parte imaginaria de P(rho), que debería ser nula
plt.plot(rho, np.abs(np.imag(rho*integration_ar)))
plt.yscale('log')
plt.ylabel(r'Im[$\rho P(\rho)$]')
plt.xlabel(r'$\rho$')
plt.show()
"""

# =======================================================================
# 7. DATAFRAME
# =======================================================================

# We save the curves in a csv
df = pd.DataFrame()
df['Density'] = rho
df['Numerical_integration'] = rho * integration_ar
df['Saddle_point'] = prob_saddle(rho)
df['NNLO'] = rho * prob3

df.to_csv('Data/curvas.csv', index=False)
