import numpy as np 
import pandas as pd
import matplotlib.pylab as plt

# Model parameters
nu  = 21/13.        # Fit in lambda vs rho approximately for nu=1.36
var = 0.45          # Variance at radius R
n   = -1.5
alpha = (n + 3) / 3

"""
# Values in the critical point from Mathematica (maximum of Fig. 1)
rhoc = 2.57
lamdac = 0.73
phic = 0.91
"""

# Equation (32) from Bernardeu
def Psi(rho):
    tau = nu * (1 - rho**(-1/nu))
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



rho = np.arange(0.7, 30, 0.1)
lambdas = dPsi_drho(rho)

# Critical values
index_c = np.argmax(lambdas)
rhoc_py, lambdac_py = rho[index_c], lambdas[index_c]
phic_py = lambdac_py * rhoc_py - Psi(rhoc_py)
print('Critical values\n')
print(f'rho_c = {rhoc_py}, lambda_c = {lambdac_py}, phi_c = {phic_py}')

#rhoc, lamdac, phic = rhoc_py, lambdac_py, phic_py
rhoc, lamdac = 0.6, 1
phic = rhoc * lamdac - Psi(rhoc)

"""
# Gráfica de Lambda vs rho, donde podemos ver el punto crítico (máximo)
plt.plot(rho, lambdas)
plt.axhline(0.4, color='r')
plt.axvline(rhoc_py, linestyle='dashed', color='gray')
plt.ylim(-0.8, 0.8)
plt.xscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Psi\'(\rho)$')

# Comparación con la curva de Bernardeu
data_bernardeu_lambda = np.loadtxt('Figures/lambda_curve.csv', delimiter=',')

rho_ber = data_bernardeu_lambda[:, 0]
lambda_ber = data_bernardeu_lambda[:, 1]

plt.plot(rho_ber, lambda_ber, label='Bernardeu')
plt.legend()

#plt.savefig('Figures/Lambda_rho.png')
plt.show()
"""

# ========================================================================
# 1. SADDLE POINT APPROXIMATION
# ========================================================================

# Equation (44) from Bernardeu
def prob_saddle(rho):
    """
    Following the Eq. (44) of Bernardeu et al.
    It is valid as long as the expression that appears in the square root is positive, i.e. rho < rhoc . When this condi-
    tion is not satisfied, the singular behavior of phi near λc dominates the integral in the complex plane.
    """    
    return np.exp( - Psi(rho) ) * np.sqrt(dPsi_drho_2(rho)) / np.sqrt(2 * np.pi)


# ========================================================================
# 2. TAYLOR EXPANSION OF THE EXPONENTIAL IN THE INTEGRAL
# ========================================================================

# Equation (45) from Bernardeu
def prob_exact(r):
    exponential = np.exp(0.964585-0.729487*r)
    term1 =   1.20388 / (r - 2.57107)**(5/2)
    term2 = - 3.80256 / (r - 2.57107)**(7/2)
    term3 = - 15.9587 / (r - 2.57107)**(9/2)

    p1 = exponential * term1
    p2 = exponential * ( term1 + term2 )
    p3 = exponential * ( term1 + term2 + term3 )
    return p1, p2, p3




# ====================================================================
# 3. CONSTRUCCIÓN DEL CONTORNO
# ====================================================================


# We build the integration contour for rho
def build_rho_contour(rho_hat, step_size=0.04):
    """
    Construye un contorno (lista de rho complejos) tal que Im[F(rho)] constante,
    donde F(rho) = Psi'(rho)*(rho - rho_hat) - Psi(rho).
    
    rho_start: punto inicial (rho_s)
    """

    rho_start = rho_hat if rho_hat < rhoc else rhoc     # en región regular; si no, pon rho_start=rho_c (crítico)
    rho_path = [rho_start]
    rho_curr = rho_start

    F = dPsi_drho(rho_curr) * (rho_curr - rho_hat) - Psi(rho_curr)

    while np.real(F)>-20:                    # Tenemos Im(F)=0, por lo que el integrando es exp(-Re(F)), donde Re(F) va creciendo a cada paso. Cuando el exponente sea muy negativo, la exponencial tenderá a cero
        Psi_dd = dPsi_drho_2(rho_curr)       # Psi''(rho)
        delta = rho_curr - rho_hat           # (rho - rho_hat)
        theta = - np.angle(Psi_dd * delta)

        #modulus = s_finder(rho_curr, rho_hat, theta)
        modulus = step_size

        delta_rho = modulus * np.exp(1j * theta)
        rho_curr = rho_curr - delta_rho
        rho_path.append(rho_curr)

        F = dPsi_drho(rho_curr)*(rho_curr - rho_hat) - Psi(rho_curr)

    rho_contour = rho_path
    rho_contour = np.array(rho_contour, dtype=complex)
    return rho_contour


def lambda_contour(rho_contour):
    lambdas = dPsi_drho(rho_contour)
    return lambdas


# ====================================================================
# 4. EJEMPLO DE UN CONTORNO PARA UN RHO_HAT DETERMINADO
# ====================================================================

rho_hat = 1.6
rho_path = build_rho_contour(rho_hat, step_size=1e-5)
lam_path = lambda_contour(rho_path)

# We plot the contour in the complex plane for rho
real_parts, imag_parts = [], []
for i in range(0, len(rho_path)):
    real_parts.append(rho_path[i].real), imag_parts.append(rho_path[i].imag)

plt.plot(real_parts, imag_parts)
plt.xlabel(r'Re[$\rho$]')
plt.ylabel(r'Im[$\rho$]')
plt.title('Example of numerical integration contour')

#plt.savefig('Figures/Contour_complex_plane.png')
plt.show()


# ====================================================================
# 5. INTEGRACIÓN NUMÉRICA PARA UNA LISTA DE RHO_HAT
# ====================================================================

def complex_integration(rho_hat, step=1e-3):
    rho_path = build_rho_contour(rho_hat, step)
    lam_path = lambda_contour(rho_path)
    
    integral = 0
    for i in range(1, len(rho_path)):
        rho_prev, lam_prev = rho_path[i-1], lam_path[i-1]
        rho, lam = rho_path[i], lam_path[i]

        exp_prev = np.exp( lam_prev * (rho_prev - rho_hat) - Psi(rho_prev) )
        exp_curr = np.exp( lam * (rho - rho_hat) - Psi(rho) )

        dlambda = lam - lam_prev
        # Regla del trapecio
        integral += 0.5 * (exp_prev + exp_curr) * dlambda

    return integral / (2j * np.pi)

rho_ar = np.arange(0, 13, 0.05)
integration = []
for rs in rho_ar:
    integration.append(complex_integration(rs, 1e-3))

integration_ar = np.array(integration)


# =======================================================================
# 6. GRÁFICA FINAL
# =======================================================================

rho = np.arange(0, 13, 0.05)
prob1, prob2, prob3 = prob_exact(rho)

plt.plot(rho, rho*prob1, label='Leading order')
plt.plot(rho, rho*prob2, label='Next-to-leading order')
plt.plot(rho, rho*prob3, label='Next-to-next-to-leading order')
plt.plot(rho, rho*prob_saddle(rho), label='Saddle point')
plt.plot(rho, np.real(rho*integration_ar), label='Numerical integration')

plt.yscale('log')
plt.ylim(1e-5, 1)
plt.ylabel(r'$\rho P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend()

#plt.savefig('Figures/PDF_rho.png')
plt.show()


# =======================================================================
# 7. DATAFRAME
# =======================================================================

df = pd.DataFrame()
df['Density'] = rho
df['Numerical_integration'] = rho * integration_ar
df['Saddle_point'] = prob_saddle(rho)
df['NNLO'] = rho * prob3

df.to_csv('Figures/curvas.csv', index=False)
