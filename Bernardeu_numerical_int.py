import numpy as np
import matplotlib.pyplot as plt

# Parámetros cosmológicos / modelo
nu = 21/13
var = 0.45
alpha = 0.5

# Basic functions
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


# Critical values 
rho = np.arange(0.7, 30, 0.1)
lambdas = dPsi_drho(rho)

# Critical values
index_c = np.argmax(lambdas)
rhoc_py, lambdac_py = rho[index_c], lambdas[index_c]
phic_py = lambdac_py * rhoc_py - Psi(rhoc_py)
print('Critical values\n')
print(f'rho_c = {rhoc_py}, lambda_c = {lambdac_py}, phi_c = {phic_py}')

rhoc, lamdac, phic = rhoc_py, lambdac_py, phic_py


# ====================================================================
# 2. CONSTRUCCIÓN DEL CONTORNO
# ====================================================================

# --- Construye el contorno en rho ---
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

    while np.real(F)>-50:                     # Tenemos Im(F)=0, por lo que el integrando es exp(-Re(F)), donde Re(F) va creciendo a cada paso. Cuando el exponente sea muy negativo, la exponencial tenderá a cero
        Psi_dd = dPsi_drho_2(rho_curr)        # Psi''(rho)
        delta = rho_curr - rho_hat            # (rho - rho_hat)
        theta = - np.angle(Psi_dd * delta)

        modulus = step_size

        delta_rho = modulus * np.exp(1j * theta)
        rho_curr = rho_curr - delta_rho
        rho_path.append(rho_curr)

        F = dPsi_drho(rho_curr) * (rho_curr - rho_hat) - Psi(rho_curr)

    rho_contour = rho_path
    rho_contour = np.array(rho_contour, dtype=complex)
    return rho_contour


def lambda_contour(rho_contour):
    lambdas = dPsi_drho(rho_contour)
    return lambdas


rho_hat = 1.1
rho_path = build_rho_contour(rho_hat, step_size=1e-5)
lam_path = lambda_contour(rho_path)

"""
# We plot the contour in the complex plane for rho
real_parts, imag_parts = [], []
for i in range(0, len(rho_path)):
    real_parts.append(rho_path[i].real), imag_parts.append(rho_path[i].imag)

plt.plot(real_parts, imag_parts)
plt.xlabel(r'Re[$\rho$]')
plt.ylabel(r'Im[$\rho$]')
plt.title('Example of numerical integration contour')

plt.show()
"""

F_ar = dPsi_drho(rho_path) * (rho_path - rho_hat) - Psi(rho_path)
ReF, ImF = np.real(F_ar), np.imag(F_ar)

rho_path_conjugate = np.conjugate(rho_path)
F_conjugate_ar = dPsi_drho(rho_path_conjugate) * (rho_path_conjugate - rho_hat) - Psi(rho_path_conjugate)
ReF_conjugate, ImF_conjugate = np.real(F_conjugate_ar), np.imag(F_conjugate_ar)

"""
fig, ax = plt.subplots(3, 1)
fig.set_size_inches(10, 6)
ax[0].plot(ImF, label='Original contour', color='r', linestyle='solid')
ax[0].plot(ImF_conjugate, label='Conjugate contour', linestyle='-.', color='b')
ax[0].set_ylabel(r'Im[$F$]')
ax[0].legend()

ax[1].plot(abs(ImF), label='Original contour', color='r', linestyle='solid')
ax[1].plot(abs(ImF_conjugate), label='Conjugate contour', linestyle='-.', color='b')
ax[1].set_ylabel(r'abs(Im[$F$])')
ax[1].legend()


ax[2].plot(ReF, label='Original contour', color='r', linestyle='solid')
ax[2].plot(ReF_conjugate, label='Conjugate contour', linestyle='-.', color='b')
ax[2].set_ylabel(r'Re[$F$]')
ax[2].legend()
fig.show()
#plt.savefig('Figures/Conjugate_contour.png')
plt.show()
plt.close()
"""



def complex_integration(rho_hat, step=1e-3):
    rho_path_original = build_rho_contour(rho_hat, step)
    rho_path_conjugate = np.conjugate(rho_path_original)
    rho_path = np.concatenate((rho_path_conjugate[::-1], rho_path_original))
    lam_path = lambda_contour(rho_path)
    
    # El contorno de integración está compuesto por el contorno rho_path y por su conjugado (donde también tenemos Im[exponente]=0)
    # Como las integrales en ambos contornos son iguales, aparece un factor 2 multiplicativo para la parte real, y las partes imaginarias se cancelan
    # Ver la figura Conjugate_contour
    
    integral = 0
    
    # Integración numérica, con el método Runge-Kutta 4
    for i in range(len(rho_path)):
        rho_i, lam_i = rho_path[i-1], lam_path[i-1]
        rho_f, lam_f = rho_path[i], lam_path[i]

        # Paso en el parámetro (lambda)
        dlambda = lam_f - lam_i

        # Definimos función integrando
        def integrand(rho, lam):
            return np.exp(lam * (rho - rho_hat) - Psi(rho))

        # Regla del trapecio
        integral += 0.5 * ( integrand(rho_i, lam_i) + integrand(rho_f, lam_f) ) * dlambda

    result = integral / (2j * np.pi)  
    
    return result

rho_ar = np.arange(0, 13, 0.05)
integration = []
for rs in rho_ar:
    integration.append(complex_integration(rs, 1e-3))


integration_ar = np.array(integration)
plt.figure()
plt.plot(rho_ar, rho_ar*integration_ar, label='My integration')

# Bernardeu curves
data_bernardeu_int = np.loadtxt('Data/numerical_integration.csv', delimiter=',')
rho = data_bernardeu_int[:, 0]
prob_rho = data_bernardeu_int[:, 1]
plt.plot(rho, prob_rho, label= 'Bernardeu integration')


plt.yscale('log')
plt.ylim(1e-5, 1)
plt.ylabel(r'$\rho P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend()
plt.show()