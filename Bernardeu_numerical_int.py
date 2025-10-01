import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros cosmológicos / modelo ---
nu = 21/13
sigma0 = 0.45
alpha = 0.5

rhoc = 2.57
lamdac = 0.73
phic = 0.91

# --- Funciones básicas ---
def tau_from_rho(rho):
    return nu * (1 - rho**(-1/nu))

def sigma_r(rho):
    # Suponemos sigma constante para simplicidad
    return sigma0 * rho**(-alpha)

def Psi(rho):
    tau = tau_from_rho(rho)
    sigma = sigma_r(rho)
    return tau**2 / (2 * sigma)

# Derivada de Psi
def dPsi_drho(rho):
    term1 = nu * rho**alpha / sigma0 * (1 - rho**(-1/nu)) * rho**(-(1+nu)/nu)
    term2 = 0.5 * nu**2 * alpha / sigma0 * (1 - rho**(-1/nu))**2 * rho**(alpha-1)
    return term1 + term2


def dPsi_drho_2(rho):
    term1 = rho**(-2 + alpha - 2/nu) / sigma0
    term2 = alpha * nu * rho**(-2+alpha-1/nu) * (1-rho**(-1/nu)) / sigma0
    term3 = (-1+alpha-1/nu) * nu * rho**(-2+alpha-1/nu) * (1-rho**(-1/nu)) / sigma0
    term4 = (alpha - 1) * alpha * nu**2 * rho**(-2+alpha) * (1-rho**(-1/nu))**2 / (2*sigma0)
    return term1 + term2 + term3 + term4



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

    F = dPsi_drho(rho_curr)*(rho_curr - rho_hat) - Psi(rho_curr)

    while np.real(F)>-20:                    # Tenemos Im(F)=0, por lo que el integrando es exp(-Re(F)), donde Re(F) va creciendo a cada paso. Cuando el exponente sea muy negativo, la exponencial tenderá a cero
        Psi_dd = dPsi_drho_2(rho_curr)       # Psi''(rho)
        delta = rho_curr - rho_hat           # (rho - rho_hat)
        theta = - np.angle(Psi_dd * delta)
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


rho_hat = 1.6
rho_path = build_rho_contour(rho_hat, step_size=1e-5)
lam_path = lambda_contour(rho_path)


"""
ImF = np.imag(dPsi_drho(rho_path)*(rho_path - rho_hat) - Psi(rho_path))
plt.plot(abs(ImF))
plt.ylabel(r'Im[$F$]')
plt.show()

ReF = np.real(dPsi_drho(rho_path)*(rho_path - rho_hat) - Psi(rho_path))
plt.plot(ReF)
plt.ylabel(r'Re[$F$]')
plt.show()
"""


# We plot the contour in the complex plane for rho
real_parts, imag_parts = [], []
for i in range(0, len(rho_path)):
    real_parts.append(rho_path[i].real), imag_parts.append(rho_path[i].imag)

plt.plot(real_parts, imag_parts)
plt.xlabel(r'Re[$\rho$]')
plt.ylabel(r'Im[$\rho$]')

plt.show()







def complex_integration(rho_hat):
    rho_path = build_rho_contour(rho_hat, 1e-3)
    lam_path = lambda_contour(rho_path)
    
    integral = 0
    for i in range(1, len(rho_path)):
        rho_prev, lam_prev = rho_path[i-1], lam_path[i-1]
        rho, lam = rho_path[i], lam_path[i]

        exp_prev = np.exp(lam_prev * (rho_prev - rho_hat) - Psi(rho_prev))
        exp_curr = np.exp(lam * (rho - rho_hat) - Psi(rho))

        dlambda = lam - lam_prev
        # Regla del trapecio
        integral += 0.5 * (exp_prev + exp_curr) * dlambda

    return integral / (2j * np.pi)

rho_ar = np.arange(0, 13, 0.05)
integration = []
for rs in rho_ar:
    integration.append(complex_integration(rs))

integration_ar = np.array(integration)
plt.plot(rho_ar, rho_ar*integration_ar)
plt.yscale('log')
plt.ylim(1e-5, 1)
plt.ylabel(r'$\rho P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend()
plt.show()