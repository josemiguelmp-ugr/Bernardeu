import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

# Model parameters
nu  = 21/13.        # Fit in lambda vs rho approximately for nu=1.36
n   = -1.5
alpha = (n + 3) / 3


# Values in the critical point from Mathematica (maximum of Fig. 1)
rhoc = 2.57107


# Equation (32) from Bernardeu
def Psi(rho, var):
    tau = nu * (1 - rho**(-1/nu))             # Eq. (13)
    return tau**2 * rho**alpha / (2 * var)


# Mathematical derivatives of Psi wrt rho (from Mathematica)
def dPsi_drho(rho, var):
    term1 = nu * rho**(alpha - 1 - 1/nu) * (1 - rho**(-1/nu)) / var 
    term2 = nu**2 * alpha * rho**(alpha-1) * (1 - rho**(-1/nu))**2 / (2*var)
    return term1 + term2


def dPsi_drho_2(rho, var):
    term1 = rho**(-2 + alpha - 2/nu) / var
    term2 = alpha * nu * rho**(-2 + alpha - 1/nu) * (1 - rho**(-1/nu)) / var
    term3 = (-1 + alpha - 1/nu) * nu * rho**(-2 + alpha - 1/nu) * (1-rho**(-1/nu)) / var
    term4 = (alpha - 1) * alpha * nu**2 * rho**(alpha - 2) * (1 - rho**(-1/nu))**2 / (2*var)
    return term1 + term2 + term3 + term4



# ====================================================================
#                   EXACT NUMERICAL INTEGRATION
# ====================================================================

def build_rho_contour(rho_hat, var, step_size=0.04):
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

    F = dPsi_drho(rho_curr, var) * (rho_curr - rho_hat) - Psi(rho_curr, var)

    while np.real(F)>-50:                     # Tenemos Im(F)=0, por lo que el integrando es exp(-Re(F)), donde Re(F) va creciendo a cada paso. Cuando el exponente sea muy negativo, la exponencial tenderá a cero
        Psi_dd = dPsi_drho_2(rho_curr, var)   # Psi''(rho)
        delta = rho_curr - rho_hat            # (rho - rho_hat)
        theta = - np.angle(Psi_dd * delta)
        modulus = step_size

        delta_rho = modulus * np.exp(1j * theta)
        rho_curr = rho_curr - delta_rho
        rho_path.append(rho_curr)

        F = dPsi_drho(rho_curr, var)*(rho_curr - rho_hat) - Psi(rho_curr, var)

    rho_contour = rho_path
    rho_contour = np.array(rho_contour, dtype=complex)
    return rho_contour


def complex_integration(rho_hat, var, step_size=1e-3):
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
        Resultado numérico de la integral, PDF exacta
    """
    rho_path = build_rho_contour(rho_hat, var, step_size)
    lam_path = dPsi_drho(rho_path, var)
    
    integral = 0
    for i in range(1, len(rho_path)):
        rho_prev, lam_prev = rho_path[i-1], lam_path[i-1]
        rho, lam = rho_path[i], lam_path[i]

        exp_prev = np.exp( lam_prev * (rho_prev - rho_hat) - Psi(rho_prev, var) )
        exp_curr = np.exp( lam * (rho - rho_hat) - Psi(rho, var) )

        dlambda = lam - lam_prev
        
        # Regla del trapecio
        integral += 0.5 * (exp_prev + exp_curr) * dlambda
    
    result = 2 * np.real(integral / (2j * np.pi))
    
    # El contorno de integración está compuesto por el contorno rho_path y por su conjugado (donde también tenemos Im[exponente]=0)
    # Como las integrales en ambos contornos son iguales, aparece un factor 2 multiplicativo
    # Ver la figura Conjugate_contour

    return result




# Calcula el área bajo la curva
def area_under_curve(rho, prob):
    integral = 0 
    for i in np.arange(1, len(prob)):
        y_prev, y = prob[i-1], prob[i]
        drho = rho[i] - rho[i-1]
        integral += 0.5 * (y_prev + y) * drho
    
    return integral


# Pequeñas marcas en el eje Y, en escala logarítmica
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
