import numpy as np
from .errors import InvalidParametersError

def generate_theoretical_pdf_points(a: float, b: float, c: float, n_samples: int = 1000) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Generate points for the theoretical probability density function (PDF).
    This function creates a set of x-values and their corresponding PDF values
    for visualization or analysis purposes. The PDF is defined by parameters a, b, and c,
    where a and b define the support interval [a, b] and c is a point within this interval.
    Args:
        a (float): Lower bound of the support interval
        b (float): Upper bound of the support interval
        c (float): Point within the support interval (a < c < b must be satisfied)
        n_samples (int, optional): Number of points to generate. Defaults to 1000.
    Returns:
        tuple[np.ndarray[float], np.ndarray[float]]: A tuple containing:
            - x_values: Array of equally spaced points from a to b
            - pdf_values: Array of corresponding PDF values at each x point
    Raises:
        InvalidParametersError: If a >= c, c >= b, or n_samples <= 0
    """
    
    if not a < c < b or not n_samples > 0:
        raise InvalidParametersError
    
    x_values = np.linspace(a, b, n_samples)
    pdf_values = [get_pdf_val_from_x(x, a, b, c) for x in x_values]
    return x_values, pdf_values

def get_pdf_val_from_x(x: float, a: float, b: float, c: float) -> float:
    if x < c:
        return (2 * (x - a)) / ((b - a) * (c - a))
    else:
        return (2 * (b - x)) / ((b - a) * (b - c))
    

def generate_theoretical_pareto_pdf_points(xm: float, alpha: int, n_samples: int = 1000) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Generate points for plotting the theoretical Pareto probability density function.
    
    Args:
        xm (float): The scale parameter (minimum possible value) of the Pareto distribution.
        alpha (int): The shape parameter of the Pareto distribution.
        n_samples (int, optional): The number of points to generate. Defaults to 1000.
    
    Returns:
        tuple[np.ndarray [float], np.ndarray[float]]: A tuple containing:
            - x_values: Array of x values from xm to 10*xm
            - pdf_values: Array of corresponding PDF values calculated using get_pareto_pdf_val_from_x()
    
    Note:
        The function generates points in the range [xm, 10*xm] and calculates the 
        corresponding PDF values using the get_pareto_pdf_val_from_x function.
    """
    x_values = np.linspace(xm, 10 * xm, n_samples)
    pdf_values = [get_pareto_pdf_val_from_x(x, xm, alpha) for x in x_values]
    return x_values, pdf_values

def get_pareto_pdf_val_from_x(x: float, xm: float, alpha: int) -> float:
    """
    Calculate the probability density function (PDF) value for a Pareto distribution at a given x.
    
    Args:
        x (float): The value at which to calculate the PDF
        xm (float): The scale parameter (minimum possible value) of the Pareto distribution
        alpha (int): The shape parameter of the Pareto distribution
        
    Returns:
        float: The PDF value at the given x
        
    Note:
        The Pareto PDF is defined as (α * xm^α) / x^(α+1) for x ≥ xm
    """
    return (alpha * pow(xm, alpha)) / pow(x, alpha + 1)


def exponential_pdf(x: np.ndarray) -> np.ndarray[float]:
    """
    Calculate the probability density function (PDF) values for an exponential distribution.
    
    Args:
        x (np.ndarray): Array of values at which to calculate the PDF
        
    Returns:
        np.ndarray[float]: Array of PDF values corresponding to the input x values
        
    Note:
        This function implements an exponential distribution with λ=0.75,
        where the PDF is defined as λ*e^(-λx) for x ≥ 0
    """
    return 0.75 * np.exp(-0.75 * x)

def gamma_pdf(x: np.ndarray) -> np.ndarray[float]:
    """
    Calculate the probability density function (PDF) values for a gamma distribution.
    
    Args:
        x (np.ndarray): Array of values at which to calculate the PDF
        
    Returns:
        np.ndarray[float]: Array of PDF values corresponding to the input x values
        
    Note:
        This function implements a gamma distribution with shape parameter k=2 and rate parameter λ=1.5,
        where the PDF is defined as (λ^k * x^(k-1) * e^(-λx)) / Γ(k)
        For k=2, this simplifies to (λ^2 * x * e^(-λx)), which equals 2.25 * x * e^(-1.5x)
    """
    return 2.25 * x * np.exp(-1.5 * x)