import numpy as np
from math import sqrt, pow
np.random.seed(42)
from helpers.errors import InvalidParametersError
from helpers.probability_density_functions import gamma_pdf, exponential_pdf

def get_value_from_probability(
        samples: np.ndarray, 
        cdf_at_c: float, 
        a: float, 
        b: float, 
        c: float
    ) -> np.ndarray[float]:
    """
    Transforms uniform samples to triangular distribution values using the inverse CDF method.
    
    Args:
        samples (np.ndarray): Uniform random samples between 0 and 1.
        cdf_at_c (float): Value of the CDF at point c.
        a (float): Lower bound of the triangular distribution.
        b (float): Upper bound of the triangular distribution.
        c (float): Mode of the triangular distribution.
        
    Returns:
        np.ndarray[float]: Samples transformed to follow a triangular distribution.
    """
    transformed_samples = [get_x_from_uniform(U, a, b, c, cdf_at_c) for U in samples]
    return transformed_samples

def get_x_from_uniform(
        U: float, 
        a: float, 
        b: float, 
        c: float, 
        cdf_at_c: float
    ) -> float:
    """
    Transforms a single uniform random value to a triangular distribution value.
    
    Args:
        U (float): Uniform random value between 0 and 1.
        a (float): Lower bound of the triangular distribution.
        b (float): Upper bound of the triangular distribution.
        c (float): Mode of the triangular distribution.
        cdf_at_c (float): Value of the CDF at point c.
        
    Returns:
        float: A value from the triangular distribution.
    """

    # Formulas derived on paper are used to calculate the transformed value
    if U < cdf_at_c:
        return sqrt(U * (b - a) * (c - a)) + a
    else:
        return b - sqrt((1 - U) * (b - a) * (b - c))

def generate_triangular(
        n_samples: int, 
        a: float, 
        b: float, 
        c: float
    ) -> np.ndarray[float]:
    """
    Generates random samples from a triangular distribution.
    
    Args:
        n_samples (int): Number of samples to generate.
        a (float): Lower bound of the triangular distribution.
        b (float): Upper bound of the triangular distribution.
        c (float): Mode of the triangular distribution (a < c < b).
        
    Returns:
        np.ndarray[float]: Random samples following a triangular distribution.
        
    Raises:
        InvalidParametersError: If parameters are not of correct type or don't satisfy a < c < b.
    """
    if  (not isinstance(a, (int, float)) 
            or not isinstance(b, (int, float)) 
            or not isinstance(c, (int, float)) 
            or not isinstance(n_samples, int)
        ):
        raise InvalidParametersError(message="a, b, and c must be of type int or float")
    
    if not a < c < b or not n_samples > 0:
        raise InvalidParametersError(message="a < c < b must be satisfied and n_samples must be greater than 0")
    
    samples = np.random.uniform(0, 1, n_samples)

    cdf_value_at_c = (c - a) / (b - a)

    transformed_samples = get_value_from_probability(samples, cdf_value_at_c, a, b, c)

    return transformed_samples

def generate_pareto_distribution(
        n_samples: int, 
        xm: float, 
        alpha: int
    ) -> np.ndarray[float]:
    """
    Generates random samples from a Pareto distribution.
    
    Args:
        n_samples (int): Number of samples to generate.
        xm (float): Scale parameter (minimum possible value).
        alpha (int): Shape parameter (tail index).
        
    Returns:
        np.ndarray[float]: Random samples following a Pareto distribution.
        
    Raises:
        InvalidParametersError: If parameters are not of correct type or n_samples <= 0.
    """
    if (not isinstance(xm, (int, float)) 
        or not isinstance(alpha, int) 
        or not isinstance(n_samples, int)
        ):
        raise InvalidParametersError(message="xm must be of type int or float and alpha must be of type int")
    
    if not n_samples > 0:
        raise InvalidParametersError(message="n_samples must be greater than 0")
    
    samples = np.random.uniform(0, 1, n_samples)
    transformed_samples = get_pareto_value_from_probability(samples, xm, alpha)
    return transformed_samples

def get_pareto_value_from_probability(
        samples: np.ndarray, 
        xm: float, 
        alpha: int
    ) -> np.ndarray[float]:
    """
    Transforms uniform samples to Pareto distribution values using the inverse CDF method.
    
    Args:
        samples (np.ndarray): Uniform random samples between 0 and 1.
        xm (float): Scale parameter of the Pareto distribution.
        alpha (int): Shape parameter of the Pareto distribution.
        
    Returns:
        np.ndarray[float]: Samples transformed to follow a Pareto distribution.
    """
    transformed_samples = [get_x_from_pareto(U, xm, alpha) for U in samples]
    return transformed_samples

def get_x_from_pareto(
        U: float, 
        xm: float, 
        alpha: int
    ) -> float:
    """
    Transforms a single uniform random value to a Pareto distribution value.
    
    Args:
        U (float): Uniform random value between 0 and 1.
        xm (float): Scale parameter of the Pareto distribution.
        alpha (int): Shape parameter of the Pareto distribution.
        
    Returns:
        float: A value from the Pareto distribution.
    """

    # Formulas derived on paper are used to calculate the transformed value
    return xm / pow(1 - U, 1/alpha)

def generate_gamma_distribution_samples(
        n_samples: int = 10000
    ) -> np.ndarray[float]:
    """
    Generates random samples from a Gamma distribution using acceptance-rejection method.
    
    This implementation uses an exponential distribution as the proposal distribution
    and accepts/rejects samples based on the ratio of gamma pdf to scaled exponential pdf.
    
    Args:
        n_samples (int, optional): Number of exponential samples to generate. Defaults to 10000.
            Note that the actual number of accepted samples will be less than this.
        
    Returns:
        np.ndarray[float]: Random samples following a Gamma distribution.
        
    Raises:
        InvalidParametersError: If n_samples is not an integer or n_samples <= 0.
    """
    if not isinstance(n_samples, int):
        raise InvalidParametersError(message="n_samples must be of type int")   

    if not n_samples > 0:
        raise InvalidParametersError(message="n_samples must be greater than 0")
    lambda_2 = 0.75
    M = 4 / np.e

    samples = np.random.uniform(0, 1, n_samples)
    exponential_samples = -np.log(1 - samples) / lambda_2   # Inverse CDF of exponential distribution as calculated on paper

    accepted_samples = []
    for sample in exponential_samples:
        if np.random.uniform(0, 1) <= (gamma_pdf(sample) / (M * exponential_pdf(sample))):
            accepted_samples.append(sample)
    
    return accepted_samples
