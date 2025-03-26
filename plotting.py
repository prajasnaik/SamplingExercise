from generating_functions import generate_triangular, generate_pareto_distribution, generate_gamma_distribution_samples
from helpers.probability_density_functions import generate_theoretical_pdf_points, generate_theoretical_pareto_pdf_points, gamma_pdf
import matplotlib.pyplot as plt
import numpy as np


def plot_triangular_distribution_and_print_numeric_summary(
        generated_samples: np.ndarray[float] | None = None, 
        n_samples: int = 10000, 
        a: float = 1, 
        b: float = 7, 
        c: float = 2
    ) -> None:

    """
    Plots sampled triangular distribution against its theoretical PDF and prints summary statistics.
    
    Args:
        generated_samples: Pre-generated samples to plot
        n_samples: Number of samples to generate
        a: Lower limit of the distribution
        b: Upper limit of the distribution
        c: Mode of the distribution
    """
    if generated_samples is None:
        sampled_data = generate_triangular(n_samples, a, b, c)
    else:
        sampled_data = generated_samples

    plt.hist(sampled_data, bins=50, density=True, alpha=0.6, color='g', label="Sampled Data")

    theoretical_x, theoretical_pdf = generate_theoretical_pdf_points(a, b, c, n_samples)
    
    # Print summary statistics
    print("\n=== Summary Statistics - Piecewise Triangular Distribution ===")
    print(f"Sampled Mean = {np.mean(sampled_data):.2f}, Theoretical Mean = {(a + b + c) / 3:.2f}")
    print(f"Sampled Variance = {np.var(sampled_data):.2f}, Theoretical Variance = {(a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18:.2f}")
    print("==============================================================\n")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical PDF", color='r')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Inverse Transform Sampling for a Triangular Distribution")
    plt.show()


def plot_pareto_distribution_and_print_numeric_summary(
        generated_samples: np.ndarray[float] | None = None,
        n_samples: int = 100000, 
        xm: float = 3, 
        alpha: int = 2
    ) -> None:
    """
    Plots sampled Pareto distribution against its theoretical PDF and prints summary statistics.
    
    Args:
        generated_samples: Pre-generated samples to plot
        n_samples: Number of samples to generate
        xm: Scale parameter (minimum possible value)
        alpha: Shape parameter (tail index)
    """
    if generated_samples is None:
        sampled_data = generate_pareto_distribution(n_samples, xm, alpha)
    else:
        sampled_data = generated_samples
    
    plt.hist(sampled_data, bins=300, density=True, alpha=0.6, color='g', label="Sampled Data")

    theoretical_x, theoretical_pdf = generate_theoretical_pareto_pdf_points(xm, alpha, n_samples)

    # Print summary statistics
    print("\n=== Summary Statistics - Pareto Distribution ===")
    theoretical_mean = alpha * xm / (alpha - 1) if alpha > 1 else np.mean(theoretical_x)
    theoretical_variance = xm ** 2 * alpha / ((alpha - 1) ** 2 * (alpha - 2)) if alpha > 2 else np.var(theoretical_x)
    if alpha <= 1:
        print(f"Sampled Mean = {np.mean(sampled_data):.2f}, Theoretical Mean (Sampled) = {theoretical_mean:.2f}, Theoretical Mean (Actual) = infinity")
    else:
        print(f"Sampled Mean = {np.mean(sampled_data):.2f}, Theoretical Mean = {theoretical_mean:.2f}")
    
    if alpha <= 2:
        print(f"Sampled Variance = {np.var(sampled_data):.2f}, Theoretical Variance (Sampled) = {theoretical_variance:.2f}, Theoretical Variance (Actual) = infinity")
    else:
        print(f"Sampled Variance = {np.var(sampled_data):.2f}, Theoretical Variance = {theoretical_variance:.2f}")
    print("================================================\n")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical PDF", color='r')
    plt.xlabel("x")
    plt.xlim(0, 50)
    plt.ylabel("Density")
    plt.legend()
    plt.title("Inverse Transform Sampling for a Pareto Distribution")
    plt.show()


def plot_gamma_distribution_and_print_summary_statistics(
        generated_samples: np.ndarray | None = None, 
        n_samples: int = 10000
    ) -> None:

    """
    Plots sampled Gamma distribution against its theoretical PDF and prints summary statistics.
    Uses acceptance-rejection sampling method.
    
    Args:
        generated_samples: Pre-generated samples to plot
        n_samples: Number of proposals to generate (actual accepted samples will be fewer)
    """

    if generated_samples is None:
        accepted_samples = generate_gamma_distribution_samples(n_samples)
    else:
        accepted_samples = generated_samples



    # Estimate theoretical gamma PDF
    theoretical_x = np.linspace(0, max(accepted_samples), 1000)
    theoretical_pdf = gamma_pdf(theoretical_x)

    # Calculate theoretical mean and variance
    theoretical_mean = 2 / 1.5  # α / λ
    theoretical_variance = 2 / (1.5 ** 2)  # α / λ^2

    observed_mean = np.mean(accepted_samples)
    observed_variance = np.var(accepted_samples)
    
    # Print summary statistics
    print("\n=== Summary Statistics - Gamma Distribution ===")
    print(f"Acceptance rate: {len(accepted_samples) / n_samples:.2f}")
    print(f"Sampled Mean: {observed_mean:.2f}, Theoretical Mean: {theoretical_mean:.2f}")
    print(f"Sampled Variance: {observed_variance:.2f}, Theoretical Variance: {theoretical_variance:.2f}")
    print("===============================================\n")

    plt.hist(accepted_samples, bins=50, density=True, alpha=0.6, color='g', label="Sampled Data")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical Gamma PDF", color='r')

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Generated Gamma Distribution vs Theoretical Gamma PDF")
    plt.show()


if __name__ == "__main__":  
    # Generate and plot distributions with the specified parameters
    plot_triangular_distribution_and_print_numeric_summary(None, 10000, 1, 7, 2)
    plot_pareto_distribution_and_print_numeric_summary(None, 10000, 3, 2)
    plot_gamma_distribution_and_print_summary_statistics(None, 10000)
