from assignment1 import generate_triangular, generate_pareto_distribution, generate_gamma_distribution_samples
from helpers.probability_density_functions import generate_theoretical_pdf_points, generate_theoretical_pareto_pdf_points, gamma_pdf
import matplotlib.pyplot as plt
import numpy as np


def plot_triangular_distribution_and_print_numeric_summary(n_samples: int, a: float, b: float, c: float) -> None:
    """
    Plots sampled triangular distribution against its theoretical PDF and prints summary statistics.
    
    Args:
        n_samples: Number of samples to generate
        a: Lower limit of the distribution
        b: Upper limit of the distribution
        c: Mode of the distribution
    """
    sampled_data = generate_triangular(n_samples, a, b, c)
    plt.hist(sampled_data, bins=50, density=True, alpha=0.6, color='g', label="Sampled Data")

    theoretical_x, theoretical_pdf = generate_theoretical_pdf_points(a, b, c, n_samples)

    print(f"Sampled Data: Mean = {np.mean(sampled_data):.2f}, Variance = {np.var(sampled_data):.2f}")
    print(f"Theoretical Data: Mean = {np.mean(theoretical_x):.2f}, Variance = {np.var(theoretical_x):.2f}")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical PDF", color='r')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Inverse Transform Sampling for a Triangular Distribution")
    plt.show()


def plot_pareto_distribution(n_samples: int = 100000, xm: float = 3, alpha: int = 2) -> None:
    """
    Plots sampled Pareto distribution against its theoretical PDF and prints summary statistics.
    
    Args:
        n_samples: Number of samples to generate
        xm: Scale parameter (minimum possible value)
        alpha: Shape parameter (tail index)
    """
    sampled_data = generate_pareto_distribution(n_samples, xm, alpha)
    
    plt.hist(sampled_data, bins=300, density=True, alpha=0.6, color='g', label="Sampled Data")

    theoretical_x, theoretical_pdf = generate_theoretical_pareto_pdf_points(xm, alpha, n_samples)

    print(f"Sampled Data: Mean = {np.mean(sampled_data):.2f}, Variance = {np.var(sampled_data):.2f}")
    print(f"Theoretical Data: Mean = {np.mean(theoretical_x):.2f}, Variance = {np.var(theoretical_x):.2f}")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical PDF", color='r')
    plt.xlabel("x")
    plt.xlim(0, 50)
    plt.ylabel("Density")
    plt.legend()
    plt.title("Inverse Transform Sampling for a Pareto Distribution")
    plt.show()


def plot_gamma_distribution_and_print_summary_statistics(n_samples: int = 10000) -> None:
    """
    Plots sampled Gamma distribution against its theoretical PDF and prints summary statistics.
    Uses acceptance-rejection sampling method.
    
    Args:
        n_samples: Number of proposals to generate (actual accepted samples will be fewer)
    """
    accepted_samples = generate_gamma_distribution_samples(n_samples)

    print(f"Acceptance rate: {len(accepted_samples) / n_samples:.2f}")

    plt.hist(accepted_samples, bins=50, density=True, alpha=0.6, color='g', label="Sampled Data")

    # Estimate theoretical gamma PDF
    theoretical_x = np.linspace(0, max(accepted_samples), 1000)
    theoretical_pdf = gamma_pdf(theoretical_x)

    # Calculate theoretical mean and variance
    theoretical_mean = 2 / 1.5  # α / λ
    theoretical_variance = 2 / (1.5 ** 2)  # α / λ^2

    observed_mean = np.mean(accepted_samples)
    observed_variance = np.var(accepted_samples)

    print(f"Theoretical Mean: {theoretical_mean:.2f}, Observed Mean: {observed_mean:.2f}")
    print(f"Theoretical Variance: {theoretical_variance:.2f}, Observed Variance: {observed_variance:.2f}")

    plt.plot(theoretical_x, theoretical_pdf, label="Theoretical Gamma PDF", color='r')

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Generated Gamma Distribution vs Theoretical Gamma PDF")
    plt.show()


if __name__ == "__main__":  
    # Generate and plot distributions with the specified parameters
    plot_triangular_distribution_and_print_numeric_summary(10000, 1, 7, 2)
    plot_pareto_distribution(10000, 3, 2)
    plot_gamma_distribution_and_print_summary_statistics(10000)
