from generating_functions import generate_triangular, generate_pareto_distribution, generate_gamma_distribution_samples
from plotting import plot_triangular_distribution_and_print_numeric_summary, plot_pareto_distribution_and_print_numeric_summary, plot_gamma_distribution_and_print_summary_statistics

if __name__ == "__main__":
    n_samples = 10000

    a, b, c = 1, 7, 2
    triangular_samples = generate_triangular(n_samples, a, b, c)

    alpha, xm = 2, 3
    pareto_samples = generate_pareto_distribution(n_samples, xm, alpha)
    gamma_samples = generate_gamma_distribution_samples(n_samples)

    print('\n=== First Five Samples ===')
    print(f"First five triangular samples: : {triangular_samples[:5]}")
    print(f"First five pareto samples: : {pareto_samples[:5]}")
    print(f"First five gamma samples: : {gamma_samples[:5]}")
    print('==========================\n')
    plot_triangular_distribution_and_print_numeric_summary(triangular_samples, n_samples, a, b, c)
    plot_pareto_distribution_and_print_numeric_summary(pareto_samples, n_samples, xm, alpha)
    plot_gamma_distribution_and_print_summary_statistics(gamma_samples, n_samples)

