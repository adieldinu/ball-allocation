import numpy as np
import matplotlib.pyplot as plt

# (1 + β)-Choice allocation strategy with beta parameter
def one_plus_beta_choice_strategy(m, n, beta):
    bin_loads = [0] * m
    for _ in range(n):
        if np.random.rand() < beta:
            chosen_bin = np.random.randint(0, m)
        else:
            bins = np.random.choice(m, 2, replace=False)
            chosen_bin = bins[np.argmin([bin_loads[bins[0]], bin_loads[bins[1]]])]
        bin_loads[chosen_bin] += 1
    return bin_loads

# Batched allocation strategy wrapper
def batched_allocation_strategy(m, n, strategy_func, b, beta):
    bin_loads = [0] * m  # Initialize bin loads
    
    # Process balls in batches
    for _ in range(n // b):  # Number of batches
        initial_bin_loads = bin_loads.copy()  # Snapshot of bin loads at the start of each batch
        
        # Allocate each ball in the batch without updating loads within the batch
        for _ in range(b):
            bins = np.random.choice(m, 2, replace=False)
            if np.random.rand() < beta:
                selected_bin = bins[0]  # Random choice for one-choice
            else:
                selected_bin = bins[np.argmin([initial_bin_loads[bins[0]], initial_bin_loads[bins[1]]])]  # Two-choice based on initial snapshot
            bin_loads[selected_bin] += 1
    
    return bin_loads

# Running the experiment with averaging across trials
def run_experiment(m, n, T, allocation_strategy, strategy_func, b, beta):
    gaps = []
    for _ in range(T):
        bin_loads = allocation_strategy(m, n, strategy_func, b, beta)
        gap = max(bin_loads) - (n / m)  # Calculate the gap
        gaps.append(gap)
    avg_gap = np.mean(gaps)
    return avg_gap, gaps

# Plotting function to show results for (1 + β)-Choice with varying batch sizes
def plot_results_for_one_plus_beta(m, max_n, T, beta, batch_sizes):
    n_values = np.arange(m, max_n + 1, m)  # Light-load to heavy-load scenario
    
    for b in batch_sizes:
        max_n = b * m  # Adjust max_n based on batch size
        n_values = np.arange(b, max_n + 1, b)  # Scale n based on batch size

        avg_gaps = []
        for n in n_values:
            avg_gap, _ = run_experiment(m, n, T, batched_allocation_strategy, one_plus_beta_choice_strategy, b, beta)
            avg_gaps.append(avg_gap)

        plt.plot(n_values, avg_gaps, label=f"(1 + β)-Choice, β={beta}, b={b}")

    plt.xlabel("Number of Balls (n)")
    plt.ylabel("Average Gap (G_n)")
    plt.legend()
    plt.title(f"Average Gap $G_n$ for (1 + β)-Choice Strategy with β={beta} across Batch Sizes")
    plt.show()

# Parameters
m = 100  # Number of bins
T = 100  # Number of trials
max_n = m ** 2  # Maximum number of balls, up to heavy-load scenario
beta = 0.5  # (1 + β)-Choice parameter
batch_sizes = [m, 2 * m, 10 * m, 20 * m, 50 * m]  # Batch sizes to test

# Run and plot the results
plot_results_for_one_plus_beta(m, max_n, T, beta, batch_sizes)
