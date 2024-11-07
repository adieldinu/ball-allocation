import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1000  # Number of bins
n = m**2  # Number of balls (you can change this as needed)
T = 100  # Number of trials per run
betas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Probability of using one-choice strategy

def initialize_bins(m):
    """ Initialize m bins, each with a load of 0 balls. """
    return [0] * m

def calculate_gap(bin_loads, n, m):
    """ Calculate the gap Gn for a given bin load distribution. """
    average_load = n / m
    max_load = max(bin_loads)
    return max_load - average_load

def one_choice_strategy(bin_loads, m):
    """ Choose a single bin at random and place a ball there. """
    chosen_bin = np.random.randint(0, m)
    bin_loads[chosen_bin] += 1

def two_choice_strategy(bin_loads, m):
    """ Choose two bins at random and place a ball in the one with the least load. """
    chosen_bins = np.random.choice(m, size=2, replace=False)
    bin1, bin2 = chosen_bins[0], chosen_bins[1]
    if bin_loads[bin1] < bin_loads[bin2]:
        bin_loads[bin1] += 1
    elif bin_loads[bin1] > bin_loads[bin2]:
        bin_loads[bin2] += 1
    else:
        chosen_bin = np.random.choice([bin1, bin2])
        bin_loads[chosen_bin] += 1

def one_plus_beta_choice_strategy(m, n, β):
    """ Allocate n balls into m bins using the (1 + β)-choice strategy. """
    bin_loads = initialize_bins(m)
    for _ in range(n):
        if np.random.rand() < β:
            one_choice_strategy(bin_loads, m)
        else:
            two_choice_strategy(bin_loads, m)
    return bin_loads

def run_experiment_one_plus_beta_choice(m, n, T, β):
    """ Run T trials of the (1 + β)-choice allocation strategy and return list of average gaps across trials. """
    trial_gaps = []
    for _ in range(T):
        bin_loads = one_plus_beta_choice_strategy(m, n, β)
        gap = calculate_gap(bin_loads, n, m)
        trial_gaps.append(gap)
    return trial_gaps  # List of gap values for each trial


def plot_experiments_one_plus_beta_multiple_betas(m, n, T, betas, R):
    """
    Plot R runs of T trials each for each β value in (1 + β)-choice strategy,
    showing the average gap across different runs.
    """
    plt.figure(figsize=(14, 8))
    
    for β in betas:
        print(f"\nRunning experiments for β = {β}")
        for run in range(R):
            print(f"  Run {run + 1}")
            gaps = run_experiment_one_plus_beta_choice(m, n, T, β)
            plt.plot(range(1, T + 1), gaps, label=f"β = {β}, Run {run + 1}")
    
    plt.xlabel("Trial Number")
    plt.ylabel("Gap (G_n)")
    plt.title(f"(1 + β)-Choice Strategy: Evolution of Average Gap Across Runs for Various β Values")
    
    # Update the legend to be inside the plot and organized into multiple columns
    plt.legend(loc='upper right', ncol=2, fontsize='small', frameon=False)
    plt.tight_layout()
    plt.show()


def batched_allocation_strategy(m, n, b, beta=0.1):
    bin_loads = [0] * m  # Initialize bin loads
    gaps = []
    # Process balls in batches
    for _ in range(n // b):  # Number of batches
        initial_bin_loads = bin_loads.copy()  # Snapshot of bin loads at the start of each batch
        
        gap = calculate_gap(bin_loads, n, m)
        gaps.append(gap)


        # Allocate each ball in the batch without updating loads within the batch
        for _ in range(b):
            if np.random.rand() < beta:
                one_choice_strategy(bin_loads, m)
            else:
                two_choice_strategy(bin_loads, m)

            
        
    initial_bin_loads = bin_loads.copy()  # Snapshot of bin loads at the start of each batch
        
    gap = calculate_gap(bin_loads, n, m)
    gaps.append(gap)

    return bin_loads, gaps

def plot_gaps_for_multiple_batches(m, n, batch_sizes):
    """
    Runs the batched allocation strategy for multiple batch sizes and plots the gap evolution for each.
    """
    plt.figure(figsize=(10, 6))

    for b in batch_sizes:
        # Run the allocation strategy for the current batch size
        _, gaps = batched_allocation_strategy(m, n, b)
        batch_steps = np.arange(1, len(gaps) + 1)
        
        # Plot the gaps for the current batch size
        plt.plot(batch_steps, gaps, marker='o', linestyle='-', label=f"Batch size = {b}")
    
    plt.xlabel("Batch Number")
    plt.ylabel("Gap (G_n)")
    plt.title("Evolution of Gap $G_n$ Across Batches for Different Batch Sizes")
    plt.legend()
    plt.show()






batch_sizes = [m, 2 * m, 5 * m, 10 * m, 20 * m]  # Different batch sizes to test

# Plot gaps for multiple batch sizes
plot_gaps_for_multiple_batches(m, n, batch_sizes)


"""

R=1
# Example usage
plot_experiments_one_plus_beta_multiple_betas(m, n, T, betas, R)


# Example usage
#R = 5  # Number of independent runs to print results for
#experiment_with_one_plus_beta(m, n, T, β, R)
"""