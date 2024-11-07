import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 100  # Number of bins
n = m**2  # Number of balls (fixed)
T = 100  # Number of experimental trials
R=10
def initialize_bins(m):
    """ Initialize m bins, each with a load of 0 balls. """
    return [0] * m

def calculate_gap(bin_loads, n, m):
    average_load = n / m
    max_load = max(bin_loads)
    return max_load - average_load

def two_choice_strategy(m, n):
    bin_loads = initialize_bins(m)

    for _ in range(n):
        # Randomly pick two bins
        chosen_bins = np.random.choice(m, size=2, replace=False)
        bin1, bin2 = chosen_bins[0], chosen_bins[1]

        # Compare loads and decide where to put the ball
        if bin_loads[bin1] < bin_loads[bin2]:
            bin_loads[bin1] += 1
        elif bin_loads[bin1] > bin_loads[bin2]:
            bin_loads[bin2] += 1
        else:
            # If loads are equal, pick one at random
            chosen_bin = np.random.choice([bin1, bin2])
            bin_loads[chosen_bin] += 1

    return bin_loads

def run_experiment_two_choice(m, n, T):
    """
    Run T trials of the two-choice allocation strategy and compute the average gap for each trial.
    
    Parameters:
    - m: number of bins
    - n: number of balls
    - T: number of trials
    
    Returns:
    - avg_gaps: List of average gaps for each trial
    """
    avg_gaps = []

    for trial in range(T):
        bin_loads = two_choice_strategy(m, n)
        gap = calculate_gap(bin_loads, n, m)
        avg_gaps.append(gap)

    return avg_gaps

def plot_results_two_choice(avg_gaps, T):
    """
    Plot the average gap from T trials for the two-choice strategy.
    
    Parameters:
    - avg_gaps: List of average gaps for each trial
    - T: number of trials
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T+1), avg_gaps, label="Two-Choice Strategy", marker='o')
    plt.xlabel("Trial Number")
    plt.ylabel("Average Gap (G_n)")
    plt.title(f"Gap Evolution in Two-Choice Strategy - {T} Trials")
    plt.legend()
    plt.show()


def multiple_runs_two_choice(m, n, T, R):
    """
    Perform R runs of T trials each, plotting the average gap across trials for each run.
    """
    plt.figure(figsize=(12, 8))
    
    for run in range(R):
        avg_gaps = run_experiment_two_choice(m, n, T)
        plt.plot(range(1, T+1), avg_gaps, label=f"Run {run + 1}", marker='o')
    
    plt.xlabel("Trial Number")
    plt.ylabel("Average Gap (G_n)")
    plt.title(f"Gap Evolution in Two-Choice Strategy - {R} Runs, {T} Trials Each")
    plt.legend()
    plt.show()

def print_results_two_choice(m, n, T, R):
    """
    Perform R runs of T trials each and print the average gap for each run.
    """
    for run in range(1, R + 1):
        avg_gaps = run_experiment_two_choice(m, n, T)
        run_avg_gap = np.mean(avg_gaps)
        print(f"Run {run}: Average Gap over {T} trials = {run_avg_gap:.4f}")


def batched_allocation_strategy(m, n, b):
    bin_loads = [0] * m  # Initialize bin loads
    gaps = []
    # Process balls in batches
    for _ in range(n // b):  # Number of batches
        initial_bin_loads = bin_loads.copy()  # Snapshot of bin loads at the start of each batch
        
        gap = calculate_gap(bin_loads, n, m)
        gaps.append(gap)


        # Allocate each ball in the batch without updating loads within the batch
        for _ in range(b):
            chosen_bins = np.random.choice(m, size=2, replace=False)
            bin1, bin2 = chosen_bins[0], chosen_bins[1]

            if bin_loads[bin1] < bin_loads[bin2]:
                bin_loads[bin1] += 1
            elif bin_loads[bin1] > bin_loads[bin2]:
                bin_loads[bin2] += 1
            else:
                # If loads are equal, pick one at random
                chosen_bin = np.random.choice([bin1, bin2])
                bin_loads[chosen_bin] += 1
        
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

# Example usage
print_results_two_choice(m, n, T, R)



# Run the experiment with fixed number of balls and bins
avg_gaps = run_experiment_two_choice(m, n, T)

# Plot the results for all trials
plot_results_two_choice(avg_gaps, T)
"""

#multiple_runs_two_choice(m, n, T, R)