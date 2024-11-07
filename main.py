import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 100  # Number of bins
n = m ** 2
T = 100  # Number of experimental runs to average results

def initialize_bins(m):
    """
    Initialize m bins, each with a load of 0 balls.
    """
    return [0] * m

def calculate_gap(bin_loads, n, m):
    average_load = n / m
    max_load = max(bin_loads)
    return max_load - average_load

def one_choice_strategy(m, n):
    """
    Allocate n balls into m bins using the one-choice strategy.
    Each ball is placed in a randomly chosen bin.
    """
    bin_loads = initialize_bins(m)

    for _ in range(n):
        chosen_bin = np.random.randint(0, m)  # Choose one bin randomly
        bin_loads[chosen_bin] += 1  # Increment the load of the chosen bin

    return bin_loads

def run_experiment_one_choice(m, n, T):
    
    gaps = []

    for _ in range(T):
        bin_loads = one_choice_strategy(m, n)
        gap = calculate_gap(bin_loads, n, m)
        gaps.append(gap)

    avg_gap = np.mean(gaps)
    return avg_gap

def experiment_over_n(m, max_n, T):
    ball_counts = []
    avg_gaps = []
    step = 5
    for num_balls in range(1, max_n + 1, step):  # Increase in steps of m balls
        avg_gap = run_experiment_one_choice(m, num_balls, T)
        ball_counts.append(num_balls)
        avg_gaps.append(avg_gap)
    return ball_counts, avg_gaps

def plot_results(ball_counts, avg_gaps, m, max_n):
    
    plt.figure(figsize=(10, 6))
    plt.plot(ball_counts, avg_gaps, label="One-Choice Strategy")
    plt.xlabel("Number of Balls (n)")
    plt.ylabel("Average Gap (G_n)")
    plt.title("Gap Evolution in One-Choice Strategy")
    plt.legend()
    plt.show()




def experiment_over_n_multiple_runs(m, max_n, T, num_runs):
    results = []
    
    for run in range(num_runs):
        print(f"\nStarting run {run + 1}")
        ball_counts, avg_gaps = experiment_over_n(m, max_n, T)
        results.append((ball_counts, avg_gaps))
    return results

def plot_multiple_results(results, m, max_n):

    plt.figure(figsize=(10, 6))
    
    for i, (ball_counts, avg_gaps) in enumerate(results):
        plt.plot(ball_counts, avg_gaps, label=f"Run {i + 1}")

    plt.xlabel("Number of Balls (n)")
    plt.ylabel("Average Gap (G_n)")
    plt.title("Gap Evolution in One-Choice Strategy - Multiple Runs - Heavy-Loaded")
    plt.legend()
    plt.show()

def experiment_once_choice_constant_n(m, n, T):
    avg_gaps = []  # List to store the average gap for each trial

    for trial in range(1, T + 1):
        avg_gap = run_experiment_one_choice(m, n, 1)  # Run one trial
        avg_gaps.append(avg_gap)  # Store the result

    # Create a plot with trial numbers on the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_gaps, marker='o', label=f"Average Gap for {n} Balls")
    
    plt.xlabel("Trial Number")
    plt.ylabel("Average Gap (G_n)")
    plt.title(f"Gap Evolution in One-Choice Strategy - Constant Balls (n = {n})")
    plt.legend()
    plt.show()


def run_average_trials(m, n, trials=100):
 
    gaps = []  # List to store gaps from each trial
    
    for trial in range(trials):
        gap = run_experiment_one_choice(m, n, 1)  # Run one trial
        gaps.append(gap)  # Store the result
    
    avg_gap = np.mean(gaps)  # Calculate the average gap
    print(f"Average gap over {trials} trials with {n} balls: {avg_gap}")


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
            chosen_bin = np.random.randint(0, m)  # Choose one bin randomly
            bin_loads[chosen_bin] += 1  # Increment the load of the chosen bin
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
num_runs = 10  # Number of different experiment runs to plot
#experiment_once_choice_constant_n(m,n,T)

run_average_trials(m, n)
"""