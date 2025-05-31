import timeit
import numpy as np
# Assuming mqi_phsp_6d_prototype.py is in the same directory or accessible in PYTHONPATH
from mqi_phsp_6d_prototype import phsp_6d_operator

def benchmark():
    """
    Benchmarks the phsp_6d_operator function.
    """
    # Define fixed input parameters
    mean_values = [0.1, 0.2, 0.3, 0.01, 0.02, 0]
    sigma_values = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    rho_values = [0.5, -0.5]

    # Number of repetitions for the benchmark
    num_repetitions = 100_000

    # Create a string for the statement to be timed
    # This setup ensures that the phsp_6d_operator function and its parameters
    # are accessible to timeit in its own namespace.
    stmt_to_time = "phsp_6d_operator(mean_values, sigma_values, rho_values)"

    # Setup code to import the function and define parameters for timeit
    # globals() is used to pass the current global namespace to timeit
    # Alternatively, we can pass a dictionary:
    setup_code = """
from __main__ import phsp_6d_operator
mean_values = [0.1, 0.2, 0.3, 0.01, 0.02, 0]
sigma_values = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
rho_values = [0.5, -0.5]
"""
    # It's generally better to pass necessary variables explicitly to timeit's context
    # to avoid relying on __main__ or globals() if the script structure changes.
    # However, for this simple script, using globals() is convenient.

    current_globals = globals().copy()
    current_globals.update({
        "mean_values": mean_values,
        "sigma_values": sigma_values,
        "rho_values": rho_values,
        "phsp_6d_operator": phsp_6d_operator # Ensure function is in the context
    })


    print(f"Starting benchmark for phsp_6d_operator...")
    print(f"Number of repetitions: {num_repetitions:,}")

    # Measure execution time
    total_time = timeit.timeit(stmt_to_time, globals=current_globals, number=num_repetitions)

    # Calculate average time per call
    average_time_per_call = total_time / num_repetitions

    print(f"\nBenchmark Results:")
    print(f"Total time for {num_repetitions:,} calls: {total_time:.4f} seconds")
    print(f"Average time per call: {average_time_per_call:.8f} seconds (or {average_time_per_call*1e6:.2f} microseconds)")

if __name__ == '__main__':
    benchmark()
