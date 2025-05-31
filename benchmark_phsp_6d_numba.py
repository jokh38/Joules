import timeit
import numpy as np
from numba import jit

# Copied and adapted from mqi_phsp_6d_prototype.py
@jit(nopython=True) # or @jit(nopython=True, fastmath=True) for potential further speedup
def phsp_6d_operator_numba(mean, sigma, rho):
    """
    Numba-jitted version of the Python prototype for phsp_6d::operator().
    Assumes mean, sigma, rho are already NumPy arrays.
    """
    # Inputs are expected to be NumPy arrays already for Numba nopython mode
    # mean_np = np.asarray(mean) # Not ideal in nopython mode if not already array
    # sigma_np = np.asarray(sigma)
    # rho_np = np.asarray(rho)

    # Generate random numbers
    Ux = np.random.normal(0, 1)
    Uy = np.random.normal(0, 1)
    Uz = np.random.normal(0, 1)
    Vx = np.random.normal(0, 1)
    Vy = np.random.normal(0, 1)
    # Vz is not used

    phsp = np.zeros(6) # Numba supports np.zeros

    # Calculate phase-space variables
    phsp[0] = mean[0] + sigma[0] * Ux
    phsp[1] = mean[1] + sigma[1] * Uy
    phsp[2] = mean[2] + sigma[2] * Uz

    # Numba supports np.sqrt
    term3_factor = rho[0] * Ux + Vx * np.sqrt(1.0 - rho[0]**2)
    phsp[3] = mean[3] + sigma[3] * term3_factor

    term4_factor = rho[1] * Uy + Vy * np.sqrt(1.0 - rho[1]**2)
    phsp[4] = mean[4] + sigma[4] * term4_factor

    phsp3_sq = phsp[3]**2
    phsp4_sq = phsp[4]**2

    # Numba's np.clip expects an array as the first argument.
    # For scalars, implement clipping manually:
    val_to_clip = phsp3_sq + phsp4_sq
    if val_to_clip < 0.0:
        sum_sq_clipped = 0.0
    elif val_to_clip > 1.0:
        sum_sq_clipped = 1.0
    else:
        sum_sq_clipped = val_to_clip
    # Or, more concisely: sum_sq_clipped = max(0.0, min(val_to_clip, 1.0))
    # The version above is slightly more explicit for Numba's type inference.

    phsp[5] = -1.0 * np.sqrt(1.0 - sum_sq_clipped)

    return phsp

def benchmark_numba():
    """
    Benchmarks the Numba-jitted phsp_6d_operator function.
    """
    # Define fixed input parameters as NumPy arrays
    mean_values_np = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.0], dtype=np.float64)
    sigma_values_np = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001], dtype=np.float64)
    rho_values_np = np.array([0.5, -0.5], dtype=np.float64)

    # Warm-up call for Numba JIT compilation
    print("Warming up Numba JIT compilation...")
    _ = phsp_6d_operator_numba(mean_values_np, sigma_values_np, rho_values_np)
    print("Warm-up complete.")

    # Number of repetitions for the benchmark
    # Numba can be significantly faster, so more repetitions might be needed
    # compared to pure Python to get a stable measurement.
    num_repetitions = 1_000_000
    # num_repetitions = 5_000_000 # Alternative if it runs too fast

    stmt_to_time = "phsp_6d_operator_numba(mean_values_np, sigma_values_np, rho_values_np)"

    # Setup code for timeit
    # Pass NumPy arrays directly to the Numba function
    setup_globals = {
        "phsp_6d_operator_numba": phsp_6d_operator_numba,
        "mean_values_np": mean_values_np,
        "sigma_values_np": sigma_values_np,
        "rho_values_np": rho_values_np
    }

    print(f"\nStarting benchmark for Numba-jitted phsp_6d_operator...")
    print(f"Number of repetitions: {num_repetitions:,}")

    # Measure execution time
    total_time = timeit.timeit(stmt_to_time, globals=setup_globals, number=num_repetitions)

    # Calculate average time per call
    average_time_per_call = total_time / num_repetitions

    print(f"\nNumba Benchmark Results:")
    print(f"Total time for {num_repetitions:,} calls: {total_time:.4f} seconds")
    print(f"Average time per call: {average_time_per_call:.8f} seconds (or {average_time_per_call*1e6:.2f} microseconds)")

if __name__ == '__main__':
    benchmark_numba()
