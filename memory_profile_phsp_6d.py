from memory_profiler import profile
import numpy as np
# Import the original function
from mqi_phsp_6d_prototype import phsp_6d_operator as original_phsp_6d_operator

# Apply the @profile decorator to the imported function directly.
# memory_profiler should be able to handle this.
# If not, we would define a wrapper function here and decorate that.
profiled_phsp_6d_operator = profile(original_phsp_6d_operator)

def main():
    """
    Main function to set up parameters and call the profiled function.
    """
    # Define fixed input parameters
    mean_values = [0.1, 0.2, 0.3, 0.01, 0.02, 0]
    sigma_values = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    rho_values = [0.5, -0.5]

    print("Starting memory profiling for phsp_6d_operator...")
    print(f"Input Mean: {mean_values}")
    print(f"Input Sigma: {sigma_values}")
    print(f"Input Rho: {rho_values}")

    # Call the profiled function
    # The result is not strictly needed for memory profiling output, but good practice to get it.
    result = profiled_phsp_6d_operator(mean_values, sigma_values, rho_values)

    print(f"\nFunction called. Memory profiler will output results.")
    print(f"Result of the function call (for verification): {result}")

if __name__ == '__main__':
    # This script needs to be run with:
    # python -m memory_profiler memory_profile_phsp_6d.py
    main()
