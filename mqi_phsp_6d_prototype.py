import numpy as np

def phsp_6d_operator(mean, sigma, rho):
    """
    Python prototype for the phsp_6d::operator() method.

    Args:
        mean (list or np.ndarray): Array of 6 floats for the mean values.
        sigma (list or np.ndarray): Array of 6 floats for the sigma values.
        rho (list or np.ndarray): Array of 2 floats for the rho values.

    Returns:
        np.ndarray: Array of 6 calculated phase-space variables.
    """
    # Ensure inputs are numpy arrays
    mean = np.asarray(mean)
    sigma = np.asarray(sigma)
    rho = np.asarray(rho)

    # Generate random numbers
    Ux = np.random.normal(0, 1)
    Uy = np.random.normal(0, 1)
    Uz = np.random.normal(0, 1)
    Vx = np.random.normal(0, 1)
    Vy = np.random.normal(0, 1)
    # Vz is not used in the C++ code for phsp_6d::operator()

    phsp = np.zeros(6)

    # Calculate phase-space variables
    phsp[0] = mean[0] + sigma[0] * Ux
    phsp[1] = mean[1] + sigma[1] * Uy
    phsp[2] = mean[2] + sigma[2] * Uz

    term3_factor = rho[0] * Ux + Vx * np.sqrt(1.0 - rho[0]**2)
    phsp[3] = mean[3] + sigma[3] * term3_factor

    term4_factor = rho[1] * Uy + Vy * np.sqrt(1.0 - rho[1]**2)
    phsp[4] = mean[4] + sigma[4] * term4_factor

    # Ensure the value inside sqrt for phsp[5] is non-negative
    phsp3_sq = phsp[3]**2
    phsp4_sq = phsp[4]**2

    # Clip sum_sq to be at most 1.0 to prevent issues with sqrt
    sum_sq_clipped = np.clip(phsp3_sq + phsp4_sq, 0, 1.0)

    phsp[5] = -1.0 * np.sqrt(1.0 - sum_sq_clipped)

    return phsp

if __name__ == '__main__':
    # Example Usage
    mean_values = [0.1, 0.2, 0.3, 0.01, 0.02, 0] # phsp[5] mean is typically 0 or not used directly
    sigma_values = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001] # sigma[5] is not directly used for phsp[5] calculation
    rho_values = [0.5, -0.5]

    print("Python prototype for phsp_6d::operator()")
    print(f"Mean: {mean_values}")
    print(f"Sigma: {sigma_values}")
    print(f"Rho: {rho_values}")

    # Generate a few sets of phase-space variables
    for i in range(3):
        phsp_result = phsp_6d_operator(mean_values, sigma_values, rho_values)
        print(f"\nGenerated phsp set {i+1}:")
        print(f"x:      {phsp_result[0]:.6f}")
        print(f"y:      {phsp_result[1]:.6f}")
        print(f"z:      {phsp_result[2]:.6f}")
        print(f"theta_x: {phsp_result[3]:.6f}")
        print(f"theta_y: {phsp_result[4]:.6f}")
        print(f"theta_z: {phsp_result[5]:.6f}")

    # Example with rho values that might cause issues if not handled
    rho_edge_case = [0.999, -0.999]
    print(f"\nExample with rho near +/-1: {rho_edge_case}")
    phsp_result_edge = phsp_6d_operator(mean_values, sigma_values, rho_edge_case)
    print(f"Generated phsp set (rho near +/-1):")
    print(f"x:      {phsp_result_edge[0]:.6f}")
    print(f"y:      {phsp_result_edge[1]:.6f}")
    print(f"z:      {phsp_result_edge[2]:.6f}")
    print(f"theta_x: {phsp_result_edge[3]:.6f}")
    print(f"theta_y: {phsp_result_edge[4]:.6f}")
    print(f"theta_z: {phsp_result_edge[5]:.6f}")

    # Example where phsp[3]^2 + phsp[4]^2 could exceed 1 without clipping
    # This requires specific random numbers, so we'll force it by setting mean and sigma for theta_x, theta_y
    # and specific random numbers (by temporarily overriding np.random.normal)

    # Store original random normal
    original_normal = np.random.normal

    # Mock np.random.normal to control its output for this specific test case
    # Ux, Uy, Uz, Vx, Vy
    mock_random_outputs = iter([1.5, 1.5, 0, 1.5, 1.5])
    def mock_normal(mean, std_dev):
        # Ensure it's our specific call, not from somewhere else in numpy
        if mean == 0 and std_dev == 1:
            return next(mock_random_outputs)
        return original_normal(mean, std_dev) # pragma: no cover

    np.random.normal = mock_normal

    mean_force_clipping = [0, 0, 0, 0.5, 0.5, 0]
    sigma_force_clipping = [1, 1, 1, 0.2, 0.2, 1]
    rho_force_clipping = [0.1, 0.1]

    print(f"\nExample to test clipping for phsp[5]:")
    print(f"Mean: {mean_force_clipping}")
    print(f"Sigma: {sigma_force_clipping}")
    print(f"Rho: {rho_force_clipping}")

    # This call will use the mocked np.random.normal
    phsp_result_clipping = phsp_6d_operator(mean_force_clipping, sigma_force_clipping, rho_force_clipping)

    # Restore original np.random.normal
    np.random.normal = original_normal

    print(f"Generated phsp set (testing clipping):")
    print(f"x:      {phsp_result_clipping[0]:.6f}")
    print(f"y:      {phsp_result_clipping[1]:.6f}")
    print(f"z:      {phsp_result_clipping[2]:.6f}")
    print(f"theta_x: {phsp_result_clipping[3]:.6f}") # Expected to be > 0.707 if clipping works
    print(f"theta_y: {phsp_result_clipping[4]:.6f}") # Expected to be > 0.707 if clipping works
    print(f"phsp[3]^2 + phsp[4]^2 before clipping: {(phsp_result_clipping[3]**2 + phsp_result_clipping[4]**2):.6f}")
    # This sum would be > 1 if not for the clipping of (phsp[3]^2 + phsp[4]^2) to 1.0 before sqrt
    print(f"theta_z: {phsp_result_clipping[5]:.6f}") # Expected to be 0 if clipping worked correctly
    if np.isclose(phsp_result_clipping[5], 0.0):
        print("Clipping for phsp[5] appears to be working correctly.")
    else:
        print("Clipping for phsp[5] might NOT be working correctly.")

    # Test with rho values exactly 1.0 and -1.0
    rho_exact_one = [1.0, -1.0]
    print(f"\nExample with rho = +/-1.0: {rho_exact_one}")
    # Reset mock random outputs for this test
    mock_random_outputs_rho_one = iter([0.5, 0.6, 0.7, 0.8, 0.9]) # Vx, Vy won't be used
    np.random.normal = lambda mean, std_dev: next(mock_random_outputs_rho_one) if mean==0 and std_dev==1 else original_normal(mean, std_dev)

    phsp_result_rho_one = phsp_6d_operator(mean_values, sigma_values, rho_exact_one)
    np.random.normal = original_normal # Restore

    print(f"Generated phsp set (rho = +/-1.0):")
    print(f"x:      {phsp_result_rho_one[0]:.6f}")
    print(f"y:      {phsp_result_rho_one[1]:.6f}")
    print(f"z:      {phsp_result_rho_one[2]:.6f}")
    print(f"theta_x: {phsp_result_rho_one[3]:.6f}") # sqrt(1-rho[0]^2) will be 0
    print(f"theta_y: {phsp_result_rho_one[4]:.6f}") # sqrt(1-rho[1]^2) will be 0
    print(f"theta_z: {phsp_result_rho_one[5]:.6f}")
    # Check if termX_factor calculation was correct
    # Ux_val = 0.5, Uy_val = 0.6
    # expected_term3_factor = rho_exact_one[0] * 0.5 + Vx * np.sqrt(1.0 - rho_exact_one[0]**2) = 1.0 * 0.5 + Vx * 0 = 0.5
    # expected_term4_factor = rho_exact_one[1] * 0.6 + Vy * np.sqrt(1.0 - rho_exact_one[1]**2) = -1.0 * 0.6 + Vy * 0 = -0.6
    expected_phsp3 = mean_values[3] + sigma_values[3] * (rho_exact_one[0] * 0.5)
    expected_phsp4 = mean_values[4] + sigma_values[4] * (rho_exact_one[1] * 0.6)
    if not np.isclose(phsp_result_rho_one[3], expected_phsp3):
        print(f"Mismatch in phsp[3] for rho=1: Expected {expected_phsp3}, Got {phsp_result_rho_one[3]}") # pragma: no cover
    if not np.isclose(phsp_result_rho_one[4], expected_phsp4):
        print(f"Mismatch in phsp[4] for rho=-1: Expected {expected_phsp4}, Got {phsp_result_rho_one[4]}") # pragma: no cover

    print("\nEnd of examples.")
