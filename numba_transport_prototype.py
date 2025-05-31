from numba import cuda
import numpy as np
import math # For math.floor in kernel
import time

# Define the Numba CUDA JIT kernel
@cuda.jit
def numba_simplified_transport_kernel(
    particles,          # 2D Device array: (n_particles, 7) [x,y,z,dx,dy,dz,E]
    dose_array,         # 1D Device array for dose deposition
    step_size,          # Fixed step size
    energy_loss_per_step, # Fixed energy loss per step
    n_steps_max,        # Maximum simulation steps
    voxel_width,        # Width of each dose voxel
    n_voxels            # Total number of dose voxels
):
    idx = cuda.grid(1) # Global thread ID

    if idx >= particles.shape[0]: # particles.shape[0] is n_particles
        return

    # Particle components indices
    X, Y, Z = 0, 1, 2
    DX, DY, DZ = 3, 4, 5
    ENERGY = 6

    # Extract current particle's data
    # Numba allows direct access like a 2D array
    x = particles[idx, X]
    y = particles[idx, Y]
    z = particles[idx, Z]
    dx = particles[idx, DX]
    dy = particles[idx, DY]
    dz = particles[idx, DZ]
    energy = particles[idx, ENERGY]

    for _ in range(n_steps_max): # Numba JIT prefers explicit range
        if energy <= 0.0:
            break

        # Update particle position
        x += dx * step_size
        y += dy * step_size
        z += dz * step_size

        # Simulate energy loss
        deposited_energy_this_step = energy_loss_per_step
        if energy < energy_loss_per_step:
            deposited_energy_this_step = energy

        energy -= deposited_energy_this_step

        # Simplified dose deposition (along x-axis)
        if x >= 0.0: # Assuming positive x space for voxels
            # math.floor is available in Numba CUDA kernels
            voxel_index = math.floor(x / voxel_width)

            # Boundary check for voxel_index (must be int for atomic.add)
            # Also ensure it's within [0, n_voxels - 1]
            if voxel_index >= 0 and voxel_index < n_voxels:
                # cuda.atomic.add(array, index, value)
                # Ensure voxel_index is an integer type for indexing
                cuda.atomic.add(dose_array, int(voxel_index), deposited_energy_this_step)

    # Update the particle array with final state
    particles[idx, X] = x
    particles[idx, Y] = y
    particles[idx, Z] = z
    particles[idx, ENERGY] = energy


def run_numba_prototype():
    # Define particle component indices for use in host and device code clarity
    X, Y, Z = 0, 1, 2
    DX, DY, DZ = 3, 4, 5
    ENERGY = 6

    # --- Parameters ---
    N_PARTICLES = 1024 * 10
    PARTICLE_COMPONENTS = 7  # x, y, z, dx, dy, dz, E
    N_VOXELS = 100
    VOXEL_WIDTH = 0.1

    INIT_ENERGY = 100.0
    ENERGY_LOSS_PER_STEP = 5.0
    STEP_SIZE = 0.05
    N_STEPS_MAX = 50

    # For benchmarking
    N_BENCHMARK_RUNS = 100

    # --- Initialize Host Data (NumPy) ---
    print("Initializing host data...")
    particles_np = np.zeros((N_PARTICLES, PARTICLE_COMPONENTS), dtype=np.float32)

    for i in range(N_PARTICLES):
        particles_np[i, X] = 0.0  # x
        particles_np[i, Y] = np.random.uniform(-0.1, 0.1)  # y
        particles_np[i, Z] = np.random.uniform(-0.1, 0.1)  # z

        dx_val, dy_val, dz_val = np.random.uniform(0.8, 1.0), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)
        dir_len = np.sqrt(dx_val**2 + dy_val**2 + dz_val**2)
        particles_np[i, DX] = dx_val / dir_len if dir_len > 0 else 0
        particles_np[i, DY] = dy_val / dir_len if dir_len > 0 else 0
        particles_np[i, DZ] = dz_val / dir_len if dir_len > 0 else 0
        particles_np[i, ENERGY] = INIT_ENERGY

    dose_np = np.zeros(N_VOXELS, dtype=np.float32)

    # --- Transfer Data to GPU (Numba CUDA device arrays) ---
    print("Transferring data to GPU...")
    particles_device = cuda.to_device(particles_np)
    dose_device = cuda.to_device(dose_np) # Initial dose array (zeros)

    # --- Kernel Launch Configuration ---
    threadsperblock = 256
    blockspergrid = (N_PARTICLES + (threadsperblock - 1)) // threadsperblock

    print(f"Launching kernel with {blockspergrid} blocks, {threadsperblock} threads per block.")

    # --- Warm-up Calls ---
    print("Performing warm-up calls...")
    for _ in range(5):
        # Re-copy initial dose array for each warm-up if it's modified
        # For this prototype, dose_device is reset before benchmark loop
        numba_simplified_transport_kernel[blockspergrid, threadsperblock](
            particles_device, dose_device, STEP_SIZE, ENERGY_LOSS_PER_STEP,
            N_STEPS_MAX, VOXEL_WIDTH, N_VOXELS
        )
    cuda.synchronize()
    # Reset particle and dose states if necessary after warm-up, or use fresh copies for benchmark
    # For simplicity, we'll reset dose_device before the benchmark loop.
    # Particle state is modified in-place, so for "fair" repeated benchmarks, it should also be reset.
    # However, for this prototype, we focus on kernel execution time itself.
    print("Warm-up complete.")

    # --- Benchmarking ---
    print(f"Starting benchmark ({N_BENCHMARK_RUNS} runs)...")
    kernel_times = []
    for i in range(N_BENCHMARK_RUNS):
        # Reset dose array for each run to ensure atomics have work
        dose_device.copy_to_device(np.zeros(N_VOXELS, dtype=np.float32))
        # Optionally, reset particle states too if their evolution significantly affects runtime
        # particles_device.copy_to_device(particles_np) # More rigorous but adds overhead

        cuda.synchronize() # Ensure previous work is done and cache is clear for timing
        start_time = time.time()

        numba_simplified_transport_kernel[blockspergrid, threadsperblock](
            particles_device, dose_device, STEP_SIZE, ENERGY_LOSS_PER_STEP,
            N_STEPS_MAX, VOXEL_WIDTH, N_VOXELS
        )
        cuda.synchronize() # Wait for kernel to finish

        end_time = time.time()
        kernel_times.append(end_time - start_time)
        if (i+1) % 10 == 0:
            print(f"Completed run {i+1}/{N_BENCHMARK_RUNS}")


    avg_kernel_time = sum(kernel_times) / len(kernel_times)
    print(f"\nAverage kernel execution time: {avg_kernel_time*1000:.3f} ms "
          f"({avg_kernel_time*1e6/N_PARTICLES:.3f} ns per particle, if workload scales linearly)")


    # --- Retrieve Results ---
    print("Retrieving results from GPU...")
    dose_host = dose_device.copy_to_host()
    particles_final_host = particles_device.copy_to_host()

    # --- Analyze Results ---
    print("\n--- Results ---")
    total_deposited_dose = np.sum(dose_host)
    print(f"Total deposited dose in all voxels: {total_deposited_dose:.2f}")
    total_initial_energy = N_PARTICLES * INIT_ENERGY
    print(f"Total initial energy of all particles: {total_initial_energy:.2f}")
    if not np.isclose(total_deposited_dose, 0.0):
        print(f"Fraction of initial energy deposited (approx): {total_deposited_dose / total_initial_energy:.4f}")


    print("\nDose distribution (first 10 voxels):")
    for i in range(min(10, N_VOXELS)):
        print(f"Voxel {i}: {dose_host[i]:.2f}")

    if N_VOXELS > 10:
        print("...")
        print(f"Voxel {N_VOXELS-1}: {dose_host[N_VOXELS-1]:.2f}")

    print("\nFinal state of first 3 particles (x, y, z, E):")
    for i in range(min(3, N_PARTICLES)):
        print(f"Particle {i}: "
              f"x={particles_final_host[i, X]:.3f}, "
              f"y={particles_final_host[i, Y]:.3f}, "
              f"z={particles_final_host[i, Z]:.3f}, "
              f"E={particles_final_host[i, ENERGY]:.2f}")

if __name__ == '__main__':
    try:
        run_numba_prototype()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Numba might raise numba.cuda.cudadrv.driver.CudaAPIError for driver issues
        import traceback
        traceback.print_exc()
