import cupy
import numpy as np

# Define the CUDA C++ kernel code as a string
# Kernel processes one particle per thread for simplicity in this version.
simplified_transport_kernel_code = """
extern "C" __global__ void simplified_transport_kernel(
    float* particles,          // Array of particles: [x,y,z,dx,dy,dz,E, x,y,z,...]
    float* dose_array,         // Output array for dose deposition
    int n_particles,           // Total number of particles
    float step_size,           // Fixed step size for particle movement
    float energy_loss_per_step,// Fixed energy loss per step
    int n_steps_max,           // Maximum simulation steps per particle
    float voxel_width,         // Width of each dose voxel (along x-axis)
    int n_voxels               // Total number of dose voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_particles) {
        return;
    }

    // Each particle has 7 components: x, y, z, dx, dy, dz, energy
    int particle_stride = 7;
    int p_offset = tid * particle_stride;

    float x = particles[p_offset + 0];
    float y = particles[p_offset + 1];
    float z = particles[p_offset + 2];
    float dx = particles[p_offset + 3];
    float dy = particles[p_offset + 4];
    float dz = particles[p_offset + 5];
    float energy = particles[p_offset + 6];

    for (int step = 0; step < n_steps_max; ++step) {
        if (energy <= 0.0f) {
            break;
        }

        // Update particle position
        x += dx * step_size;
        y += dy * step_size;
        z += dz * step_size;

        // Simulate energy loss
        float deposited_energy_this_step = energy_loss_per_step;
        if (energy < energy_loss_per_step) { // Deposit remaining energy if less than typical loss
            deposited_energy_this_step = energy;
        }
        energy -= deposited_energy_this_step;

        // Simplified dose deposition (along x-axis)
        // Ensure voxel_index is within bounds [0, n_voxels - 1]
        if (x >= 0.0f) { // Assuming positive x space for voxels
            int voxel_index = static_cast<int>(floorf(x / voxel_width));
            if (voxel_index >= 0 && voxel_index < n_voxels) {
                atomicAdd(&dose_array[voxel_index], deposited_energy_this_step);
            }
        }

        // In a more complex simulation, dx, dy, dz might change due to scattering
    }

    // Update the particle array with final state (optional for this simple kernel)
    // For this example, we demonstrate updating energy and position.
    particles[p_offset + 0] = x;
    particles[p_offset + 1] = y;
    particles[p_offset + 2] = z;
    particles[p_offset + 6] = energy; // Store remaining energy
}
"""

# Create a CuPy RawKernel object
simplified_transport_kernel = cupy.RawKernel(
    simplified_transport_kernel_code,
    "simplified_transport_kernel"
)

def run_prototype():
    # --- Parameters ---
    N_PARTICLES = 1024 * 10  # Number of particles
    PARTICLE_COMPONENTS = 7  # x, y, z, dx, dy, dz, E
    N_VOXELS = 100           # Number of dose voxels
    VOXEL_WIDTH = 0.1        # Physical width of each voxel (e.g., cm)

    INIT_ENERGY = 100.0      # Initial energy for all particles
    ENERGY_LOSS_PER_STEP = 5.0 # Energy lost in one step
    STEP_SIZE = 0.05         # Simulation step size (e.g., cm)
    N_STEPS_MAX = 50         # Max steps per particle

    # --- Initialize Host Data (NumPy) ---
    print("Initializing host data...")
    particles_np = np.zeros(N_PARTICLES * PARTICLE_COMPONENTS, dtype=np.float32)

    # Example: particles starting at origin, moving along +x or randomized
    for i in range(N_PARTICLES):
        offset = i * PARTICLE_COMPONENTS
        particles_np[offset + 0] = 0.0  # x: Start at origin or slightly offset
        particles_np[offset + 1] = np.random.uniform(-0.1, 0.1) # y: small spread
        particles_np[offset + 2] = np.random.uniform(-0.1, 0.1) # z: small spread

        # Direction (dx, dy, dz) - mostly along +x
        particles_np[offset + 3] = np.random.uniform(0.8, 1.0) # dx
        particles_np[offset + 4] = np.random.uniform(-0.2, 0.2) # dy
        particles_np[offset + 5] = np.random.uniform(-0.2, 0.2) # dz
        # Normalize direction (optional for this simple sim, but good practice)
        dir_len = np.sqrt(particles_np[offset+3]**2 + particles_np[offset+4]**2 + particles_np[offset+5]**2)
        if dir_len > 0:
            particles_np[offset+3] /= dir_len
            particles_np[offset+4] /= dir_len
            particles_np[offset+5] /= dir_len

        particles_np[offset + 6] = INIT_ENERGY # Initial Energy

    # --- Transfer Data to GPU (CuPy) ---
    print("Transferring data to GPU...")
    particles_gpu = cupy.asarray(particles_np)
    dose_gpu = cupy.zeros(N_VOXELS, dtype=cupy.float32)

    # --- Kernel Launch Configuration ---
    threads_per_block = 256
    # Ensure grid_size is an int. Using tuple for block/grid dimensions.
    block_dim = (threads_per_block,)
    grid_dim = ((N_PARTICLES + threads_per_block - 1) // threads_per_block,)

    print(f"Launching kernel with grid_dim={grid_dim}, block_dim={block_dim}")
    print(f"Total threads: {grid_dim[0] * block_dim[0]}")

    # --- Launch Kernel ---
    simplified_transport_kernel(
        grid_dim,
        block_dim,
        (
            particles_gpu,
            dose_gpu,
            N_PARTICLES,
            STEP_SIZE,
            ENERGY_LOSS_PER_STEP,
            N_STEPS_MAX,
            VOXEL_WIDTH,
            N_VOXELS
        )
    )

    # --- Synchronize and Retrieve Results ---
    print("Kernel execution finished. Synchronizing...")
    cupy.cuda.runtime.deviceSynchronize() # Ensure kernel completion

    print("Retrieving results from GPU...")
    dose_cpu = dose_gpu.get()
    particles_final_cpu = particles_gpu.get() # Get final particle states

    # --- Analyze Results ---
    print("\n--- Results ---")
    total_deposited_dose = np.sum(dose_cpu)
    print(f"Total deposited dose in all voxels: {total_deposited_dose:.2f}")

    # Sanity check: expected total energy deposited if all particles lose all energy
    # This is an approximation as some energy might be lost outside defined voxels or particles stop early
    total_initial_energy = N_PARTICLES * INIT_ENERGY
    print(f"Total initial energy of all particles: {total_initial_energy:.2f}")
    if not np.isclose(total_deposited_dose, 0.0): # Avoid printing if no dose (e.g., all particles missed voxels)
         print(f"Fraction of initial energy deposited (approx): {total_deposited_dose / total_initial_energy:.4f}")


    print("\nDose distribution (first 10 voxels):")
    for i in range(min(10, N_VOXELS)):
        print(f"Voxel {i}: {dose_cpu[i]:.2f}")

    if N_VOXELS > 10:
        print("...")
        print(f"Voxel {N_VOXELS-1}: {dose_cpu[N_VOXELS-1]:.2f}")

    print("\nFinal state of first 3 particles (x, y, z, E):")
    for i in range(min(3, N_PARTICLES)):
        offset = i * PARTICLE_COMPONENTS
        print(f"Particle {i}: "
              f"x={particles_final_cpu[offset+0]:.3f}, "
              f"y={particles_final_cpu[offset+1]:.3f}, "
              f"z={particles_final_cpu[offset+2]:.3f}, "
              f"E={particles_final_cpu[offset+6]:.2f}")

if __name__ == '__main__':
    try:
        run_prototype()
    except Exception as e:
        print(f"An error occurred: {e}")
        # In case of CUDA errors, CuPy often raises cupy.cuda.runtime.CUDARuntimeError
        # or similar, which will be caught here.
        import traceback
        traceback.print_exc()
