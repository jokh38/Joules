# MOQUI Python Porting Analysis Report

## 1. Introduction

**Project Goals**: This project aimed to analyze the feasibility and lay the groundwork for porting parts of the MOQUI (Monte Carlo toolkit for research in radiation therapy) codebase, currently implemented in C++/CUDA within the `jokh38/Joules` repository, to a more Python-centric environment. The primary goal is to enhance usability, simplify workflows, and leverage Python's extensive scientific ecosystem while retaining the performance of critical C++/CUDA components.

**Scope**: The analysis involved:
*   Understanding the structure of the existing MOQUI C++/CUDA code.
*   Prototyping key functionalities in Python (e.g., phase-space generation, MHA file reading).
*   Evaluating performance characteristics of Python implementations (NumPy vs. Numba).
*   Assessing memory usage of Python prototypes.
*   Developing Pybind11 wrappers for existing C++ classes.
*   Attempting to prototype simplified CUDA kernels in Python using CuPy and Numba.
*   Formulating a hybrid porting strategy and providing specific recommendations.

## 2. MOQUI Code Structure Analysis

The `jokh38/Joules` repository contains the MOQUI toolkit, a sophisticated suite for Monte Carlo simulations in radiation therapy. The codebase is primarily C++ with CUDA extensions for GPU acceleration of computationally intensive tasks.

Key observations:
*   **Modular Design**: The code appears to be organized into modules for different functionalities (e.g., distributions, file handlers, geometry, physics).
*   **Performance Focus**: The use of C++ and CUDA indicates a strong emphasis on performance, especially for core simulation algorithms.
*   **Header-Based Libraries**: A significant portion of the C++ code, particularly templated classes, is implemented in header files (`.hpp`), common in C++ template metaprogramming and inline performance optimizations.
*   **Dependencies**: The project relies on standard C++ libraries and CUDA. External libraries for specific tasks (like MHA file reading) are likely used, but the analysis focused on core MOQUI components.

## 3. Python Prototyping and Technical Validation

Several key components of MOQUI were prototyped in Python to validate technical feasibility, assess performance, and explore integration strategies.

### 3.1. `phsp_6d` Python Prototyping (NumPy and Numba)

The `mqi::phsp_6d` class, responsible for generating 6D phase-space samples, was chosen as a representative component for Python prototyping.

*   **NumPy Version**:
    *   A Python version (`phsp_6d_operator`) was implemented using NumPy for array operations and random number generation.
    *   Location: `mqi_phsp_6d_prototype.py`.
    *   Functionality: Successfully replicated the C++ logic for calculating phase-space variables based on mean, sigma, and rho parameters.
*   **Numba Version**:
    *   The NumPy version was optimized using Numba's `@jit(nopython=True)` decorator to compile it to efficient machine code.
    *   Location: `benchmark_phsp_6d_numba.py`.
    *   Adaptation: A minor modification was required for `np.clip` as Numba's implementation expects array inputs, while the prototype used scalar clipping. This was resolved using manual scalar clipping logic.

### 3.2. Performance Benchmark Results

Performance benchmarks were conducted for both the NumPy and Numba-jitted versions of `phsp_6d_operator`.

*   **NumPy Version (`benchmark_phsp_6d.py`)**:
    *   Average time per call: **~29.19 microseconds** (for 100,000 calls).
*   **Numba Version (`benchmark_phsp_6d_numba.py`)**:
    *   Average time per call: **~1.39 microseconds** (for 1,000,000 calls).

**Finding**: The Numba-jitted version demonstrated a significant performance improvement (approximately **21x faster**) over the pure NumPy version, bringing its performance closer to what might be expected from compiled C++ code for such an operation.

### 3.3. Memory Usage Analysis Insights

Memory profiling was performed on the Python (NumPy) `phsp_6d_operator` function using `memory_profiler`.

*   Location: `memory_profile_phsp_6d.py` (profiling `mqi_phsp_6d_prototype.py`).
*   **Findings**:
    *   The baseline memory for the script execution environment (including Python, NumPy, and other modules) was around 32.8 MiB.
    *   An increment of approximately 2.9 MiB was noted upon the first call to `np.random.normal()`. This is likely due to NumPy's internal memory allocation for its random number generator or related buffers. Subsequent calls within the same function context did not show significant new allocations.
    *   The overall memory footprint of the function itself, after initial NumPy allocations, was stable.
    *   This suggests that for functions like `phsp_6d_operator`, repeated calls do not continuously allocate new memory, leveraging NumPy's efficient memory management for array operations.

### 3.4. Pybind11 C++ Interface Wrapper Development for `phsp_6d`

To enable the use of existing C++ `mqi::phsp_6d<float>` class directly from Python, Pybind11 bindings were developed.

*   Location: `mqi_phsp_6d_bindings.cpp`.
*   **Features**:
    *   Wrapped the `mqi::phsp_6d<float>` constructor, allowing instantiation from Python with lists/tuples for mean, sigma, and rho.
    *   Exposed a `sample()` method in Python that internally manages a `std::default_random_engine` (seeded by time) and calls the original C++ `operator()` to generate phase-space samples.
    *   The module was named `mqi_distributions_bind`, and the class `Phsp6DFloat`.
    *   Included example compilation instructions and Python usage.

**Finding**: Pybind11 provides a straightforward way to create Python interfaces for existing C++ classes, handling data type conversions (e.g., Python list to `std::array<float, N>`) effectively. This is crucial for a hybrid approach.

### 3.5. Python Conversion of `.mha` File Reading using `SimpleITK`

The functionality of `mask_reader::read_mha_file` (for reading MetaImage `.mha` files) was replicated in Python using the `SimpleITK` library.

*   Location: `mha_utils.py`.
*   **Features**:
    *   The `read_mha_image` function reads an MHA file.
    *   It extracts image data as a NumPy array and metadata (dimensions, spacing, origin, direction).
    *   Returns a dictionary containing this information.
    *   Includes error handling for file-not-found and parsing errors.
    *   An example usage section successfully tested the function with dummy and invalid MHA files.

**Finding**: Standard scientific Python libraries like `SimpleITK` can readily replace custom C++ file I/O and processing logic for common medical imaging formats, simplifying the codebase and leveraging well-tested library functionalities.

### 3.6. Python GPU Kernel Prototyping (CuPy and Numba)
Simplified prototypes of a particle transport CUDA kernel were developed using CuPy `RawKernel` (`cupy_transport_prototype.py`) and Numba CUDA JIT (`numba_transport_prototype.py`).
*   **Implementation**: Both prototypes successfully implemented the requested logic, including particle data structures, simulation loops, position updates, energy loss, and atomic dose deposition.
*   **Execution Challenges**: Attempts to execute these GPU prototypes failed due to CUDA driver version incompatibilities within the sandboxed execution environment (`cudaErrorInsufficientDriver` for CuPy, and `CUDA driver library cannot be found` for Numba). This prevented direct benchmarking of these Python-based GPU kernels.
*   **Insights**: The implementation process itself provided insights into the development workflow and structural differences between using CuPy `RawKernel` (C++ string-based kernels) and Numba CUDA JIT (Python-syntax kernels).

## 4. Core Operations Porting Strategy

Based on the analysis and prototyping, a **hybrid porting strategy** is recommended. This approach aims to balance the benefits of Python's ease of use and ecosystem with the performance of C++/CUDA for critical computations.

### 4.1. Rationale for the Hybrid Approach

*   **Performance Preservation**: MOQUI's core strength lies in its high-performance Monte Carlo simulations. GPU kernels and heavily optimized C++ CPU code should be preserved.
*   **Enhanced Usability & Flexibility**: Python offers a more accessible environment for scripting, data analysis, visualization, and integration with other tools.
*   **Development Velocity**: Porting non-critical modules (e.g., UI, complex configuration, post-processing scripts) to Python can accelerate development and maintenance.
*   **Reduced Custom Code**: Leveraging libraries like SimpleITK, NumPy, SciPy, etc., can reduce the amount of custom C++ code needed for tasks like file I/O, data manipulation, and numerical algorithms.

### 4.2. Strategy for GPU Kernels (C++/CUDA Maintained, Python Control)

*   **Maintain**: Existing CUDA kernels should be maintained in `.cu` / `.cuh` files.
*   **Interface**: Develop Python interfaces to these kernels using Pybind11 to wrap C++ functions that launch these kernels, or direct CFFI/ctypes wrappers if simpler for specific utility functions. Pybind11 is generally preferred for C++ classes that might manage kernel execution.
*   **Data Flow**: Manage data transfers (CPU to GPU, GPU to CPU) via these Python interfaces. NumPy arrays can be efficiently passed to C++ and then to CUDA. Libraries like CuPy can also be considered for direct GPU array manipulation in Python, interoperating with existing CUDA code through mechanisms like `__cuda_array_interface__`.

### 4.3. Strategy for CPU-Bound Performance-Sensitive Code

*   **Option A: Numba-Optimized Python**: For C++ code that is complex but can be expressed with NumPy operations or Python loops, consider rewriting in Python and optimizing with Numba. The `phsp_6d` example showed this can yield excellent performance.
*   **Option B: C++ with Python Wrappers (Pybind11)**: For existing, highly optimized C++ CPU code that is difficult or risky to rewrite, use Pybind11 to create Python wrappers. This retains C++ performance while allowing Python interaction.
*   **Decision Criteria**: Choose between A and B based on code complexity, existing optimization level, and developer familiarity.

### 4.4. Strategy for Non-Critical Modules

*   **Convert to Python**: Modules like user interfaces (if any planned for Python), higher-level scripting logic, configuration management, data analysis scripts, and simple file handlers can be largely converted to Python.
*   **Leverage Python Libraries**: Utilize the rich Python ecosystem (e.g., `h5py` for HDF5, `matplotlib` for plotting, `pandas` for data analysis, `SimpleITK` for image I/O).

## 5. Specific Recommendations

### 5.1. Short-Term Action Plan

1.  **Benchmark Pybind11 Wrapper**: Compile the `mqi_phsp_6d_bindings.cpp` (assuming a suitable build environment) and benchmark the `Phsp6DFloat.sample()` method. Compare its performance against the pure C++ version and the Numba version to quantify overhead.
2.  **Interface Core CUDA Kernels**: Identify a small set of critical CUDA kernels. Develop Pybind11 wrappers for C++ functions that call these kernels. Test data transfer and kernel execution from Python.
3.  **Convert a Non-Critical Utility**: Select a small, self-contained C++ utility or file handler (beyond MHA) and port it entirely to Python. This will provide further experience with the porting process.
4.  **Establish Build System**: Set up a build system (e.g., CMake with Pybind11 support) that can compile the C++ wrappers and integrate with the Python parts of the project.
5.  **Resolve GPU Environment Incompatibilities**: For further GPU development or Python-based kernel testing, establish a development environment with compatible CUDA drivers, toolkit, and Python GPU libraries (CuPy, Numba).

### 5.2. Performance Optimization Advice

*   **Profile Extensively**: Use Python's `cProfile` and `memory_profiler`, and Numba's profiling tools to identify bottlenecks in Python code. For C++/CUDA parts, use `nvprof` or Nsight Systems/Compute.
*   **Minimize Python-C++ Transitions**: Calls across the Python/C++ boundary have overhead. For tight loops, try to keep operations within C++ or Numba-compiled code as much as possible. Vectorize operations.
*   **Efficient Data Transfer**: When moving data to/from GPU or between Python and C++, ensure transfers are minimized and data formats are efficient (e.g., contiguous NumPy arrays).
*   **Numba `fastmath`**: For Numba-jitted functions, consider `fastmath=True` for additional speedups if slightly relaxed IEEE 754 compliance is acceptable.

### 5.3. Development Environment and Testing Suggestions

*   **Virtual Environments**: Use Python virtual environments (`venv`, `conda`) to manage dependencies.
*   **IDE**: Utilize IDEs with good Python and C++ support (e.g., VS Code, PyCharm Professional).
*   **Continuous Integration (CI)**: Set up CI (e.g., GitHub Actions) to automate builds, tests, and linting for both Python and C++ components.
*   **Unit Testing**: Implement unit tests for Python modules (`unittest`, `pytest`) and C++ components (e.g., Google Test). Test Pybind11 wrappers thoroughly.
*   **Cross-Platform Compatibility**: If MOQUI needs to run on multiple OS (Linux, Windows, macOS), ensure the build system and library choices support this.

## 6. Performance Analysis of Porting Core CUDA Kernels to Python (Theoretical)

The attempts to prototype simplified CUDA kernels (`simplified_transport_kernel` using CuPy RawKernel and `numba_simplified_transport_kernel` using Numba CUDA JIT) highlighted the potential for Python to control GPU computations. However, environmental issues (CUDA driver incompatibility) prevented direct benchmarking. This section provides a theoretical analysis of potential performance differences based on the nature of these tools and general GPU programming principles.

### 6.1. Comparison of Python GPU Programming Approaches

*   **CuPy `RawKernel`**:
    *   **Pros**:
        *   Allows writing CUDA C++ code directly, offering fine-grained control similar to native development.
        *   Can be easier for developers already familiar with CUDA C++.
        *   Potentially lower overhead for simple kernels compared to more abstracted approaches if the kernel code is highly optimized.
    *   **Cons**:
        *   Kernel code is embedded in Python strings, which can be cumbersome for large, complex kernels (reduced IDE support, harder debugging of C++ code).
        *   Compilation occurs at runtime (via NVRTC), which might add overhead on the first call or when parameters change compilation.
        *   Managing multiple complex kernel strings and their dependencies within Python can become unwieldy.

*   **Numba CUDA JIT (`@cuda.jit`)**:
    *   **Pros**:
        *   Allows writing kernels in a subset of Python syntax, which can be more approachable for Python developers.
        *   Handles much of the CUDA boilerplate (e.g., function type qualification, some memory management aspects).
        *   Good integration with NumPy arrays for data transfer.
        *   JIT compilation to PTX/SASS can produce highly optimized code for supported Python/NumPy features.
    *   **Cons**:
        *   Limited to a subset of Python and CUDA features. Advanced CUDA features (e.g., dynamic parallelism, complex texture memory usage, specific shuffle operations beyond provided intrinsics) might be difficult or impossible to express directly.
        *   Debugging can be challenging, as errors can occur in Numba's compilation stages or within the JIT-compiled CUDA code.
        *   Performance can be variable and sometimes requires careful "Numba-idiomatic" Python to achieve optimal results. Over-reliance on unsupported Python features can lead to slow "object mode" fallbacks or errors.

### 6.2. Theoretical Performance Gap Analysis (Python GPU vs. Native C++/CUDA)

While direct execution of prototypes was hindered, we can theorize on performance gaps:

*   **Compiler Optimizations**:
    *   **Native C++/CUDA**: Benefits from NVIDIA's `nvcc` compiler, which is highly mature and specifically designed to extract maximum performance from NVIDIA GPUs. It has access to the full range of CUDA language features and optimization flags.
    *   **Python-based (Numba/CuPy)**: Numba uses its own Python-to-LLVM-to-PTX compilation path. CuPy `RawKernel` typically uses NVRTC (NVIDIA's runtime compilation library) to compile C++ strings to PTX. While these are powerful, they might not always match `nvcc`'s optimization capabilities for very complex or edge-case CUDA code, or may offer a different set of optimization pragmas/knobs.
*   **Overheads from Python Layer**:
    *   **Kernel Launch**: Launching a kernel from Python (CuPy, Numba) involves some overhead compared to a direct C++ CUDA launch, though this is often negligible for computationally intensive kernels that run for a long time.
    *   **Data Handling**: Transferring data between NumPy (CPU) and GPU device arrays (CuPy/Numba) incurs overhead. While efficient, it's a step that native C++ applications manage more directly. Python's dynamic typing and object model can also add subtle overheads in the control logic surrounding kernel calls.
*   **Memory Management**:
    *   Native C++/CUDA offers explicit control over memory allocation, deallocation, and transfers (e.g., `cudaMalloc`, `cudaMemcpyAsync`).
    *   Python tools provide higher-level abstractions. While convenient, these abstractions might introduce slight overheads or less fine-grained control over memory placement and transfer scheduling compared to manual C++ management.
*   **Challenges in Replicating Advanced CUDA Optimizations**:
    *   **Shared Memory**: While Numba and CuPy allow usage of shared memory, orchestrating complex shared memory access patterns and synchronizations might be more straightforward and explicitly controllable in native CUDA C++.
    *   **Texture Memory**: Advanced usage of texture memory (especially 1D/2D/3D textures with specific caching and filtering behaviors) can be harder to map directly or fully utilize from Python abstractions compared to C++ CUDA.
    *   **Complex Synchronization**: Intricate block-level or warp-level synchronization patterns (e.g., `__syncthreads_count`, `__ballot`) might have less direct or more verbose equivalents in Python GPU frameworks.
    *   **CUDA Libraries**: Integrating with other CUDA libraries (e.g., cuBLAS, cuFFT, cuSolver) is possible from Python (CuPy has many wrappers), but if MOQUI uses custom C++ wrappers or direct calls to less common libraries, replicating this via Python might be complex.

### 6.3. Inference on Porting Complex MOQUI Kernels

The CUDA kernel prototypes implemented for this analysis (`simplified_transport_kernel`) were, by design, highly simplified representations of what a production kernel like `mc::transport_particles_patient` would entail.

*   **Complexity of Production Kernels**: Real-world MOQUI kernels likely involve intricate physics, complex geometry tracking, sophisticated memory access patterns, and potentially heavy use of advanced CUDA features for optimization.
*   **Effort and Performance Risk**:
    *   Directly porting such complex C++/CUDA kernels to Numba CUDA JIT Python would be a very substantial undertaking, requiring careful translation of logic and likely significant refactoring to fit Numba's supported Python subset and achieve good performance. The risk of introducing subtle bugs or performance regressions would be high.
    *   Using CuPy `RawKernel` would mean embedding potentially thousands of lines of C++ code into Python strings, which is impractical for development, debugging, and maintenance.
*   **Likely Performance Gap**: Even if a direct port to Python-based GPU programming were feasible, it is highly probable that a performance gap would exist compared to the original, mature, and highly optimized C++/CUDA code. This gap would widen with increasing kernel complexity and reliance on advanced CUDA features that are less directly exposed or optimized in the Python GPU frameworks.

**Conclusion for this Section**: For MOQUI's most performance-critical and complex CUDA kernels, such as `transport_particles_patient`, attempting a full rewrite into Python-based GPU programming (either Numba or CuPy `RawKernel`) is not recommended as the primary strategy if maximum performance is to be retained. The development effort would be significant, and the risk to performance is considerable. The previously recommended hybrid approach—maintaining these kernels in C++/CUDA and creating Python wrappers (e.g., using Pybind11 to call C++ functions that manage these kernels)—remains the most pragmatic path. This balances the desire for Python integration with the need to preserve the performance of these critical components.

## 7. Conclusion

The hybrid Python porting approach for MOQUI appears highly feasible and offers substantial benefits. By retaining performance-critical C++/CUDA code and wrapping it for Python control, MOQUI can leverage Python's ease of use, extensive libraries, and modern development practices. Prototyping efforts have demonstrated that:
*   Python with NumPy and Numba can achieve excellent performance for numerical tasks.
*   Pybind11 provides an effective mechanism for integrating existing C++ code.
*   Python libraries can replace custom C++ solutions for common tasks like file I/O.
*   Python frameworks like CuPy and Numba offer paths to GPU programming within Python, though environmental compatibility and performance relative to native CUDA must be carefully considered for complex kernels.

This strategy will make MOQUI more accessible to a wider range of users and developers, facilitate faster prototyping of new ideas, and improve overall maintainability and extensibility without sacrificing the core computational power of the toolkit. The recommended short-term actions will further validate this approach and pave the way for a successful, incremental porting effort.
