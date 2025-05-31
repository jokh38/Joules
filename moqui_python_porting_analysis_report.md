# MOQUI Python Porting Analysis Report

## 1. Introduction

**Project Goals**: This project aimed to analyze the feasibility and lay the groundwork for porting parts of the MOQUI (Monte Carlo toolkit for research in radiation therapy) codebase, currently implemented in C++/CUDA within the `jokh38/Joules` repository, to a more Python-centric environment. The primary goal is to enhance usability, simplify workflows, and leverage Python's extensive scientific ecosystem while retaining the performance of critical C++/CUDA components.

**Scope**: The analysis involved:
*   Understanding the structure of the existing MOQUI C++/CUDA code.
*   Prototyping key functionalities in Python (e.g., phase-space generation, MHA file reading).
*   Evaluating performance characteristics of Python implementations (NumPy vs. Numba).
*   Assessing memory usage of Python prototypes.
*   Developing Pybind11 wrappers for existing C++ classes.
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

## 4. Core Operations Porting Strategy

Based on the analysis and prototyping, a **hybrid porting strategy** is recommended. This approach aims to balance the benefits of Python's ease of use and ecosystem with the performance of C++/CUDA for critical computations.

### 4.1. Rationale for the Hybrid Approach

*   **Performance Preservation**: MOQUI's core strength lies in its high-performance Monte Carlo simulations. GPU kernels and heavily optimized C++ CPU code should be preserved.
*   **Enhanced Usability & Flexibility**: Python offers a more accessible environment for scripting, data analysis, visualization, and integration with other tools.
*   **Development Velocity**: Porting non-critical modules (e.g., UI, complex configuration, post-processing scripts) to Python can accelerate development and maintenance.
*   **Reduced Custom Code**: Leveraging libraries like SimpleITK, NumPy, SciPy, etc., can reduce the amount of custom C++ code needed for tasks like file I/O, data manipulation, and numerical algorithms.

### 4.2. Strategy for GPU Kernels (C++/CUDA Maintained, Python Control)

*   **Maintain**: Existing CUDA kernels should be maintained in `.cu` / `.cuh` files.
*   **Interface**: Develop Python interfaces to these kernels using Pybind11 or direct CFFI/ctypes wrappers if simpler for specific functions. Pybind11 is generally preferred for C++ classes.
*   **Data Flow**: Manage data transfers (CPU to GPU, GPU to CPU) via these Python interfaces. NumPy arrays can be efficiently passed to C++ and then to CUDA. Libraries like CuPy can also be considered for direct GPU array manipulation in Python, interoperating with existing CUDA code.

### 4.3. Strategy for CPU-Bound Performance-Sensitive Code

*   **Option A: Numba-Optimized Python**: For C++ code that is complex but can be expressed with NumPy operations or Python loops, consider rewriting in Python and optimizing with Numba. The `phsp_6d` example showed this can yield excellent performance.
*   **Option B: C++ with Python Wrappers (Pybind11)**: For existing, highly optimized C++ CPU code that is difficult or risky to rewrite, use Pybind11 to create Python wrappers. This retains C++ performance while allowing Python interaction.
*   **Decision Criteria**: Choose between A and B based on code complexity, existing optimization level, and developer familiarity.

### 4.4. Strategy for Non-Critical Modules

*   **Convert to Python**: Modules like user interfaces (if any planned for Python), higher-level scripting logic, configuration management, data analysis scripts, and simple file handlers can be largely converted to Python.
*   **Leverage Python Libraries**: Utilize the rich Python ecosystem (e.g., `h5py` for HDF5, `matplotlib` for plotting, `pandas` for data analysis, `SimpleITK` for image I/O).

## 5. Specific Recommendations

### 5.1. Short-Term Action Plan

1.  **Benchmark Pybind11 Wrapper**: Compile the `mqi_phsp_6d_bindings.cpp` and benchmark the `Phsp6DFloat.sample()` method. Compare its performance against the pure C++ version and the Numba version to quantify overhead.
2.  **Interface Core CUDA Kernels**: Identify a small set of critical CUDA kernels. Develop Pybind11 or CFFI interfaces for them. Test data transfer and kernel execution from Python.
3.  **Convert a Non-Critical Utility**: Select a small, self-contained C++ utility or file handler (beyond MHA) and port it entirely to Python. This will provide further experience with the porting process.
4.  **Establish Build System**: Set up a build system (e.g., CMake with Pybind11 support) that can compile the C++ wrappers and integrate with the Python parts of the project.

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

## 6. Conclusion

The hybrid Python porting approach for MOQUI appears highly feasible and offers substantial benefits. By retaining performance-critical C++/CUDA code and wrapping it for Python control, MOQUI can leverage Python's ease of use, extensive libraries, and modern development practices. Prototyping efforts have demonstrated that:
*   Python with NumPy and Numba can achieve excellent performance for numerical tasks.
*   Pybind11 provides an effective mechanism for integrating existing C++ code.
*   Python libraries can replace custom C++ solutions for common tasks like file I/O.

This strategy will make MOQUI more accessible to a wider range of users and developers, facilitate faster prototyping of new ideas, and improve overall maintainability and extensibility without sacrificing the core computational power of the toolkit. The recommended short-term actions will further validate this approach and pave the way for a successful, incremental porting effort.
