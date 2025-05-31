#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::array automatic conversions
#include <pybind11/chrono.h> // For seeding with time

// Assuming MOQUI headers are in the include path provided to the compiler
// e.g., -I/path/to/moqui/include or similar
#include "moqui/base/distributions/mqi_phsp6d.hpp"
// mqi_phsp6d.hpp should include mqi_pdfMd.hpp and other necessary headers like <array>, <random>

#include <random> // For std::default_random_engine
#include <chrono> // For std::chrono for seeding

namespace py = pybind11;

// Helper function or lambda to provide the 'sample' method
// This keeps the original mqi::phsp_6d<float> class unchanged.
// It creates a new RNG engine for each call to sample().
// For more performance-critical applications, one might consider
// allowing the user to provide a seed, or managing the RNG state differently.
std::array<float, 6> sample_phsp6d(mqi::phsp_6d<float>& phsp_dist) {
    // Seed with current time for variety in samples across runs / instances
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    return phsp_dist(&rng); // Call the original operator()
}

PYBIND11_MODULE(mqi_distributions_bind, m) {
    m.doc() = "Pybind11 bindings for MOQUI phsp_6d distribution";

    py::class_<mqi::phsp_6d<float>>(m, "Phsp6DFloat")
        .def(py::init<const std::array<float, 6>&,
                      const std::array<float, 6>&,
                      const std::array<float, 2>&>(),
             py::arg("mean"), py::arg("sigma"), py::arg("rho"),
             "Constructor for Phsp6DFloat.\n\n"
             "Args:\n"
             "  mean (list[float[6]]): Mean values (x, y, z, theta_x, theta_y, theta_z).\n"
             "  sigma (list[float[6]]): Sigma values (std devs for x, y, z, theta_x, theta_y, theta_z).\n"
             "  rho (list[float[2]]): Correlation coefficients (rho_x, rho_y).")
        // Bind the original operator() if direct access to it with an external RNG is desired (advanced)
        // .def("__call__", [](mqi::phsp_6d<float>& self, std::default_random_engine* rng_ptr) {
        //     if (!rng_ptr) throw std::runtime_error("RNG engine pointer cannot be null.");
        //     return self(rng_ptr);
        // }, py::arg("rng_engine"), "Call the phase space generator with a user-provided RNG engine.")
        .def("sample", &sample_phsp6d,
             "Generate a phase-space sample using an internally managed RNG.\n\n"
             "Returns:\n"
             "  list[float[6]]: A 6D phase-space sample [x, y, z, theta_x, theta_y, theta_z].")
        // Expose members if needed, e.g., for inspection, though not requested.
        // .def_property_readonly("mean", [](const mqi::phsp_6d<float>& self) { return self.mean(); }) // Assuming getters exist
        // .def_property_readonly("sigma", [](const mqi::phsp_6d<float>& self) { return self.sigma(); })
        // .def_property_readonly("rho", [](const mqi::phsp_6d<float>& self) { return self.rho(); })
        ;

    // If mqi::pdf_Md<float, 6> were needed for type casting or other reasons:
    // py::class_<mqi::pdf_Md<float, 6>, std::shared_ptr<mqi::pdf_Md<float, 6>>>(m, "PdfMdFloat6")
    //    .def("operator()", py::overload_cast<std::default_random_engine*>(&mqi::pdf_Md<float, 6>::operator()), "Abstract operator()");
    // Note: Binding abstract classes requires care, often a trampoline class is needed if Python classes will inherit from it.
    // Here, Phsp6DFloat is concrete, so direct binding is simpler.
}

/*
================================================================================
EXAMPLE COMPILATION (Linux/macOS with g++):
================================================================================

Ensure Pybind11 is installed (e.g., `pip install pybind11`).

1. Locate Pybind11 headers:
   Run: `python3 -m pybind11 --includes`
   This will output something like: -I/path/to/python/include -I/path/to/pybind11/include

2. Locate MOQUI headers:
   You need to provide the path to the directory containing `moqui/base/distributions/mqi_phsp6d.hpp`.
   Let's assume this path is `../MOQUI_SRC/`. (Adjust as necessary)

3. Compilation command:
   g++ -O3 -Wall -shared -std=c++17 -fPIC \
       $(python3 -m pybind11 --includes) \
       -I../MOQUI_SRC \  // Adjust this path to your MOQUI source directory
       mqi_phsp_6d_bindings.cpp \
       -o mqi_distributions_bind$(python3-config --extension-suffix)

   Explanation:
   - `-O3`: Optimization level.
   - `-Wall`: Enable all warnings.
   - `-shared`: Create a shared library.
   - `-std=c++17`: Use C++17 standard (or newer if MOQUI requires).
   - `-fPIC`: Position Independent Code (required for shared libraries).
   - `$(python3 -m pybind11 --includes)`: Gets include paths for Python and Pybind11.
   - `-I../MOQUI_SRC`: Tells the compiler where to find MOQUI headers. **MODIFY THIS PATH**.
   - `mqi_phsp_6d_bindings.cpp`: Your binding code file.
   - `-o mqi_distributions_bind$(python3-config --extension-suffix)`: Output file name.
     `python3-config --extension-suffix` provides the correct suffix (e.g., .so, .pyd).

================================================================================
EXAMPLE PYTHON USAGE:
================================================================================

# Save this as example.py in the same directory as the compiled module:
# (After compiling mqi_phsp_6d_bindings.cpp to mqi_distributions_bind.so or similar)

# import mqi_distributions_bind as mdb
# import numpy as np # For convenience if using numpy arrays later

# def run_example():
#     print("Testing Phsp6DFloat bindings...")

#     mean = [0.1, 0.2, 0.3, 0.01, 0.02, 0.0]  # theta_z mean typically 0
#     sigma = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001] # sigma[5] not used for phsp[5] value
#     rho = [0.5, -0.5]

#     print(f"Mean: {mean}")
#     print(f"Sigma: {sigma}")
#     print(f"Rho: {rho}")

#     try:
#         phsp_generator = mdb.Phsp6DFloat(mean, sigma, rho)
#         print("Phsp6DFloat object created successfully.")
#     except Exception as e:
#         print(f"Error creating Phsp6DFloat object: {e}")
#         return

#     print("\nGenerating a few samples:")
#     for i in range(5):
#         try:
#             sample = phsp_generator.sample()
#             # The sample is std::array<float, 6>, converted to a Python list by Pybind11
#             print(f"Sample {i+1}: {sample}")
#             # Example: Convert to numpy array for easier manipulation if needed
#             # sample_np = np.array(sample)
#             # print(f"Sample {i+1} (numpy): {sample_np}")
#         except Exception as e:
#             print(f"Error generating sample {i+1}: {e}")

# if __name__ == '__main__':
#     run_example()

*/
