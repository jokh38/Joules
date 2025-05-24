cmake_minimum_required(VERSION 3.10)
project(moquimc LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14) # Or the appropriate CUDA standard if known
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include utilities
include(CMakePrintHelpers)

# Find Dependencies
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: TRUE")
    message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDA is required but was not found.")
endif()

find_package(GDCM REQUIRED)
if(GDCM_FOUND)
    message(STATUS "GDCM found: TRUE")
    message(STATUS "GDCM version: ${GDCM_VERSION}")
    message(STATUS "GDCM include dirs: ${GDCM_INCLUDE_DIRS}")
    message(STATUS "GDCM libraries: ${GDCM_LIBRARIES}")
else()
    message(FATAL_ERROR "GDCM is required but was not found.")
endif()

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "ZLIB found: TRUE")
    message(STATUS "ZLIB version: ${ZLIB_VERSION_STRING}")
    message(STATUS "ZLIB include_dirs: ${ZLIB_INCLUDE_DIRS}")
    message(STATUS "ZLIB libraries: ${ZLIB_LIBRARIES}")
else()
    message(FATAL_ERROR "ZLIB is required but was not found.")
endif()

# Add the library target (INTERFACE library for header-only)
add_library(moquimc_lib INTERFACE)

# Specify include directories for the library
target_include_directories(moquimc_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/base"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_functions"
    "${CMAKE_CURRENT_SOURCE_DIR}/treatment_machines"
    ${CUDA_INCLUDE_DIRS}  # Add CUDA include directory
    ${GDCM_INCLUDE_DIRS}  # Add GDCM include directory
    ${ZLIB_INCLUDE_DIRS}  # Add ZLIB include directory
)

# Link dependencies to the library target
# For an INTERFACE library, these are effectively propagated to consumers
target_link_libraries(moquimc_lib INTERFACE
    CUDA::cudart            # For CUDA runtime
    CUDA::curand            # For CUDA random number generation if mqi_rng uses it directly from kernels
    # CUDA::cublas         # If cublas is used directly, not apparent from headers
    GDCM::GDCM              # This is typically how GDCM provides its targets
    ZLIB::ZLIB              # This is typically how ZLIB provides its target
)

# Configure CUDA Architectures
option(MOQUIMC_CUDA_ARCHITECTURES "Build for specific CUDA architectures (e.g., \"70;75;86\")." "70;75;80;86")
if(NOT CMAKE_CUDA_ARCHITECTURES AND CUDA_FOUND) # Ensure CUDA_FOUND before using CUDA specific variables
    set(CMAKE_CUDA_ARCHITECTURES ${MOQUIMC_CUDA_ARCHITECTURES})
endif()

# The CUDA_ARCHITECTURES property will be automatically picked up by CUDA-aware targets.
# For an INTERFACE library, this primarily serves to inform consuming targets or if this library
# itself were to compile CUDA code directly in the future.
# If moquimc_lib itself had .cu files or CUDA_PTX_COMPILATION, then this would directly affect its compilation.
# As an INTERFACE library, it's more about setting a project-wide default if not already set.
message(STATUS "Configured CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Print a summary
cmake_print_variables(CUDA_FOUND CUDA_VERSION_STRING CUDA_TOOLKIT_ROOT_DIR CUDA_INCLUDE_DIRS CUDA_LIBRARIES CMAKE_CUDA_ARCHITECTURES)
cmake_print_variables(GDCM_FOUND GDCM_VERSION GDCM_INCLUDE_DIRS GDCM_LIBRARIES)
cmake_print_variables(ZLIB_FOUND ZLIB_VERSION_STRING ZLIB_INCLUDE_DIRS ZLIB_LIBRARIES)

message(STATUS "moquimc_lib INTERFACE library created.")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/base;${CMAKE_CURRENT_SOURCE_DIR}/kernel_functions;${CMAKE_CURRENT_SOURCE_DIR}/treatment_machines;${CUDA_INCLUDE_DIRS};${GDCM_INCLUDE_DIRS};${ZLIB_INCLUDE_DIRS}")
message(STATUS "Linked libraries: CUDA::cudart, CUDA::curand, GDCM::GDCM, ZLIB::ZLIB")

# Google Test Integration
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    release-1.12.1 # Or a more recent stable tag
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

message(STATUS "Google Test integration complete.")

# Add tests subdirectory
add_subdirectory(tests)
