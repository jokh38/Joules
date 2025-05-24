#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create the build directory if it doesn't exist
mkdir -p build

# Change to the build directory
cd build

# Run CMake to configure the project, passing any additional arguments
echo "Configuring project with CMake..."
cmake .. "$@"

# Run Make to compile the project using all available processors
echo "Building project with Make..."
make -j$(nproc)

echo "Build completed successfully."
