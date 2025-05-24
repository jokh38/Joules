# Add the test executable
add_executable(test_mqi_scorer test_mqi_scorer.cpp)

# Link the test executable against our library and Google Test
# moquimc_lib is an INTERFACE library, so its include directories and link libraries
# (like CUDA, GDCM, ZLIB) are transitively applied.
target_link_libraries(test_mqi_scorer PRIVATE moquimc_lib GTest::gtest_main)

# Discover and add tests to CTest
include(GoogleTest)
gtest_discover_tests(test_mqi_scorer)

message(STATUS "Added test executable: test_mqi_scorer")
