cmake_minimum_required(VERSION 3.20)
project(thermomech_twin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Your library: every .cpp except the tests.
#    A missing "return" becomes a compile error instead of a silent bug.
add_library(twin_core
    src/CSR_and_Vectors.cpp
    src/JacobiPreconditioner.cpp
    src/CGSolver.cpp
    src/Fem1D.cpp)
target_include_directories(twin_core PUBLIC include)
target_compile_options(twin_core PRIVATE -Wall -Wextra -Werror=return-type)

# 2) GoogleTest: CMake downloads it for you the first time you configure.
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)

# 3) The test program, linked against your library.
#    Every new test file goes in this list.
enable_testing()
add_executable(unit_tests
    tests/test_csr.cpp
    tests/test_cg.cpp
    tests/test_fem1d.cpp)
target_link_libraries(unit_tests PRIVATE twin_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)
