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
    src/Fem1D.cpp
    src/Thermoelastic1D.cpp
    src/Mesh2D.cpp
    src/Fem2D.cpp
    src/Triplets.cpp
    src/Assembly2D.cpp
    src/Dirichlet2D.cpp
    src/Solver2D.cpp
    src/Verification2D.cpp
    src/Elasticity2D.cpp
    src/Thermoelastic2D.cpp)
target_include_directories(twin_core PUBLIC include)
target_compile_options(twin_core PRIVATE -Wall -Wextra -Werror=return-type)
# Programma che stampa le tabelle di convergenza (per il README).
add_executable(convergence_study apps/convergence_study.cpp)
target_link_libraries(convergence_study PRIVATE twin_core)
target_compile_options(convergence_study PRIVATE -Wall -Wextra)

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
    tests/test_fem1d.cpp
    tests/test_thermoelastic1d.cpp
    tests/test_mesh2d.cpp
    tests/test_fem2d.cpp
    tests/test_triplets.cpp
    tests/test_assembly2d.cpp
    tests/test_dirichlet2d.cpp
    tests/test_solver2d.cpp
    tests/test_load2d.cpp
    tests/test_verification2d.cpp
    tests/test_transient2d.cpp
    tests/test_decay2d.cpp
    tests/test_elasticity2d.cpp
    tests/test_elasticity_element2d.cpp
    tests/test_elasticity_stiffness2d.cpp
    tests/test_elasticity_thermal_load2d.cpp
    tests/test_elasticity_assembly2d.cpp
    tests/test_elasticity_load2d.cpp
    tests/test_elasticity_solve2d.cpp
    tests/test_elasticity_stress2d.cpp
    tests/test_elasticity_beam2d.cpp
    tests/test_thermoelastic2d.cpp
    tests/test_von_mises2d.cpp)
target_link_libraries(unit_tests PRIVATE twin_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)

# 4) The Python module "thermomech" (pybind11). Switch it off with -DTWIN_BUILD_PYTHON=OFF.
option(TWIN_BUILD_PYTHON "Build the Python module and its tests" ON)
if(TWIN_BUILD_PYTHON)
    # On Windows, look for Python in PATH before the registry: it is the "python" you type in the terminal.
    set(Python_FIND_REGISTRY LAST)
    set(PYBIND11_FINDPYTHON ON)

    # pybind11 is downloaded the first time you configure, like GoogleTest.
    FetchContent_Declare(pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v3.1.0.zip)
    FetchContent_MakeAvailable(pybind11)

    # A shared module can only contain position-independent code (needed on Linux, ignored on Windows).
    set_target_properties(twin_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # The name of the target is the name of the module: "import thermomech".
    # NO_EXTRAS: no link-time optimization and no stripping, to keep the build simple.
    pybind11_add_module(thermomech NO_EXTRAS python/bindings.cpp)
    target_link_libraries(thermomech PRIVATE twin_core)
    set_target_properties(thermomech PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python)

    # MinGW: copy the compiler runtime into the module, otherwise Python cannot find libstdc++-6.dll and friends.
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(thermomech PRIVATE
            -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive)
    endif()

    # All the Python tests are one CTest test. PYTHONPATH tells Python where the module has been built.
    add_test(NAME python_tests
        COMMAND ${Python_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/python/tests -q)
    set_tests_properties(python_tests PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:thermomech>")
endif()
