# thermomech-twin

A thermo-mechanical *digital twin* of a heated component: a C++ finite-element core, with a Python reduced-order model and uncertainty quantification planned on top. The goal is a small simulation stack in which every layer is checked against hand calculations, exact solutions and measured convergence orders before the next layer is built.

**Status: work in progress.** The 1D heat-conduction solver is complete and verified (see below). 2D/3D and the Python layer are next.

## What works today

- Sparse matrices in CSR format, Conjugate Gradient (CG) and Jacobi-preconditioned CG.
- 1D linear (P1) finite elements for `rho_c dT/dt - k d2T/dx2 = q`: stiffness, consistent mass and load assembly.
- Dirichlet conditions by symmetric elimination, so the system matrix stays symmetric and CG applies.
- Steady-state solver, and time stepping with the theta scheme (`theta = 1`: backward Euler, `theta = 0.5`: Crank-Nicolson):

  `(M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt F`
- Failures are reported, not hidden: `CGSolver` exposes `has_converged()`, and the steady and transient solvers throw `std::runtime_error` when the linear solver does not converge (for example with NaN in the data).
- 47 unit tests (GoogleTest) and a program that prints the convergence tables below.

## Build and test

Requirements: CMake 3.20 or newer and a C++17 compiler. Developed with GCC 14 (MinGW-w64, Windows); it also builds and passes all tests with GCC 13 on Linux. GoogleTest is downloaded by CMake the first time you configure, so an internet connection is needed once.

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

`-G Ninja` is optional; any CMake generator works. To reproduce the tables below, run `./build/convergence_study` (`.\build\convergence_study.exe` on Windows).

## Verification

How do we know the solver is right? Three kinds of evidence, from simple to demanding.

1. **Hand calculations.** Tiny meshes whose results can be written on paper: stiffness, load, mass, Dirichlet elimination, the L2 error and the observed order.
2. **Exact solutions.**
   - Steady state with constant `k` and `q`: `T(x) = T_left + (T_right - T_left) x / L + q x (L - x) / (2k)`. For this problem linear elements give the exact nodal values on any mesh, including graded ones, so the tests use a tolerance of 1e-9.
   - Time stepping: on a uniform mesh, `sin(pi x / L)` is an eigenvector of both `K` and `M`, so the theta scheme multiplies it by a known factor `g` at every step. This gives the exact solution of the *discrete scheme* (not of the PDE), and the tests compare the solver with `g^n sin(pi x / L)` to within 1e-9, for three values of `theta`.
   - Properties: the stationary solution is a fixed point of the scheme, any initial condition relaxes to it, the mass matrix sums to the total heat capacity.
3. **Convergence orders** (below).

### Convergence study

Test problem: a bar of length 1, `k = 1`, `rho_c = 1`, no source, both ends held at 0, initial temperature `sin(pi x)`. The exact solution is `T(x, t) = exp(-pi^2 t) sin(pi x)`.

- **Error:** the L2 norm of the difference between the numerical and the exact nodal values at `t = 0.1`. Because both fields are piecewise linear, it is computed as `sqrt(e^T M e)`, with `M` the mass matrix for `rho_c = 1`.
- **Observed order:** if the error behaves like `C h^p`, then `p = ln(e_coarse / e_fine) / ln(ratio)`, where `ratio` is the refinement factor (here always 2).
- **Separating the two error sources:** the total error is a mesh error plus a time error. The spatial study uses a tiny time step so that the time error is negligible. The temporal studies use a fine mesh (`N = 800`) so that the mesh error (about 3.5e-7) is far below the time error.

### Spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)

|     N |        h | L2 error | order |
|------:|---------:|---------:|------:|
|    10 | 1.00e-01 | 2.12e-03 |     - |
|    20 | 5.00e-02 | 5.34e-04 |  1.99 |
|    40 | 2.50e-02 | 1.34e-04 |  2.00 |
|    80 | 1.25e-02 | 3.34e-05 |  2.00 |
|   160 | 6.25e-03 | 8.38e-06 |  2.00 |

### Temporal convergence (backward Euler, N = 800, t = 0.1)

| steps |       dt | L2 error | order |
|------:|---------:|---------:|------:|
|     5 | 2.00e-02 | 2.37e-02 |     - |
|    10 | 1.00e-02 | 1.23e-02 |  0.94 |
|    20 | 5.00e-03 | 6.29e-03 |  0.97 |
|    40 | 2.50e-03 | 3.18e-03 |  0.99 |

### Temporal convergence (Crank-Nicolson, N = 800, t = 0.1)

| steps |       dt | L2 error | order |
|------:|---------:|---------:|------:|
|     5 | 2.00e-02 | 8.49e-04 |     - |
|    10 | 1.00e-02 | 2.12e-04 |  2.00 |
|    20 | 5.00e-03 | 5.31e-05 |  1.99 |
|    40 | 2.50e-03 | 1.35e-05 |  1.97 |

**Reading the tables.**
- Linear elements are second order in space: halving `h` divides the error by 4.
- Backward Euler is first order in time. The observed order approaches 1 from below as `dt` decreases (0.94, 0.97, 0.99): an order is a statement about the limit `dt -> 0`.
- Crank-Nicolson is second order in time. The small drop at the finest `dt` (1.97) is the mesh error starting to show next to a very small time error, not a loss of accuracy of the scheme: on a finer mesh (`N = 1600`) the three orders are 2.00, 2.00 and 1.99.

## Project layout

```
include/   CSR_and_Vectors.h, CGSolver.h, JacobiPreconditioner.h, Fem1D.h
src/       implementations
tests/     GoogleTest unit tests: test_csr.cpp, test_cg.cpp, test_fem1d.cpp
apps/      convergence_study.cpp  (prints the tables above)
```

## Known limitations

- Matrices are assembled as dense tables, and the system matrix of the time loop is rebuilt and converted to CSR at every step even though it does not change. This is fine in 1D (a few seconds for the whole study) and will be replaced by sparse assembly in 2D.
- There is no stability check for `theta < 0.5`. With a time step above the stability limit the temperatures grow at every step. The solver throws only when the growth produces NaN or infinity; otherwise it returns huge but finite numbers. The schemes used in the convergence study (`theta = 1` and `theta = 0.5`) are unconditionally stable.

## Roadmap

Planned, in this order:

1. 2D triangular elements with sparse assembly.
2. Thermo-mechanical coupling (thermal stresses).
3. Python interface and a proper-orthogonal-decomposition (POD) reduced-order model.
4. Sobol sensitivity analysis on the reduced model.

## Author

Vsevolod Ozmitel Volpato, (for my contacts see my main github page)

note : la matrice del ciclo temporale viene ricostruita a ogni passo scrivilo
