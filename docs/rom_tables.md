### Benchmark problem

| item        | description                                                                                                                |
|:------------|:---------------------------------------------------------------------------------------------------------------------------|
| plate       | 200 mm x 100 mm, steel, plane strain; left edge clamped, the other edges free                                              |
| mesh        | 64 x 32 cells, 2145 nodes, 4096 triangles                                                                                  |
| temperature | rise above ambient; edges held at 0 K except the left one, 40 K * sin(pi y / H); the plate starts from the boundary values |
| heat source | Gaussian of width 25 mm centred at (140, 50) mm, peak value q                                                              |
| time        | backward Euler, dt = 3 s, 100 steps (300 s), 11 states kept                                                                |
| parameters  | k 36-54 W/(m K), rho_c 3.42-4.18 MJ/(m3 K), q 1.4-2.6 MW/m3, E 180-220 GPa, alpha 10.8-13.2 1e-6/K                         |
| training    | 16 heat runs of the C++ solver, (k, rho_c, q) from a Sobol sequence: 336 snapshots (21 per run)                            |
| test        | 8 other runs of the full C++ model, the five parameters from another Sobol sequence                                        |
| machine     | Windows-11-10.0.26200-SP0, Python 3.14.7, NumPy 2.5.4                                                                      |

### Offline cost, paid once (ROM with 12 modes)

| step                                            |  time | in full-model runs |
|:------------------------------------------------|------:|-------------------:|
| training: 16 heat runs (C++)                    | 9.3 s |               5.55 |
| POD: basis of 12 modes                          | 71 ms |               0.04 |
| ROM: projection and 13 elasticity solves (C++)  | 1.6 s |               0.96 |
| offline, total                                  |  11 s |               6.55 |
| one run of the full model (heat and elasticity) | 1.7 s |               1.00 |

### Accuracy and speed against the number of modes (worst of 8 test runs, relative L2 error)

| modes | sigma_r / sigma_1 |        T |        u |   stress | von Mises | ROM run | speed-up |
|------:|------------------:|---------:|---------:|---------:|----------:|--------:|---------:|
|     1 |          1.00e+00 | 2.58e-01 | 2.12e-01 | 1.80e-01 |  2.34e-01 | 4.13 ms |      410 |
|     2 |          1.30e-01 | 7.86e-02 | 6.08e-02 | 6.75e-02 |  6.45e-02 | 3.82 ms |      440 |
|     3 |          5.62e-02 | 1.94e-02 | 9.06e-03 | 2.02e-02 |  1.44e-02 | 4.34 ms |      390 |
|     4 |          1.91e-02 | 4.55e-03 | 1.27e-03 | 5.50e-03 |  3.80e-03 | 4.40 ms |      380 |
|     5 |          3.34e-03 | 1.39e-03 | 3.91e-04 | 1.57e-03 |  8.93e-04 | 4.13 ms |      410 |
|     6 |          1.03e-03 | 1.20e-03 | 2.60e-04 | 1.43e-03 |  8.39e-04 | 3.78 ms |      440 |
|     8 |          9.82e-05 | 1.94e-04 | 4.60e-05 | 2.13e-04 |  1.39e-04 | 3.86 ms |      430 |
|    10 |          1.40e-05 | 6.11e-05 | 9.61e-06 | 7.66e-05 |  3.88e-05 | 4.02 ms |      420 |
|    12 |          1.68e-06 | 3.10e-05 | 4.70e-06 | 3.87e-05 |  1.79e-05 | 4.58 ms |      370 |

### Cost of N evaluations (ROM with 12 modes, offline phase included)

|       N | C++ full model |     ROM | ratio |
|--------:|---------------:|--------:|------:|
|       1 |          1.7 s |    11 s |  0.15 |
|      10 |           17 s |    11 s |   1.5 |
|     100 |        2.8 min |    11 s |    15 |
|    1000 |         28 min |    16 s |   110 |
|   10000 |          4.6 h |    57 s |   290 |
|  100000 |       1.9 days | 7.8 min |   360 |
| 1000000 |        19 days |   1.3 h |   360 |

The ROM is cheaper than the full model from N = 7 evaluations on.
