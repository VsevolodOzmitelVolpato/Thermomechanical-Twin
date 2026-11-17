### Uncertainty study

| item              | description                                                                                                                                                                                                              |
|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model             | the ROM of the benchmark: 12 modes, trained on 16 heat runs of the C++ solver; mesh 64 x 32 cells, 2145 nodes, 4096 triangles                                                                                            |
| uncertain numbers | k, rho_c, q, E, alpha: independent, each uniform on its range (next table)                                                                                                                                               |
| results           | read at t = 300 s: the peak temperature at the source (a rise above the ambient one, over the nodes within 25 mm of its centre), the peak von Mises stress (all the triangles) and the peak displacement (all the nodes) |
| indices           | S (first order) and ST (total) of every number and every result; estimators of Saltelli and of Jansen on a scrambled Sobol sequence, seed 0; 95 % intervals from 200 bootstrap resamples                                 |
| size              | n = 4096: n (d + 2) = 28672 runs of the model; the convergence is followed from n = 128                                                                                                                                  |
| study             | 28672 distinct runs of the ROM (17.92 ms each, the three results at the last state only): 8.6 min; the whole study, with the indices and their intervals: 8.6 min                                                        |
| machine           | Windows-11-10.0.26200-SP0, Python 3.14.7, NumPy 2.5.4, SciPy 1.18.1                                                                                                                                                      |

### Uncertain numbers (independent, uniform on the range)

| number | unit      | nominal |       range | spread |
|:-------|:----------|--------:|------------:|-------:|
| k      | W/(m K)   |      45 |     36 - 54 | +-20 % |
| rho_c  | MJ/(m3 K) |     3.8 | 3.42 - 4.18 | +-10 % |
| q      | MW/m3     |       2 |   1.4 - 2.6 | +-30 % |
| E      | GPa       |     200 |   180 - 220 | +-10 % |
| alpha  | 1e-6/K    |      12 | 10.8 - 13.2 | +-10 % |

### Uncertainty of the results (n = 4096: mean and standard deviation; the 95 % range of the 28672 runs of the study)

| result                         | unit |  mean |  std | std / mean | 2.5 % | 97.5 % |
|:-------------------------------|-----:|------:|-----:|-----------:|------:|-------:|
| peak temperature at the source |    K | 23.96 | 4.86 |     20.3 % | 15.81 |  34.04 |
| peak von Mises stress          |  MPa | 115.9 | 9.47 |      8.2 % | 98.56 |  134.6 |
| peak displacement              |   um | 35.78 | 4.87 |     13.6 % | 27.47 |  46.04 |

### Sobol indices of the peak temperature at the source (n = 4096)

| number |     S |   95 % interval |    ST |  95 % interval |
|:-------|------:|----------------:|------:|---------------:|
| k      | 0.291 |  [0.271, 0.312] | 0.300 | [0.288, 0.312] |
| rho_c  | 0.000 | [-0.001, 0.001] | 0.000 | [0.000, 0.000] |
| q      | 0.700 |  [0.665, 0.736] | 0.709 | [0.683, 0.737] |
| E      | 0.000 |  [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| alpha  | 0.000 |  [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| sum    | 0.991 |                 | 1.009 |                |

### Sobol indices of the peak von Mises stress (n = 4096)

| number |     S |  95 % interval |    ST |  95 % interval |
|:-------|------:|---------------:|------:|---------------:|
| k      | 0.000 | [0.000, 0.001] | 0.000 | [0.000, 0.000] |
| rho_c  | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| q      | 0.000 | [0.000, 0.001] | 0.000 | [0.000, 0.000] |
| E      | 0.499 | [0.475, 0.525] | 0.501 | [0.481, 0.524] |
| alpha  | 0.499 | [0.473, 0.528] | 0.501 | [0.480, 0.521] |
| sum    | 0.998 |                | 1.002 |                |

### Sobol indices of the peak displacement (n = 4096)

| number |     S |   95 % interval |    ST |  95 % interval |
|:-------|------:|----------------:|------:|---------------:|
| k      | 0.226 |  [0.206, 0.244] | 0.234 | [0.224, 0.246] |
| rho_c  | 0.000 | [-0.001, 0.001] | 0.001 | [0.000, 0.001] |
| q      | 0.584 |  [0.555, 0.617] | 0.593 | [0.570, 0.616] |
| E      | 0.000 |  [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| alpha  | 0.180 |  [0.162, 0.198] | 0.182 | [0.176, 0.191] |
| sum    | 0.990 |                 | 1.010 |                |

### Convergence with n: the largest half-width of the 95 % intervals of the indices

|    n |  runs | peak temperature at the source | peak von Mises stress | peak displacement |
|-----:|------:|-------------------------------:|----------------------:|------------------:|
|  128 |   896 |                         0.1567 |                0.1598 |            0.1625 |
|  256 |  1792 |                         0.1256 |                0.1150 |            0.1123 |
|  512 |  3584 |                         0.0883 |                0.0817 |            0.0828 |
| 1024 |  7168 |                         0.0553 |                0.0565 |            0.0503 |
| 2048 | 14336 |                         0.0450 |                0.0395 |            0.0412 |
| 4096 | 28672 |                         0.0358 |                0.0273 |            0.0310 |

### Other seeds, n = 256: the largest difference of an index between the 4 seeds (0, 1, 2, 3), against the half-width of the intervals of seed 0

| result                         | half-width | difference of S | difference of ST |
|:-------------------------------|-----------:|----------------:|-----------------:|
| peak temperature at the source |     0.1256 |          0.0280 |           0.0013 |
| peak von Mises stress          |     0.1150 |          0.0030 |           0.0011 |
| peak displacement              |     0.1123 |          0.0243 |           0.0081 |

### ROM against the C++ full model, relative error of the three results (worst of 8 test runs, worst of the 8 corners of the box)

| result                         | test runs | corners |
|:-------------------------------|----------:|--------:|
| peak temperature at the source |   3.8e-07 | 5.5e-07 |
| peak von Mises stress          |   6.8e-09 | 8.4e-09 |
| peak displacement              |   4.8e-08 | 5.3e-08 |

### Structure of the indices: what the ROM makes certain (n = 4096)

| check                                                              | expected | estimated |  95 % interval | verdict |
|:-------------------------------------------------------------------|---------:|----------:|---------------:|:--------|
| peak temperature at the source: S and ST of E and alpha            |        0 |   0.0e+00 |                | ok      |
| peak displacement: S and ST of E                                   |        0 |   0.0e+00 |                | ok      |
| peak displacement: S of alpha, from the mean and the variance      |    0.180 |     0.180 | [0.162, 0.198] | inside  |
| peak displacement: ST of alpha, from the mean and the variance     |    0.182 |     0.182 | [0.176, 0.191] | inside  |
| peak von Mises stress: S of E, from the mean and the variance      |    0.499 |     0.499 | [0.475, 0.525] | inside  |
| peak von Mises stress: ST of E, from the mean and the variance     |    0.501 |     0.501 | [0.481, 0.524] | inside  |
| peak von Mises stress: S of alpha, from the mean and the variance  |    0.499 |     0.499 | [0.473, 0.528] | inside  |
| peak von Mises stress: ST of alpha, from the mean and the variance |    0.501 |     0.501 | [0.480, 0.521] | inside  |

### Cost of the study (ROM: offline phase included; full model: estimated from its runs)

|    n |  runs |     ROM | C++ full model | ratio |
|-----:|------:|--------:|---------------:|------:|
|  128 |   896 | 2.4 min |          1.8 h |    44 |
|  256 |  1792 | 2.7 min |          3.5 h |    78 |
|  512 |  3584 | 3.2 min |          7.0 h |   130 |
| 1024 |  7168 | 4.3 min |           14 h |   200 |
| 2048 | 14336 | 6.4 min |       1.2 days |   260 |
| 4096 | 28672 |  11 min |       2.3 days |   310 |

One run of the ROM takes 17.92 ms and one run of the full model 7.0 s, both with the last state only; the offline phase of the ROM takes 2.1 min.
