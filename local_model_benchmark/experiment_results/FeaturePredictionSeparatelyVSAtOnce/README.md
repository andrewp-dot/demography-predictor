
# FeaturePredictionSeparatelyVSAtOnce

**Description:** Compares single LSTM model vs LSTM for every feature.

## Per target metrics - model comparision
```
                             target           mae           mse          rmse           r2    state           model   rank
15                             year      0.218014  5.389312e-02      0.232149     0.981522  Czechia  ensemble-model    7.0
16            fertility rate, total      0.117055  1.427269e-02      0.119468    -2.949399  Czechia  ensemble-model   17.0
22                agricultural land      0.288746  1.009383e-01      0.317708    -2.078029  Czechia  ensemble-model   22.0
24          rural population growth      0.405590  7.030404e-01      0.838475     0.000726  Czechia  ensemble-model   23.0
4                 birth rate, crude      0.352422  1.534165e-01      0.391684    -4.753118  Czechia       base-lstm   27.0
27                population growth      0.409404  7.253968e-01      0.851702    -0.178874  Czechia  ensemble-model   31.0
14  life expectancy at birth, total      0.764063  6.790136e-01      0.824023    -0.365373  Czechia       base-lstm   36.0
8                  rural population      0.571098  4.693335e-01      0.685079    -8.660418  Czechia       base-lstm   36.0
1             fertility rate, total      0.505086  2.559855e-01      0.505950   -69.833817  Czechia       base-lstm   38.0
12                population growth      0.713929  7.382620e-01      0.859222    -0.199782  Czechia       base-lstm   38.0
6                 death rate, crude      1.030492  1.344427e+00      1.159494    -0.067005  Czechia       base-lstm   43.0
21                death rate, crude      0.760172  1.363110e+00      1.167523    -0.081834  Czechia  ensemble-model   44.0
29  life expectancy at birth, total      0.741436  1.111819e+00      1.054428    -1.235666  Czechia  ensemble-model   44.0
9           rural population growth      1.038426  1.123808e+00      1.060098    -0.597338  Czechia       base-lstm   49.0
0                              year      1.828730  3.444079e+00      1.855823    -0.180827  Czechia       base-lstm   62.0
11                 urban population      1.315225  1.753765e+00      1.324298   -35.098216  Czechia       base-lstm   65.0
20                       gdp growth      2.332684  1.201854e+01      3.466777    -0.007349  Czechia  ensemble-model   71.0
19                birth rate, crude      1.785880  3.340180e+00      1.827616  -124.256761  Czechia  ensemble-model   73.0
18                      arable land      1.707554  2.933086e+00      1.712625  -452.119377  Czechia  ensemble-model   74.0
5                        gdp growth      2.049742  1.290748e+01      3.592698    -0.081857  Czechia       base-lstm   75.0
7                 agricultural land      2.500488  6.600135e+00      2.569073  -200.265689  Czechia       base-lstm   84.0
28        adolescent fertility rate      2.998487  1.015079e+01      3.186030   -15.070151  Czechia  ensemble-model   84.0
23                 rural population      2.566054  6.626962e+00      2.574289  -135.404570  Czechia  ensemble-model   85.0
26                 urban population      3.072643  9.495411e+00      3.081462  -194.446626  Czechia  ensemble-model   90.0
25             age dependency ratio      8.854680  8.154884e+01      9.030440   -18.661144  Czechia  ensemble-model   96.0
3                       arable land      3.362448  1.165000e+01      3.413209 -1798.756424  Czechia       base-lstm   99.0
10             age dependency ratio     10.081318  1.055228e+02     10.272430   -24.441187  Czechia       base-lstm  100.0
13        adolescent fertility rate     30.666391  9.440672e+02     30.725676 -1493.593810  Czechia       base-lstm  111.0
2                     net migration  64329.755522  4.404080e+09  66363.241602 -2092.779977  Czechia       base-lstm  116.0
17                    net migration  78427.942066  6.401054e+09  80006.585458 -3042.177827  Czechia  ensemble-model  120.0
```


## Overall metrics - model comparision
```
           mae           mse         rmse          r2    state           model  rank
0  4292.435692  2.936054e+08  4428.165357 -381.978323  Czechia       base-lstm   5.0
1  5230.280031  4.267369e+08  5335.736410 -265.845691  Czechia  ensemble-model   7.0
```


