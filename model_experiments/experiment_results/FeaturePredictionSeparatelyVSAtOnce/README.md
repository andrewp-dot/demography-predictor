
# FeaturePredictionSeparatelyVSAtOnce

**Description:** Compares single LSTM model vs LSTM for every feature.

## Per target metrics - model comparision
```
                             target          mae           mse         rmse           r2    state           model   rank
16            fertility rate, total     0.039323  1.918905e-03     0.043805     0.469019  Czechia  ensemble-model    5.0
15                             year     0.833333  8.333333e-01     0.912871     0.714286  Czechia  ensemble-model   19.0
8                  rural population     0.473969  2.995913e-01     0.547349    -5.166570  Czechia       base-lstm   23.0
12                population growth     0.472807  9.727048e-01     0.986258    -0.580786  Czechia       base-lstm   23.0
9           rural population growth     0.778823  8.642945e-01     0.929674    -0.228475  Czechia       base-lstm   23.0
1             fertility rate, total     0.515987  2.676147e-01     0.517315   -73.051721  Czechia       base-lstm   32.0
22                agricultural land     0.820128  7.269955e-01     0.852640   -21.169127  Czechia  ensemble-model   32.0
6                 death rate, crude     1.026750  1.464526e+00     1.210176    -0.162322  Czechia       base-lstm   32.0
14  life expectancy at birth, total     0.807415  1.548079e+00     1.244218    -2.112905  Czechia       base-lstm   35.0
26                 urban population     1.035881  1.082161e+00     1.040270   -21.274411  Czechia  ensemble-model   44.0
24          rural population growth     1.469632  2.879571e+00     1.696930    -3.092911  Czechia  ensemble-model   47.0
4                 birth rate, crude     1.520725  2.383760e+00     1.543943   -88.391009  Czechia       base-lstm   59.0
20                       gdp growth     1.991985  1.366463e+01     3.696570    -0.145318  Czechia  ensemble-model   62.0
11                 urban population     1.847461  3.418898e+00     1.849026   -69.372116  Czechia       base-lstm   63.0
27                population growth     1.948550  4.515049e+00     2.124865    -6.337606  Czechia  ensemble-model   63.0
19                birth rate, crude     1.555051  2.468361e+00     1.571102   -91.563554  Czechia  ensemble-model   63.0
21                death rate, crude     1.980059  5.389359e+00     2.321499    -3.277269  Czechia  ensemble-model   64.0
23                 rural population     1.872887  3.513142e+00     1.874338   -71.311952  Czechia  ensemble-model   67.0
29  life expectancy at birth, total     2.016016  4.614830e+00     2.148216    -8.279585  Czechia  ensemble-model   69.0
0                              year     4.166667  1.750000e+01     4.183300    -5.000000  Czechia       base-lstm   76.0
5                        gdp growth     4.566734  2.153631e+01     4.640723    -0.805093  Czechia       base-lstm   77.0
18                      arable land     2.497209  6.257039e+00     2.501407  -965.622203  Czechia  ensemble-model   87.0
2                     net migration  1050.537459  2.208388e+06  1486.064758    -0.049908  Czechia       base-lstm   90.0
3                       arable land     4.484155  2.105609e+01     4.588692 -3251.862527  Czechia       base-lstm   96.0
25             age dependency ratio    12.242948  1.524332e+02    12.346385   -35.751125  Czechia  ensemble-model   97.0
28        adolescent fertility rate     6.664784  4.620003e+01     6.797060   -72.141276  Czechia  ensemble-model   98.0
7                 agricultural land     5.315587  2.915164e+01     5.399226  -887.955298  Czechia       base-lstm  100.0
10             age dependency ratio    14.228277  2.077772e+02    14.414477   -49.094357  Czechia       base-lstm  101.0
17                    net migration  2917.196536  9.654211e+06  3107.122595    -3.589788  Czechia  ensemble-model  102.0
13        adolescent fertility rate    20.865250  4.366367e+02    20.895853  -690.258496  Czechia       base-lstm  111.0
```


## Overall metrics - model comparision
```
          mae            mse        rmse          r2    state           model  rank
0   74.107204  147275.556161  103.267666 -341.606106  Czechia       base-lstm   5.0
1  196.944288  643630.360151  209.803370  -86.824855  Czechia  ensemble-model   7.0
```


