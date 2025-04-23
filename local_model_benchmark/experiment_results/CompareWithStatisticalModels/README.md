
# CompareWithStatisticalModels

**Description:** Compares BaseRNN with statistical models and BaseRNN for single feature prediction.

## Per target metrics - model comparision
```
                     target           mae           mse          rmse             r2    state           model   rank
25         urban population  1.167077e-01  2.451168e-02  1.565621e-01       0.495469  Czechia  ensemble-arima   10.0
9     fertility rate, total  8.496299e-02  8.560262e-03  9.252168e-02      -1.368712  Czechia   ensemble-lstm   12.0
18    fertility rate, total  1.098889e-01  1.488060e-02  1.219861e-01      -3.117614  Czechia  ensemble-arima   20.0
0     fertility rate, total  1.886056e-01  3.781882e-02  1.944706e-01      -9.464855  Czechia       base-lstm   27.0
23        agricultural land  3.392725e-01  1.294221e-01  3.597529e-01      -2.946622  Czechia  ensemble-arima   28.0
16         urban population  4.047473e-01  1.834312e-01  4.282887e-01      -2.775615  Czechia   ensemble-lstm   30.0
26        population growth  4.475007e-01  6.558341e-01  8.098359e-01      -0.065825  Czechia  ensemble-arima   32.0
24  rural population growth  4.299030e-01  8.186902e-01  9.048150e-01      -0.163655  Czechia  ensemble-arima   36.0
14        agricultural land  5.713530e-01  3.907355e-01  6.250884e-01     -10.915158  Czechia   ensemble-lstm   40.0
8         population growth  8.328186e-01  1.483085e+00  1.217820e+00      -1.410227  Czechia       base-lstm   47.0
11              arable land  5.797912e-01  4.356674e-01  6.600510e-01     -66.304322  Czechia   ensemble-lstm   47.0
22        death rate, crude  8.312332e-01  1.845434e+00  1.358468e+00      -0.464630  Czechia  ensemble-arima   48.0
12               gdp growth  2.057724e+00  1.260390e+01  3.550197e+00      -0.056412  Czechia   ensemble-lstm   49.0
20              arable land  6.312667e-01  4.583281e-01  6.769994e-01     -69.805080  Czechia  ensemble-arima   51.0
6   rural population growth  1.346925e+00  1.844140e+00  1.357991e+00      -1.621189  Czechia       base-lstm   51.0
21               gdp growth  2.117953e+00  1.263876e+01  3.555104e+00      -0.059334  Czechia  ensemble-arima   53.0
3                gdp growth  2.234311e+00  1.305257e+01  3.612834e+00      -0.094018  Czechia       base-lstm   61.0
7          urban population  3.518873e+00  1.239748e+01  3.521005e+00    -254.180636  Czechia       base-lstm   72.0
13        death rate, crude  3.455223e+00  1.324044e+01  3.638742e+00      -9.508286  Czechia   ensemble-lstm   74.0
19        population, total  5.257250e+04  4.474421e+09  6.689111e+04      -0.076005  Czechia  ensemble-arima   80.0
5         agricultural land  3.579638e+00  1.284301e+01  3.583715e+00    -390.637040  Czechia       base-lstm   80.0
17        population growth  3.625890e+00  1.382113e+01  3.717679e+00     -21.461337  Czechia   ensemble-lstm   82.0
15  rural population growth  4.260827e+00  1.893163e+01  4.351050e+00     -25.908692  Czechia   ensemble-lstm   86.0
4         death rate, crude  4.362346e+00  2.027975e+01  4.503305e+00     -15.095044  Czechia       base-lstm   87.0
2               arable land  2.063630e+01  4.258588e+02  2.063635e+01  -65788.034901  Czechia       base-lstm   98.0
10        population, total  9.446225e+06  9.151886e+13  9.566549e+06  -22007.372263  Czechia   ensemble-lstm  103.0
1         population, total  2.785734e+07  7.777316e+14  2.788784e+07 -187027.183074  Czechia       base-lstm  108.0
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2    state           model  rank
2  5.841947e+03  4.971579e+08  7.433228e+03     -8.467033  Czechia  ensemble-arima   4.0
1  1.049582e+06  1.016876e+13  1.062952e+06  -2460.630088  Czechia   ensemble-lstm   8.0
0  3.095264e+06  8.641462e+13  3.098653e+06 -28165.302331  Czechia       base-lstm  12.0
```


