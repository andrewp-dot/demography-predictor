
# CompareWithStatisticalModels

**Description:** Compares BaseLSTM with statistical models and BaseLSTM for single feature prediction.

## Per target metrics - model comparision
```
                             target           mae           mse          rmse           r2    state           model   rank
30                             year      0.000000  0.000000e+00      0.000000     1.000000  Czechia  ensemble-arima    4.0
41                 urban population      0.116707  2.451150e-02      0.156561     0.495473  Czechia  ensemble-arima   15.0
38                 rural population      0.116707  2.451185e-02      0.156563     0.495466  Czechia  ensemble-arima   19.0
15                             year      0.219697  5.527790e-02      0.235113     0.981048  Czechia   ensemble-lstm   24.0
34                birth rate, crude      0.147699  3.024905e-02      0.173923    -0.134339  Czechia  ensemble-arima   31.0
31            fertility rate, total      0.109889  1.488060e-02      0.121986    -3.117615  Czechia  ensemble-arima   32.0
16            fertility rate, total      0.136165  1.914156e-02      0.138353    -4.296665  Czechia   ensemble-lstm   39.0
22                agricultural land      0.185747  9.129621e-02      0.302153    -1.784003  Czechia   ensemble-lstm   47.0
24          rural population growth      0.403553  6.922743e-01      0.832030     0.016028  Czechia   ensemble-lstm   49.0
40             age dependency ratio      0.614078  7.877605e-01      0.887559     0.810074  Czechia  ensemble-arima   51.0
37                agricultural land      0.339272  1.294220e-01      0.359753    -2.946616  Czechia  ensemble-arima   52.0
42                population growth      0.447501  6.558341e-01      0.809836    -0.065825  Czechia  ensemble-arima   53.0
43        adolescent fertility rate      0.716490  6.236195e-01      0.789696     0.012721  Czechia  ensemble-arima   53.0
4                 birth rate, crude      0.387977  2.060678e-01      0.453947    -6.727542  Czechia       base-lstm   62.0
1             fertility rate, total      0.380264  1.454245e-01      0.381346   -39.240441  Czechia       base-lstm   62.0
39          rural population growth      0.429903  8.186902e-01      0.904815    -0.163655  Czechia  ensemble-arima   62.0
27                population growth      0.550726  9.182520e-01      0.958255    -0.492292  Czechia   ensemble-lstm   72.0
33                      arable land      0.631266  4.583280e-01      0.676999   -69.805065  Czechia  ensemble-arima   76.0
12                population growth      0.902707  8.908614e-01      0.943855    -0.447778  Czechia       base-lstm   78.0
9           rural population growth      0.635366  1.273059e+00      1.128299    -0.809477  Czechia       base-lstm   82.0
29  life expectancy at birth, total      0.760597  9.813485e-01      0.990630    -0.973313  Czechia   ensemble-lstm   83.0
0                              year      1.295683  1.693316e+00      1.301275     0.419434  Czechia       base-lstm   83.0
21                death rate, crude      0.750513  1.443954e+00      1.201646    -0.145995  Czechia   ensemble-lstm   83.0
44  life expectancy at birth, total      0.726080  1.355599e+00      1.164302    -1.725864  Czechia  ensemble-arima   89.0
14  life expectancy at birth, total      0.961884  1.078992e+00      1.038745    -1.169657  Czechia       base-lstm   90.0
6                 death rate, crude      0.853352  1.729895e+00      1.315255    -0.372933  Czechia       base-lstm   92.0
36                death rate, crude      0.831233  1.845434e+00      1.358468    -0.464630  Czechia  ensemble-arima   95.0
20                       gdp growth      2.422407  1.196866e+01      3.459576    -0.003169  Czechia   ensemble-lstm  110.0
35                       gdp growth      2.117953  1.263876e+01      3.555104    -0.059334  Czechia  ensemble-arima  112.0
11                 urban population      1.380231  2.004461e+00      1.415790   -40.258365  Czechia       base-lstm  117.0
5                        gdp growth      2.734431  1.276035e+01      3.572164    -0.069525  Czechia       base-lstm  119.0
19                birth rate, crude      1.787476  3.339228e+00      1.827356  -124.221059  Czechia   ensemble-lstm  124.0
8                  rural population      1.813117  3.732007e+00      1.931840   -75.816927  Czechia       base-lstm  126.0
7                 agricultural land      1.982672  4.157335e+00      2.038954  -125.774490  Czechia       base-lstm  132.0
23                 rural population      2.470850  6.143791e+00      2.478667  -125.459323  Czechia   ensemble-lstm  136.0
26                 urban population      3.098721  9.662458e+00      3.108449  -197.885002  Czechia   ensemble-lstm  142.0
25             age dependency ratio      8.705049  7.852112e+01      8.861214   -17.931172  Czechia   ensemble-lstm  150.0
28        adolescent fertility rate      6.438529  4.440719e+01      6.663872   -69.302950  Czechia   ensemble-lstm  151.0
10             age dependency ratio     11.060585  1.260782e+02     11.228453   -29.397008  Czechia       base-lstm  154.0
18                      arable land      3.853500  1.499625e+01      3.872499 -2315.703612  Czechia   ensemble-lstm  154.0
32                    net migration   2806.847773  1.031056e+07   3211.005680    -3.901827  Czechia  ensemble-arima  156.0
3                       arable land      4.146230  1.792785e+01      4.234129 -2768.593453  Czechia       base-lstm  158.0
13        adolescent fertility rate     27.955777  7.839363e+02     27.998863 -1240.083742  Czechia       base-lstm  168.0
2                     net migration  33235.046148  1.168052e+09  34176.773397  -554.313175  Czechia       base-lstm  173.0
17                    net migration  82193.669376  7.024254e+09  83810.820511 -3338.458449  Czechia   ensemble-lstm  180.0
```


## Overall metrics - model comparision
```
           mae           mse         rmse          r2    state           model  rank
2   187.612837  6.873718e+05   214.808083   -5.304736  Czechia  ensemble-arima   4.0
0  2219.435762  7.787019e+07  2282.383754 -325.510339  Czechia       base-lstm   8.0
1  5481.696860  4.682836e+08  5589.716688 -413.043995  Czechia   ensemble-lstm  12.0
```


