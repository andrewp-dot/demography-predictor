
# FineTunedModels

**Description:** See if finetuning the model helps the model to be more accurate.

## Per target metrics - model comparision
```
                             target           mae           mse          rmse            r2    state                   model   rank
4                 birth rate, crude  3.524216e-01  1.534165e-01  3.916842e-01 -4.753118e+00  Czechia               base-lstm   17.0
31            fertility rate, total  2.011391e-01  4.117035e-02  2.029048e-01 -1.039226e+01  Czechia  group-states-finetuned   18.0
39          rural population growth  4.728296e-01  7.386692e-01  8.594586e-01 -4.991583e-02  Czechia  group-states-finetuned   19.0
12                population growth  7.139292e-01  7.382620e-01  8.592218e-01 -1.997823e-01  Czechia               base-lstm   24.0
14  life expectancy at birth, total  7.640634e-01  6.790136e-01  8.240228e-01 -3.653731e-01  Czechia               base-lstm   24.0
8                  rural population  5.710985e-01  4.693335e-01  6.850792e-01 -8.660418e+00  Czechia               base-lstm   27.0
6                 death rate, crude  1.030492e+00  1.344427e+00  1.159494e+00 -6.700549e-02  Czechia               base-lstm   31.0
9           rural population growth  1.038426e+00  1.123808e+00  1.060098e+00 -5.973380e-01  Czechia               base-lstm   33.0
1             fertility rate, total  5.050863e-01  2.559855e-01  5.059501e-01 -6.983382e+01  Czechia               base-lstm   34.0
30                             year  1.197108e+00  1.521017e+00  1.233295e+00  4.785086e-01  Czechia  group-states-finetuned   34.0
27                population growth  1.044200e+00  1.310483e+00  1.144763e+00 -1.129724e+00  Czechia  single-state-finetuned   37.0
0                              year  1.828730e+00  3.444079e+00  1.855823e+00 -1.808270e-01  Czechia               base-lstm   47.0
42                population growth  2.046524e+00  4.819497e+00  2.195335e+00 -6.832378e+00  Czechia  group-states-finetuned   57.0
5                        gdp growth  2.049742e+00  1.290748e+01  3.592698e+00 -8.185685e-02  Czechia               base-lstm   58.0
11                 urban population  1.315225e+00  1.753765e+00  1.324298e+00 -3.509822e+01  Czechia               base-lstm   59.0
41                 urban population  1.296500e+00  1.799507e+00  1.341457e+00 -3.603975e+01  Czechia  group-states-finetuned   61.0
44  life expectancy at birth, total  2.485692e+00  7.969043e+00  2.822949e+00 -1.502430e+01  Czechia  group-states-finetuned   68.0
7                 agricultural land  2.500488e+00  6.600135e+00  2.569073e+00 -2.002657e+02  Czechia               base-lstm   77.0
28        adolescent fertility rate  3.427165e+00  1.505352e+01  3.879886e+00 -2.283188e+01  Czechia  single-state-finetuned   80.0
3                       arable land  3.362448e+00  1.165000e+01  3.413209e+00 -1.798756e+03  Czechia               base-lstm   89.0
35                       gdp growth  5.552198e+00  4.136852e+01  6.431837e+00 -2.467355e+00  Czechia  group-states-finetuned   91.0
34                birth rate, crude  3.828930e+00  1.472724e+01  3.837608e+00 -5.512714e+02  Czechia  group-states-finetuned   91.0
24          rural population growth  4.718083e+00  2.358354e+01  4.856289e+00 -3.252074e+01  Czechia  single-state-finetuned   93.0
40             age dependency ratio  5.910759e+00  3.676077e+01  6.063066e+00 -7.862894e+00  Czechia  group-states-finetuned   93.0
38                 rural population  4.264426e+00  1.821571e+01  4.267987e+00 -3.739390e+02  Czechia  group-states-finetuned   94.0
25             age dependency ratio  5.231343e+00  5.339305e+01  7.307055e+00 -1.187288e+01  Czechia  single-state-finetuned   99.0
16            fertility rate, total  4.279585e+00  1.834188e+01  4.282743e+00 -5.074387e+03  Czechia  single-state-finetuned  107.0
29  life expectancy at birth, total  6.390550e+00  5.070345e+01  7.120636e+00 -1.009554e+02  Czechia  single-state-finetuned  110.0
33                      arable land  5.439392e+00  2.960378e+01  5.440935e+00 -4.572356e+03  Czechia  group-states-finetuned  113.0
15                             year  9.643289e+00  9.312084e+01  9.649914e+00 -3.092714e+01  Czechia  single-state-finetuned  113.0
10             age dependency ratio  1.008132e+01  1.055228e+02  1.027243e+01 -2.444119e+01  Czechia               base-lstm  115.0
21                death rate, crude  1.312015e+01  1.770110e+02  1.330455e+01 -1.394849e+02  Czechia  single-state-finetuned  125.0
37                agricultural land  9.142399e+00  8.508100e+01  9.223936e+00 -2.593475e+03  Czechia  group-states-finetuned  126.0
43        adolescent fertility rate  1.871403e+01  3.507082e+02  1.872720e+01 -5.542214e+02  Czechia  group-states-finetuned  133.0
36                death rate, crude  2.825526e+01  7.993452e+02  2.827269e+01 -6.334010e+02  Czechia  group-states-finetuned  140.0
13        adolescent fertility rate  3.066639e+01  9.440672e+02  3.072568e+01 -1.493594e+03  Czechia               base-lstm  144.0
18                      arable land  2.620674e+01  6.882152e+02  2.623386e+01 -1.063183e+05  Czechia  single-state-finetuned  148.0
19                birth rate, crude  3.166162e+01  1.002547e+03  3.166303e+01 -3.759453e+04  Czechia  single-state-finetuned  153.0
20                       gdp growth  6.983434e+01  4.897694e+03  6.998353e+01 -4.095064e+02  Czechia  single-state-finetuned  155.0
22                agricultural land  3.956950e+01  1.566685e+03  3.958137e+01 -4.777376e+04  Czechia  single-state-finetuned  157.0
26                 urban population  6.534473e+01  4.270981e+03  6.535274e+01 -8.790976e+04  Czechia  single-state-finetuned  161.0
2                     net migration  6.432976e+04  4.404080e+09  6.636324e+04 -2.092780e+03  Czechia               base-lstm  164.0
23                 rural population  6.685604e+01  4.472578e+03  6.687734e+01 -9.205930e+04  Czechia  single-state-finetuned  165.0
17                    net migration  1.054710e+06  1.432970e+12  1.197067e+06 -6.812591e+05  Czechia  single-state-finetuned  176.0
32                    net migration  1.921612e+06  3.705908e+12  1.925074e+06 -1.761856e+06  Czechia  group-states-finetuned  180.0
```


## Overall metrics - model comparision
```
             mae           mse           rmse             r2    state                   model  rank
0    4292.435692  2.936054e+08    4428.165357    -381.978323  Czechia               base-lstm   4.0
1   70337.140677  9.553133e+10   79827.898450  -70582.558634  Czechia  single-state-finetuned   8.0
2  128113.402905  2.470606e+11  128344.303073 -118080.827435  Czechia  group-states-finetuned  12.0
```


