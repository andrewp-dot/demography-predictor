
# FineTunedModels

**Description:** See if finetuning the model helps the model to be more accurate.

## Per target metrics - model comparision
```
                             target           mae           mse          rmse            r2    state                   model   rank
31            fertility rate, total  1.107800e-01  1.286622e-02  1.134294e-01 -2.560216e+00  Czechia  group-states-finetuned   13.0
7                 agricultural land  2.055820e-01  7.628507e-02  2.761975e-01 -1.326250e+00  Czechia               base-lstm   14.0
34                birth rate, crude  4.621003e-01  2.487916e-01  4.987901e-01 -8.329684e+00  Czechia  group-states-finetuned   24.0
12                population growth  6.583757e-01  6.939387e-01  8.330298e-01 -1.277504e-01  Czechia               base-lstm   30.0
4                 birth rate, crude  4.931853e-01  3.384846e-01  5.817943e-01 -1.169317e+01  Czechia               base-lstm   32.0
9           rural population growth  5.571689e-01  8.415187e-01  9.173433e-01 -1.961022e-01  Czechia               base-lstm   32.0
38                 rural population  5.885501e-01  3.470806e-01  5.891355e-01 -6.144055e+00  Czechia  group-states-finetuned   33.0
6                 death rate, crude  9.402090e-01  1.381701e+00  1.175458e+00 -9.658819e-02  Czechia               base-lstm   36.0
8                  rural population  5.725187e-01  4.919633e-01  7.014009e-01 -9.126215e+00  Czechia               base-lstm   37.0
1             fertility rate, total  5.193876e-01  2.716440e-01  5.211948e-01 -7.416666e+01  Czechia               base-lstm   40.0
40             age dependency ratio  1.223419e+00  1.924603e+00  1.387301e+00  5.359849e-01  Czechia  group-states-finetuned   43.0
11                 urban population  8.902539e-01  8.245876e-01  9.080681e-01 -1.597271e+01  Czechia               base-lstm   47.0
14  life expectancy at birth, total  1.185875e+00  1.654575e+00  1.286303e+00 -2.327051e+00  Czechia               base-lstm   47.0
0                              year  1.785183e+00  3.244129e+00  1.801147e+00 -1.122728e-01  Czechia               base-lstm   50.0
24          rural population growth  1.659458e+00  3.851502e+00  1.962524e+00 -4.474377e+00  Czechia  single-state-finetuned   58.0
41                 urban population  1.195280e+00  1.512445e+00  1.229815e+00 -3.013106e+01  Czechia  group-states-finetuned   59.0
42                population growth  1.963683e+00  4.351401e+00  2.086001e+00 -6.071654e+00  Czechia  group-states-finetuned   63.0
5                        gdp growth  2.541093e+00  1.236437e+01  3.516301e+00 -3.633545e-02  Czechia               base-lstm   65.0
30                             year  2.540828e+00  6.480972e+00  2.545775e+00 -1.222048e+00  Czechia  group-states-finetuned   67.0
39          rural population growth  2.132565e+00  5.327304e+00  2.308095e+00 -6.572025e+00  Czechia  group-states-finetuned   68.0
33                      arable land  2.289880e+00  5.586077e+00  2.363488e+00 -8.619682e+02  Czechia  group-states-finetuned   90.0
27                population growth  3.727106e+00  1.402796e+01  3.745392e+00 -2.179746e+01  Czechia  single-state-finetuned   92.0
28        adolescent fertility rate  4.590522e+00  2.288228e+01  4.783542e+00 -3.522593e+01  Czechia  single-state-finetuned   98.0
35                       gdp growth  1.086188e+01  1.292935e+02  1.137073e+01 -9.836901e+00  Czechia  group-states-finetuned  101.0
44  life expectancy at birth, total  4.596185e+00  2.331759e+01  4.828829e+00 -4.588743e+01  Czechia  group-states-finetuned  102.0
3                       arable land  3.497197e+00  1.250094e+01  3.535667e+00 -1.930215e+03  Czechia               base-lstm  102.0
10             age dependency ratio  1.038195e+01  1.117582e+02  1.057158e+01 -2.594453e+01  Czechia               base-lstm  102.0
16            fertility rate, total  3.589461e+00  1.289929e+01  3.591558e+00 -3.568365e+03  Czechia  single-state-finetuned  106.0
25             age dependency ratio  1.435011e+01  2.282395e+02  1.510760e+01 -5.402776e+01  Czechia  single-state-finetuned  117.0
29  life expectancy at birth, total  1.336971e+01  1.860861e+02  1.364134e+01 -3.731853e+02  Czechia  single-state-finetuned  117.0
15                             year  1.477821e+01  2.187126e+02  1.478894e+01 -7.398718e+01  Czechia  single-state-finetuned  117.0
21                death rate, crude  1.678275e+01  2.853354e+02  1.689187e+01 -2.254566e+02  Czechia  single-state-finetuned  128.0
37                agricultural land  1.511021e+01  2.358836e+02  1.535850e+01 -7.192077e+03  Czechia  group-states-finetuned  134.0
13        adolescent fertility rate  2.673087e+01  7.172990e+02  2.678244e+01 -1.134587e+03  Czechia               base-lstm  139.0
36                death rate, crude  2.983569e+01  8.905685e+02  2.984239e+01 -7.058004e+02  Czechia  group-states-finetuned  142.0
43        adolescent fertility rate  2.709303e+01  7.349852e+02  2.711061e+01 -1.162587e+03  Czechia  group-states-finetuned  143.0
18                      arable land  2.014990e+01  4.077210e+02  2.019210e+01 -6.298601e+04  Czechia  single-state-finetuned  143.0
19                birth rate, crude  3.125165e+01  9.770244e+02  3.125739e+01 -3.663741e+04  Czechia  single-state-finetuned  153.0
2                     net migration  1.987699e+04  3.980488e+08  1.995116e+04 -1.882397e+02  Czechia               base-lstm  157.0
22                agricultural land  3.756686e+01  1.412009e+03  3.757670e+01 -4.305704e+04  Czechia  single-state-finetuned  157.0
20                       gdp growth  9.960781e+01  9.946817e+03  9.973373e+01 -8.327050e+02  Czechia  single-state-finetuned  158.0
26                 urban population  5.970845e+01  3.567514e+03  5.972867e+01 -7.343011e+04  Czechia  single-state-finetuned  162.0
23                 rural population  6.633277e+01  4.401996e+03  6.634754e+01 -9.060648e+04  Czechia  single-state-finetuned  166.0
32                    net migration  2.446491e+06  6.022703e+12  2.454120e+06 -2.863302e+06  Czechia  group-states-finetuned  176.0
17                    net migration  2.859006e+06  8.451165e+12  2.907089e+06 -4.017837e+06  Czechia  single-state-finetuned  180.0
```


## Overall metrics - model comparision
```
             mae           mse           rmse             r2    state                   model  rank
0    1328.529778  2.653664e+07    1333.637894    -226.277843  Czechia               base-lstm   4.0
2  163106.070496  4.015135e+11  163614.746092 -191556.036514  Czechia  group-states-finetuned   8.0
1  190626.199341  5.634110e+11  193831.875628 -288649.569263  Czechia  single-state-finetuned  12.0
```


