
# FineTunedModels

**Description:** See if finetuning the model helps the model to be more accurate.

## Per target metrics - model comparision
```
                             target           mae           mse          rmse            r2    state                   model   rank
19                birth rate, crude      0.141324  2.434534e-02      0.156030      0.087050  Czechia  single-state-finetuned    4.0
42                population growth      0.479494  6.393077e-01      0.799567     -0.038967  Czechia  group-states-finetuned   21.0
1             fertility rate, total      0.206955  4.403437e-02      0.209844    -11.184759  Czechia               base-lstm   22.0
39          rural population growth      0.738018  7.333475e-01      0.856357     -0.042352  Czechia  group-states-finetuned   26.0
34                birth rate, crude      0.442940  2.257168e-01      0.475097     -7.464379  Czechia  group-states-finetuned   26.0
31            fertility rate, total      0.264082  7.320734e-02      0.270569    -19.257219  Czechia  group-states-finetuned   28.0
8                  rural population      0.608035  5.268511e-01      0.725845     -9.844319  Czechia               base-lstm   31.0
9           rural population growth      0.821538  1.240507e+00      1.113780     -0.763208  Czechia               base-lstm   32.0
44  life expectancy at birth, total      1.065380  1.632406e+00      1.277657     -2.282472  Czechia  group-states-finetuned   40.0
21                death rate, crude      1.467952  2.426922e+00      1.557858     -0.926129  Czechia  single-state-finetuned   41.0
10             age dependency ratio      1.651260  3.882181e+00      1.970325      0.064019  Czechia               base-lstm   44.0
36                death rate, crude      1.587666  3.778941e+00      1.943950     -1.999160  Czechia  group-states-finetuned   47.0
35                       gdp growth      2.271948  1.204693e+01      3.470868     -0.009728  Czechia  group-states-finetuned   50.0
12                population growth      1.612781  3.060672e+00      1.749478     -3.974033  Czechia               base-lstm   50.0
16            fertility rate, total      0.894000  8.035222e-01      0.896394   -221.342794  Czechia  single-state-finetuned   52.0
5                        gdp growth      3.338085  2.181409e+01      4.670556     -0.828376  Czechia               base-lstm   62.0
14  life expectancy at birth, total      3.476895  1.219879e+01      3.492677    -23.529559  Czechia               base-lstm   73.0
7                 agricultural land      3.289917  1.113108e+01      3.336327   -338.433116  Czechia               base-lstm   76.0
6                 death rate, crude      4.643092  2.281278e+01      4.776272    -17.105378  Czechia               base-lstm   78.0
11                 urban population      3.346036  1.244130e+01      3.527222   -255.082597  Czechia               base-lstm   82.0
24          rural population growth      4.811921  2.354464e+01      4.852282    -32.465440  Czechia  single-state-finetuned   84.0
40             age dependency ratio      7.096882  5.446780e+01      7.380231    -12.132000  Czechia  group-states-finetuned   89.0
27                population growth      5.946413  3.575595e+01      5.979628    -57.108572  Czechia  single-state-finetuned   92.0
4                 birth rate, crude      5.760947  3.327109e+01      5.768110  -1246.665928  Czechia               base-lstm   99.0
37                agricultural land      7.445224  5.547229e+01      7.447972  -1690.581842  Czechia  group-states-finetuned  110.0
38                 rural population      7.566981  5.731155e+01      7.570439  -1178.659247  Czechia  group-states-finetuned  113.0
30                             year     12.500000  1.591667e+02     12.616127    -53.571429  Czechia  group-states-finetuned  115.0
33                      arable land      7.561631  5.718323e+01      7.561960  -8832.983985  Czechia  group-states-finetuned  119.0
25             age dependency ratio     17.828493  3.236953e+02     17.991533    -77.041812  Czechia  single-state-finetuned  120.0
41                 urban population      9.504095  9.037932e+01      9.506804  -1859.302205  Czechia  group-states-finetuned  123.0
18                      arable land      7.894408  6.248155e+01      7.904527  -9651.498189  Czechia  single-state-finetuned  126.0
23                 rural population     11.117844  1.236633e+02     11.120400  -2544.395419  Czechia  single-state-finetuned  127.0
0                              year     24.833333  6.211667e+02     24.923215   -211.971429  Czechia               base-lstm  131.0
13        adolescent fertility rate     22.614104  5.126802e+02     22.642443   -810.646398  Czechia               base-lstm  133.0
15                             year     27.833333  7.791667e+02     27.913557   -266.142857  Czechia  single-state-finetuned  137.0
2                     net migration   2930.732877  1.012059e+07   3181.286715     -3.811511  Czechia               base-lstm  140.0
17                    net migration   2935.276420  1.015745e+07   3187.075995     -3.829039  Czechia  single-state-finetuned  144.0
3                       arable land     18.646351  3.476935e+02     18.646541 -53712.628055  Czechia               base-lstm  144.0
43        adolescent fertility rate     31.014146  9.625308e+02     31.024681  -1522.824472  Czechia  group-states-finetuned  145.0
20                       gdp growth     37.606516  1.424932e+03     37.748273   -118.432489  Czechia  single-state-finetuned  145.0
26                 urban population     32.076886  1.029785e+03     32.090266 -21195.349423  Czechia  single-state-finetuned  157.0
29  life expectancy at birth, total     38.797970  1.505381e+03     38.799241  -3026.047989  Czechia  single-state-finetuned  161.0
22                agricultural land     32.993471  1.089060e+03     33.000912 -33208.991711  Czechia  single-state-finetuned  161.0
28        adolescent fertility rate     44.405275  1.972437e+03     44.412131  -3121.651452  Czechia  single-state-finetuned  165.0
32                    net migration  88749.889436  7.881223e+09  88776.252416  -3745.877331  Czechia  group-states-finetuned  175.0
```


## Overall metrics - model comparision
```
           mae           mse         rmse           r2    state                   model  rank
0   201.705480  6.748126e+05   218.589290 -3776.426976  Czechia               base-lstm   5.0
1   213.272815  6.777218e+05   230.099935 -4901.675751  Czechia  single-state-finetuned   9.0
2  5922.628528  5.254150e+08  5924.563646 -1261.801786  Czechia  group-states-finetuned  10.0
```


