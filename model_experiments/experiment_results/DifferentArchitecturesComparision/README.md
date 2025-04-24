
# DifferentArchitecturesComparision

**Description:** Compares performance of different architecture models.

## Per target metrics - model comparision
```
                     target        mae          mse       rmse            r2          state       model   rank
28        agricultural land   1.012052     1.204830   1.097647 -3.574029e+01        Czechia   base-lstm  105.0
52        agricultural land   4.617544    22.544756   4.748132 -6.864838e+02        Czechia    base-gru  227.0
4         agricultural land  17.735073   370.302489  19.243245 -1.129107e+04        Czechia  simple-rnn  270.0
25              arable land   0.496319     0.338150   0.581507 -5.123930e+01        Czechia   base-lstm   79.0
49              arable land   2.664760     7.592268   2.755407 -1.171896e+03        Czechia    base-gru  190.0
1               arable land  16.237014   294.524803  17.161725 -4.549883e+04        Czechia  simple-rnn  269.0
27        death rate, crude   1.070519     1.364578   1.168151 -8.299830e-02        Czechia   base-lstm   73.0
51        death rate, crude   1.610600     3.255196   1.804216 -1.583489e+00        Czechia    base-gru  111.0
3         death rate, crude   2.540282     7.641447   2.764317 -5.064640e+00        Czechia  simple-rnn  155.0
24    fertility rate, total   0.110808     0.013842   0.117652 -2.830234e+00        Czechia   base-lstm   24.0
48    fertility rate, total   0.234633     0.060727   0.246429 -1.580387e+01        Czechia    base-gru   40.0
0     fertility rate, total   0.852764     0.866764   0.931002 -2.388426e+02        Czechia  simple-rnn  103.0
2                gdp growth   2.795465    13.147909   3.626005 -1.020086e-01        Czechia  simple-rnn  148.0
26               gdp growth   3.536353    14.265634   3.776987 -1.956921e-01        Czechia   base-lstm  157.0
50               gdp growth   4.248990    18.804241   4.336386 -5.761013e-01        Czechia    base-gru  172.0
31        population growth   1.137983     1.305476   1.142574 -1.121587e+00        Czechia   base-lstm   83.0
7         population growth   0.921591     1.647209   1.283436 -1.676952e+00        Czechia  simple-rnn   85.0
55        population growth   1.618141     3.104362   1.761920 -4.045037e+00        Czechia    base-gru  115.0
5   rural population growth   1.705585     3.175548   1.782007 -3.513602e+00        Czechia  simple-rnn  118.0
29  rural population growth   1.757328     3.584758   1.893346 -4.095237e+00        Czechia   base-lstm  128.0
53  rural population growth   2.653829     7.702013   2.775250 -9.947344e+00        Czechia    base-gru  159.0
30         urban population   1.663959     2.791731   1.670847 -5.646296e+01        Czechia   base-lstm  135.0
54         urban population   7.434223    57.327240   7.571475 -1.178982e+03        Czechia    base-gru  241.0
6          urban population   7.363032    58.425343   7.643647 -1.201585e+03        Czechia  simple-rnn  243.0
36        agricultural land   4.007347    18.298695   4.277697 -1.866928e+01       Honduras   base-lstm  188.0
60        agricultural land  10.249706   120.743563  10.988338 -1.287873e+02       Honduras    base-gru  244.0
12        agricultural land  29.290654  1060.647300  32.567581 -1.139090e+03       Honduras  simple-rnn  274.0
33              arable land   0.486335     0.265570   0.515335 -8.416257e+28       Honduras   base-lstm   97.0
57              arable land   2.326848     6.509558   2.551383 -2.062961e+30       Honduras    base-gru  189.0
9               arable land   7.976723    69.603605   8.342877 -2.205826e+31       Honduras  simple-rnn  255.0
11        death rate, crude   0.494193     0.300537   0.548212 -8.805454e-01       Honduras  simple-rnn   43.0
59        death rate, crude   1.759541     3.307487   1.818650 -1.969590e+01       Honduras    base-gru  135.0
35        death rate, crude   2.810791     8.076695   2.841953 -4.953821e+01       Honduras   base-lstm  183.0
56    fertility rate, total   0.253577     0.067419   0.259652 -4.634441e-01       Honduras    base-gru   20.0
32    fertility rate, total   0.298978     0.090361   0.300601 -9.614409e-01       Honduras   base-lstm   29.0
8     fertility rate, total   0.967814     1.090090   1.044074 -2.266223e+01       Honduras  simple-rnn   97.0
34               gdp growth   2.746906    21.081846   4.591497 -6.144070e-03       Honduras   base-lstm  156.0
58               gdp growth   3.415261    22.517549   4.745266 -7.466391e-02       Honduras    base-gru  164.0
10               gdp growth   2.866517    23.188489   4.815443 -1.066849e-01       Honduras  simple-rnn  169.0
39        population growth   0.778917     0.610607   0.781414 -1.941992e+01       Honduras   base-lstm   78.0
63        population growth   0.934297     0.875173   0.935507 -2.826753e+01       Honduras    base-gru   92.0
15        population growth   0.871691     1.023406   1.011635 -3.322474e+01       Honduras  simple-rnn   93.0
61  rural population growth   0.313305     0.120018   0.346436 -1.382281e+00       Honduras    base-gru   34.0
37  rural population growth   0.367119     0.152262   0.390208 -2.022304e+00       Honduras   base-lstm   44.0
13  rural population growth   2.145853     5.618159   2.370266 -1.105169e+02       Honduras  simple-rnn  166.0
38         urban population   4.681563    24.034744   4.902524 -3.094886e+00       Honduras   base-lstm  197.0
62         urban population  10.538045   126.718104  11.256914 -2.058942e+01       Honduras    base-gru  231.0
14         urban population  20.960648   595.804260  24.409102 -1.005093e+02       Honduras  simple-rnn  260.0
44        agricultural land   4.068922    18.778761   4.333447 -7.680521e+02  United States   base-lstm  219.0
68        agricultural land   8.446469    81.918611   9.050890 -3.353837e+03  United States    base-gru  251.0
20        agricultural land  35.864437  1637.065511  40.460666 -6.704222e+04  United States  simple-rnn  285.0
41              arable land   0.668245     0.571830   0.756194 -4.096004e+01  United States   base-lstm   83.0
65              arable land   2.728875     9.495173   3.081424 -6.957422e+02  United States    base-gru  196.0
17              arable land  18.614577   387.958325  19.696658 -2.846683e+04  United States  simple-rnn  274.0
43        death rate, crude   0.968123     1.049593   1.024496 -7.638276e-01  United States   base-lstm   71.0
67        death rate, crude   1.449385     2.390465   1.546113 -3.017147e+00  United States    base-gru  103.0
19        death rate, crude   3.568911    14.495450   3.807289 -2.335942e+01  United States  simple-rnn  191.0
40    fertility rate, total   0.320540     0.121104   0.348001 -1.036849e+01  United States   base-lstm   50.0
64    fertility rate, total   0.667161     0.514166   0.717054 -4.726652e+01  United States    base-gru   82.0
16    fertility rate, total   1.593253     2.935891   1.713444 -2.746022e+02  United States  simple-rnn  142.0
18               gdp growth   1.984162     6.537359   2.556826 -5.075883e-01  United States  simple-rnn  126.0
42               gdp growth   2.505413     7.267534   2.695836 -6.759748e-01  United States   base-lstm  134.0
66               gdp growth   3.043368    10.551466   3.248302 -1.433286e+00  United States    base-gru  161.0
47        population growth   0.669130     0.513564   0.716634 -1.216575e+01  United States   base-lstm   68.0
71        population growth   1.267766     1.674128   1.293881 -4.191799e+01  United States    base-gru  121.0
23        population growth   1.716356     4.406394   2.099141 -1.119624e+02  United States  simple-rnn  157.0
45  rural population growth   1.243254     1.665333   1.290478 -2.029800e+01  United States   base-lstm  110.0
69  rural population growth   2.005252     4.181917   2.044974 -5.248269e+01  United States    base-gru  155.0
21  rural population growth   1.876730     5.163430   2.272318 -6.503529e+01  United States  simple-rnn  159.0
46         urban population   0.220719     0.051333   0.226569  8.976576e-01  United States   base-lstm    7.0
70         urban population  10.264543   116.756912  10.805411 -2.317758e+02  United States    base-gru  244.0
22         urban population  13.283773   246.838634  15.711099 -4.911169e+02  United States  simple-rnn  255.0
```


## Overall metrics - model comparision
```
        mae         mse       rmse            r2          state       model  rank
3  1.348165    3.108625   1.431089 -1.897104e+01        Czechia   base-lstm   6.0
6  3.135340   15.048851   3.249902 -3.836647e+02        Czechia    base-gru  15.0
0  6.268851   93.716439   6.804423 -7.280086e+03        Czechia  simple-rnn  26.0
4  2.022245    9.076348   2.325154 -1.052032e+28       Honduras   base-lstm  16.0
7  3.723823   35.107359   4.112768 -2.578701e+29       Honduras    base-gru  25.0
1  8.196762  219.659481   9.388649 -2.757283e+30       Honduras  simple-rnn  33.0
5  1.333043    3.752382   1.423957 -1.065483e+02  United States   base-lstm   6.0
8  3.734102   28.435355   3.973506 -5.534341e+02  United States    base-gru  20.0
2  9.812775  288.175124  11.039680 -1.205945e+04  United States  simple-rnn  33.0
```


