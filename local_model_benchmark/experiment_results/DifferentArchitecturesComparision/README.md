
# DifferentArchitecturesComparision

**Description:** Compares performance of different architecture models.

## Per target metrics - model comparision
```
                     target           mae           mse          rmse            r2          state               model   rank
32        agricultural land  1.459512e+00  2.224227e+00  1.491384e+00 -6.682596e+01        Czechia    future-base-lstm  164.0
5         agricultural land  5.126958e+00  2.649787e+01  5.147608e+00 -8.070306e+02        Czechia           base-lstm  264.0
59        agricultural land  6.444277e+00  4.265826e+01  6.531329e+00 -1.299829e+03        Czechia  simple-funnel-lstm  276.0
29              arable land  1.883568e+00  3.772395e+00  1.942265e+00 -5.817806e+02        Czechia    future-base-lstm  183.0
2               arable land  3.147906e+00  9.993479e+00  3.161246e+00 -1.542848e+03        Czechia           base-lstm  230.0
56              arable land  8.285101e+00  6.998756e+01  8.365857e+00 -1.081107e+04        Czechia  simple-funnel-lstm  289.0
4         death rate, crude  8.085382e-01  1.667473e+00  1.291307e+00 -3.233909e-01        Czechia           base-lstm  103.0
31        death rate, crude  1.232552e+00  1.750529e+00  1.323076e+00 -3.893090e-01        Czechia    future-base-lstm  113.0
58        death rate, crude  3.625059e+00  1.459007e+01  3.819696e+00 -1.057942e+01        Czechia  simple-funnel-lstm  216.0
54    fertility rate, total  3.158481e-02  1.605720e-03  4.007144e-02  5.556808e-01        Czechia  simple-funnel-lstm    6.0
0     fertility rate, total  5.620412e-02  3.949772e-03  6.284721e-02 -9.294232e-02        Czechia           base-lstm   18.0
27    fertility rate, total  3.781116e-01  1.463551e-01  3.825638e-01 -3.949795e+01        Czechia    future-base-lstm   84.0
57               gdp growth  2.600106e+00  1.199845e+01  3.463878e+00 -5.665610e-03        Czechia  simple-funnel-lstm  160.0
3                gdp growth  2.510674e+00  1.678216e+01  4.096603e+00 -4.066176e-01        Czechia           base-lstm  186.0
30               gdp growth  3.965108e+00  1.674235e+01  4.091742e+00 -4.032817e-01        Czechia    future-base-lstm  200.0
8         population growth  4.140534e-01  7.022652e-01  8.380126e-01 -1.412822e-01        Czechia           base-lstm   76.0
62        population growth  6.921295e-01  6.722723e-01  8.199221e-01 -9.253945e-02        Czechia  simple-funnel-lstm   81.0
35        population growth  9.941301e-01  1.124342e+00  1.060350e+00 -8.272181e-01        Czechia    future-base-lstm  103.0
55        population, total  1.280158e+06  1.655538e+12  1.286677e+06 -3.971222e+02        Czechia  simple-funnel-lstm  287.0
28        population, total  5.626801e+06  3.201827e+13  5.658469e+06 -7.698724e+03        Czechia    future-base-lstm  303.0
1         population, total  1.894216e+07  3.745325e+14  1.935284e+07 -9.006623e+04        Czechia           base-lstm  309.0
6   rural population growth  1.078816e+00  1.169540e+00  1.081453e+00 -6.623389e-01        Czechia           base-lstm  105.0
60  rural population growth  1.378938e+00  1.971525e+00  1.404110e+00 -1.802249e+00        Czechia  simple-funnel-lstm  127.0
33  rural population growth  1.388677e+00  2.061044e+00  1.435634e+00 -1.929489e+00        Czechia    future-base-lstm  131.0
34         urban population  2.358199e+00  5.950487e+00  2.439362e+00 -1.214805e+02        Czechia    future-base-lstm  188.0
7          urban population  3.815206e+00  1.467059e+01  3.830221e+00 -3.009688e+02        Czechia           base-lstm  235.0
61         urban population  3.827489e+00  1.525001e+01  3.905127e+00 -3.128952e+02        Czechia  simple-funnel-lstm  241.0
68        agricultural land  2.475210e+00  7.268399e+00  2.695997e+00 -6.812805e+00       Honduras  simple-funnel-lstm  173.0
41        agricultural land  2.649408e+00  8.269050e+00  2.875596e+00 -7.888405e+00       Honduras    future-base-lstm  190.0
14        agricultural land  2.738519e+00  7.897851e+00  2.810312e+00 -7.489404e+00       Honduras           base-lstm  190.0
38              arable land  4.102844e-01  2.350986e-01  4.848697e-01 -7.450572e+28       Honduras    future-base-lstm  123.0
65              arable land  1.640257e+00  2.885565e+00  1.698695e+00 -9.144721e+29       Honduras  simple-funnel-lstm  190.0
11              arable land  2.597749e+00  7.386376e+00  2.717789e+00 -2.340836e+30       Honduras           base-lstm  218.0
67        death rate, crude  2.698741e+00  7.463343e+00  2.731912e+00 -4.570029e+01       Honduras  simple-funnel-lstm  195.0
40        death rate, crude  2.710966e+00  7.488473e+00  2.736507e+00 -4.585753e+01       Honduras    future-base-lstm  199.0
13        death rate, crude  2.880803e+00  8.418123e+00  2.901400e+00 -5.167462e+01       Honduras           base-lstm  209.0
36    fertility rate, total  1.675589e-01  2.844617e-02  1.686599e-01  3.825285e-01       Honduras    future-base-lstm   19.0
63    fertility rate, total  3.901783e-01  1.599589e-01  3.999486e-01 -2.472174e+00       Honduras  simple-funnel-lstm   66.0
9     fertility rate, total  4.578843e-01  2.134263e-01  4.619809e-01 -3.632775e+00       Honduras           base-lstm   79.0
66               gdp growth  2.543899e+00  2.162156e+01  4.649898e+00 -3.190204e-02       Honduras  simple-funnel-lstm  177.0
39               gdp growth  2.975624e+00  2.226911e+01  4.719016e+00 -6.280691e-02       Honduras    future-base-lstm  191.0
12               gdp growth  3.587056e+00  2.697734e+01  5.193972e+00 -2.875104e-01       Honduras           base-lstm  206.0
17        population growth  3.044637e-01  9.787584e-02  3.128511e-01 -2.273164e+00       Honduras           base-lstm   56.0
71        population growth  3.388945e-01  1.161600e-01  3.408225e-01 -2.884622e+00       Honduras  simple-funnel-lstm   62.0
44        population growth  5.890232e-01  3.573264e-01  5.977678e-01 -1.094971e+01       Honduras    future-base-lstm  106.0
64        population, total  2.339857e+05  8.965835e+10  2.994300e+05  7.700633e-01       Honduras  simple-funnel-lstm  220.0
37        population, total  3.642119e+06  1.669601e+13  4.086076e+06 -4.181840e+01       Honduras    future-base-lstm  277.0
10        population, total  2.888299e+07  1.215202e+15  3.485974e+07 -3.115492e+03       Honduras           base-lstm  307.0
42  rural population growth  1.076723e-01  1.261625e-02  1.123221e-01  7.495754e-01       Honduras    future-base-lstm   12.0
69  rural population growth  1.953192e-01  4.403137e-02  2.098365e-01  1.260052e-01       Honduras  simple-funnel-lstm   23.0
15  rural population growth  5.088403e-01  2.692040e-01  5.188487e-01 -4.343528e+00       Honduras           base-lstm   86.0
16         urban population  2.946872e+00  9.296418e+00  3.049003e+00 -5.838642e-01       Honduras           base-lstm  176.0
70         urban population  3.445570e+00  1.417728e+01  3.765273e+00 -1.415434e+00       Honduras  simple-funnel-lstm  190.0
43         urban population  7.529993e+00  6.093659e+01  7.806190e+00 -9.381987e+00       Honduras    future-base-lstm  255.0
23        agricultural land  1.112416e+00  1.297140e+00  1.138920e+00 -5.212215e+01  United States           base-lstm  146.0
50        agricultural land  1.845885e+00  3.461213e+00  1.860433e+00 -1.407481e+02  United States    future-base-lstm  174.0
77        agricultural land  6.087227e+00  4.035353e+01  6.352443e+00 -1.651610e+03  United States  simple-funnel-lstm  275.0
47              arable land  6.038773e-01  4.790788e-01  6.921552e-01 -3.415412e+01  United States    future-base-lstm  115.0
20              arable land  3.831291e+00  1.488822e+01  3.858526e+00 -1.091477e+03  United States           base-lstm  245.0
74              arable land  4.981643e+00  2.710037e+01  5.205801e+00 -1.987586e+03  United States  simple-funnel-lstm  272.0
22        death rate, crude  9.807999e-01  1.021864e+00  1.010873e+00 -7.172290e-01  United States           base-lstm   99.0
76        death rate, crude  8.516413e-01  1.385167e+00  1.176931e+00 -1.327756e+00  United States  simple-funnel-lstm  110.0
49        death rate, crude  1.832486e+00  3.969699e+00  1.992410e+00 -5.671030e+00  United States    future-base-lstm  156.0
45    fertility rate, total  9.522402e-02  1.364885e-02  1.168283e-01 -2.812647e-01  United States    future-base-lstm   25.0
72    fertility rate, total  2.427957e-01  7.606273e-02  2.757947e-01 -6.140271e+00  United States  simple-funnel-lstm   62.0
18    fertility rate, total  6.111160e-01  4.073585e-01  6.382464e-01 -3.724015e+01  United States           base-lstm  113.0
75               gdp growth  1.259829e+00  4.566497e+00  2.136936e+00 -5.308541e-02  United States  simple-funnel-lstm  124.0
48               gdp growth  1.606844e+00  4.488792e+00  2.118677e+00 -3.516572e-02  United States    future-base-lstm  125.0
21               gdp growth  3.551231e+00  1.674655e+01  4.092255e+00 -2.861942e+00  United States           base-lstm  209.0
80        population growth  2.218508e-01  9.711488e-02  3.116326e-01 -1.489639e+00  United States  simple-funnel-lstm   49.0
53        population growth  3.948675e-01  1.918894e-01  4.380519e-01 -3.919283e+00  United States    future-base-lstm   75.0
26        population growth  1.059676e+00  1.200280e+00  1.095573e+00 -2.977041e+01  United States           base-lstm  133.0
73        population, total  1.508510e+08  2.417328e+16  1.554776e+08 -3.710186e+02  United States  simple-funnel-lstm  301.0
46        population, total  4.820031e+08  2.814129e+17  5.304837e+08 -4.329851e+03  United States    future-base-lstm  314.0
19        population, total  9.370597e+08  1.144675e+18  1.069895e+09 -1.761516e+04  United States           base-lstm  320.0
51  rural population growth  5.238039e-01  3.196874e-01  5.654091e-01 -3.088494e+00  United States    future-base-lstm   85.0
78  rural population growth  5.433321e-01  3.572638e-01  5.977155e-01 -3.569060e+00  United States  simple-funnel-lstm   89.0
24  rural population growth  5.554433e-01  4.484179e-01  6.696402e-01 -4.734833e+00  United States           base-lstm  100.0
25         urban population  2.127378e+00  4.697985e+00  2.167484e+00 -8.366274e+00  United States           base-lstm  170.0
79         urban population  4.615026e+00  2.397799e+01  4.896733e+00 -4.680441e+01  United States  simple-funnel-lstm  248.0
52         urban population  8.710632e+00  8.284981e+01  9.102187e+00 -1.641759e+02  United States    future-base-lstm  277.0
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2          state               model  rank
6  1.422428e+05  1.839486e+11  1.429673e+05 -1.425871e+03        Czechia  simple-funnel-lstm  10.0
3  6.252017e+05  3.557585e+12  6.287203e+05 -9.457620e+02        Czechia    future-base-lstm  15.0
0  2.104686e+06  4.161472e+13  2.150318e+06 -1.030208e+04        Czechia           base-lstm  21.0
7  2.599994e+04  9.962039e+09  3.327184e+04 -1.016080e+29       Honduras  simple-funnel-lstm  11.0
4  4.046817e+05  1.855113e+12  4.540106e+05 -8.278413e+27       Honduras    future-base-lstm  16.0
1  3.209222e+06  1.350224e+14  3.873307e+06 -2.600929e+29       Honduras           base-lstm  27.0
8  1.676122e+07  2.685920e+15  1.727529e+07 -4.521777e+02  United States  simple-funnel-lstm  22.0
5  5.355590e+07  3.126810e+16  5.894263e+07 -5.202138e+02  United States    future-base-lstm  26.0
2  1.041177e+08  1.271861e+17  1.188772e+08 -2.093605e+03  United States           base-lstm  32.0
```


