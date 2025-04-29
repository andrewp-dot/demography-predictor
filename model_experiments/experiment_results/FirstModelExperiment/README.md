
# FirstModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
             mae           mse          rmse        r2                           state          model    rank
42  6.120462e+06  5.945562e+14  7.710815e+06 -0.170721                         Comoros  feature_ARIMA    13.0
37  1.629761e+10  3.405784e+21  1.845477e+10 -0.345412  Central Europe and the Baltics  feature_ARIMA  2072.0
2   1.096780e+10  1.623932e+21  1.274337e+10 -0.420692      Africa Western and Central  feature_ARIMA  1925.0
84  4.164694e+08  1.973865e+18  4.442830e+08 -0.484228                       Guatemala  feature_ARIMA   563.0
46  1.098309e+09  2.026609e+19  1.423599e+09 -0.528000                   Cote d'Ivoire  feature_ARIMA   936.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
91   7.719091e+07  1.142937e+17  1.069083e+08 -1.128951e+26               Honduras  feature_ARIMA  1315.0
87   1.209139e+08  2.210125e+17  1.486664e+08 -1.297880e+26                 Guyana  feature_ARIMA  1384.0
226  1.723293e+07  3.745008e+15  1.935207e+07 -1.369377e+27                Vanuatu  feature_ARIMA  1163.0
86   4.175445e+07  2.411018e+16  4.910220e+07 -1.783865e+27          Guinea-Bissau  feature_ARIMA  1242.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  1186.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse         r2                                   state        model    rank
446  6.882683e+09  9.942232e+20  9.971085e+09 -13.163941                                Tanzania  feature_RNN  2011.0
283  1.208696e+09  3.044674e+19  1.744906e+09 -27.338305                           Cote d'Ivoire  feature_RNN  1329.0
239  3.420672e+10  2.130955e+22  4.616230e+10 -38.664133              Africa Western and Central  feature_RNN  2781.0
395  6.371621e+10  7.600662e+22  8.718178e+10 -41.276298                                 Nigeria  feature_RNN  3061.0
326  1.596995e+10  3.653046e+21  1.911301e+10 -41.743216  Heavily indebted poor countries (HIPC)  feature_RNN  2474.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
324  1.633715e+10  7.043502e+21  2.653960e+10 -6.098590e+31             Guyana  feature_RNN  3329.0
375  1.496143e+08  3.736606e+17  1.933032e+08 -6.525285e+31   Marshall Islands  feature_RNN  1483.0
313  1.323240e+11  4.706337e+23  2.169440e+11 -5.619996e+32              Gabon  feature_RNN  4120.0
264  7.995765e+10  1.492755e+23  1.221784e+11 -1.584539e+33  Brunei Darussalam  feature_RNN  3931.0
376  1.823105e+09  6.828066e+19  2.613058e+09 -4.944921e+33         Mauritania  feature_RNN  2396.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse        r2                       state        model    rank
920  2.733188e+09  1.374208e+20  3.707042e+09 -3.902954                    Tanzania  feature_GRU  1521.0
757  3.560567e+09  1.983433e+20  4.453581e+09 -4.329597               Cote d'Ivoire  feature_GRU  1609.0
740  1.590289e+09  4.352066e+19  2.086163e+09 -6.281693                Burkina Faso  feature_GRU  1265.0
713  1.593479e+10  3.170100e+21  1.780480e+10 -7.065894  Africa Western and Central  feature_GRU  2238.0
812  4.289750e+10  2.909685e+22  5.394150e+10 -7.755977                   Indonesia  feature_GRU  2674.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2                  state        model    rank
893  2.907737e+07  1.173388e+16  3.425483e+07 -1.294986e+30  Sao Tome and Principe  feature_GRU  1237.0
787  2.424491e+08  9.505913e+17  3.083174e+08 -2.268767e+30                  Gabon  feature_GRU  1599.0
849  2.898448e+07  1.011273e+16  3.180054e+07 -3.990352e+30       Marshall Islands  feature_GRU  1239.0
738  7.179006e+08  6.864777e+18  8.285396e+08 -6.521201e+30      Brunei Darussalam  feature_GRU  1924.0
850  2.661962e+08  1.146068e+18  3.385369e+08 -8.290764e+31             Mauritania  feature_GRU  1638.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2         state         model    rank
504  7.526793e+07  1.238082e+17  1.112784e+08 -2.015677       Burundi  feature_LSTM   267.0
503  6.047805e+08  6.452870e+18  8.033001e+08 -3.058291  Burkina Faso  feature_LSTM   841.0
577  8.922718e+09  1.145370e+21  1.070222e+10 -3.512220          Iraq  feature_LSTM  1964.0
660  1.934020e+08  6.459566e+17  2.541574e+08 -4.955696  Sierra Leone  feature_LSTM   533.0
610  6.103486e+08  5.473528e+18  7.398383e+08 -5.323287          Mali  feature_LSTM   880.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
561  1.707344e+08  4.357094e+17  2.087378e+08 -5.579149e+29             Guyana  feature_LSTM  1486.0
538  6.539688e+08  5.701228e+18  7.550655e+08 -9.903310e+29           Eswatini  feature_LSTM  1889.0
612  4.155074e+07  1.887215e+16  4.344237e+07 -1.423135e+30   Marshall Islands  feature_LSTM  1259.0
501  3.171944e+09  1.402339e+20  3.744782e+09 -1.826363e+31  Brunei Darussalam  feature_LSTM  2581.0
613  2.665482e+08  1.026901e+18  3.204535e+08 -7.960191e+31         Mauritania  feature_LSTM  1628.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2          state            model    rank
1028  1.667372e+09  3.111802e+19  1.764031e+09 -1.663154          Ghana  feature_LSTM_NN  1101.0
1134  1.720452e+08  4.811797e+17  2.193587e+08 -2.245237   Sierra Leone  feature_LSTM_NN   431.0
977   6.593540e+08  7.329320e+18  8.561170e+08 -2.456874   Burkina Faso  feature_LSTM_NN   852.0
994   2.233836e+09  7.016868e+19  2.648945e+09 -3.047127  Cote d'Ivoire  feature_LSTM_NN  1369.0
1013  2.278670e+09  6.496036e+19  2.548740e+09 -5.170331       Ethiopia  feature_LSTM_NN  1398.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2              state            model    rank
1012  8.338465e+08  9.259405e+18  9.622588e+08 -8.239116e+29           Eswatini  feature_LSTM_NN  1947.0
1035  9.847051e+07  1.360888e+17  1.166584e+08 -1.704977e+30             Guyana  feature_LSTM_NN  1374.0
1086  2.834658e+07  8.544478e+15  2.923114e+07 -1.719434e+30   Marshall Islands  feature_LSTM_NN  1228.0
975   1.915148e+09  5.005119e+19  2.237212e+09 -2.507907e+31  Brunei Darussalam  feature_LSTM_NN  2342.0
1087  2.204808e+08  6.231408e+17  2.496285e+08 -1.059318e+32         Mauritania  feature_LSTM_NN  1565.0
```


## Per target metrics - model comparision
```
                     target            model           mae           mse          rmse            r2  rank
0         agricultural_land    feature_ARIMA  1.038430e+00  4.068997e+00  1.180353e+00 -5.777964e+25    14
1         agricultural_land  feature_LSTM_NN  2.685988e+00  1.255860e+01  2.892581e+00 -2.228540e+27    29
2         agricultural_land     feature_LSTM  2.659216e+00  1.375354e+01  2.834605e+00 -3.475758e+27    28
3         agricultural_land      feature_GRU  3.742976e+00  2.671724e+01  4.010980e+00 -1.559060e+28    35
4         agricultural_land      feature_RNN  7.472887e+00  9.560387e+01  8.268118e+00 -1.080230e+29    38
5               arable_land    feature_ARIMA  6.640336e-01  2.085780e+00  7.520904e-01 -2.390997e+28     8
6               arable_land      feature_GRU  1.826414e+00  6.702921e+00  1.979404e+00 -4.148935e+30    20
7               arable_land     feature_LSTM  2.130137e+00  8.455499e+00  2.241893e+00 -4.299595e+30    25
8               arable_land  feature_LSTM_NN  1.842294e+00  5.917938e+00  1.950512e+00 -5.791022e+30    21
9               arable_land      feature_RNN  9.919478e+00  1.375982e+02  1.069747e+01 -3.079497e+32    40
10         death_rate_crude    feature_ARIMA  7.767297e-01  2.296183e+00  9.973218e-01 -8.457243e+00     9
11         death_rate_crude      feature_GRU  2.102672e+00  7.797430e+00  2.194312e+00 -4.742535e+01    24
12         death_rate_crude     feature_LSTM  1.823606e+00  5.833640e+00  1.939726e+00 -6.778163e+01    19
13         death_rate_crude  feature_LSTM_NN  1.926499e+00  6.240270e+00  2.033021e+00 -7.136230e+01    22
14         death_rate_crude      feature_RNN  2.281445e+00  8.711195e+00  2.390638e+00 -9.237594e+01    26
15     fertility_rate_total    feature_ARIMA  1.696104e-01  6.530363e-02  2.010135e-01 -5.781290e+00     1
16     fertility_rate_total     feature_LSTM  2.855739e-01  1.363684e-01  3.042722e-01 -3.318900e+01     2
17     fertility_rate_total  feature_LSTM_NN  2.882275e-01  1.414434e-01  3.066510e-01 -3.477256e+01     3
18     fertility_rate_total      feature_GRU  3.599682e-01  2.295837e-01  3.837034e-01 -3.506364e+01     5
19     fertility_rate_total      feature_RNN  3.513191e-01  2.189876e-01  3.806295e-01 -4.094228e+01     4
20                      gdp    feature_ARIMA  6.096785e+11  5.157204e+24  6.878604e+11 -1.266974e+01    46
21                      gdp  feature_LSTM_NN  1.172570e+12  2.489573e+25  1.261402e+12 -1.103436e+02    49
22                      gdp     feature_LSTM  8.659802e+11  5.703709e+24  1.010744e+12 -3.462119e+02    47
23                      gdp      feature_GRU  1.054332e+12  1.328399e+25  1.219431e+12 -9.468071e+02    48
24                      gdp      feature_RNN  6.409605e+12  4.713478e+26  8.375936e+12 -7.477298e+05    50
25               gdp_growth      feature_RNN  3.581535e+00  3.943214e+01  4.736175e+00 -9.424979e-01    34
26               gdp_growth  feature_LSTM_NN  3.231130e+00  3.647218e+01  4.448208e+00 -1.163278e+00    32
27               gdp_growth     feature_LSTM  3.162147e+00  3.541902e+01  4.334448e+00 -1.304431e+00    31
28               gdp_growth      feature_GRU  3.805785e+00  4.205549e+01  4.987700e+00 -1.381717e+00    36
29               gdp_growth    feature_ARIMA  3.402220e+00  4.058877e+01  4.641870e+00 -1.808673e+00    33
30            net_migration  feature_LSTM_NN  1.905681e+08  1.654469e+18  2.610471e+08 -1.413341e+08    43
31            net_migration      feature_RNN  1.794536e+08  2.638960e+18  3.176750e+08 -1.456851e+09    42
32            net_migration      feature_GRU  2.422500e+08  1.074271e+19  4.290501e+08 -1.039213e+10    44
33            net_migration     feature_LSTM  1.121744e+09  1.131857e+20  1.797657e+09 -7.861669e+10    45
34            net_migration    feature_ARIMA  5.755165e+07  6.720970e+17  7.494914e+07 -7.551326e+11    41
35        population_growth      feature_GRU  8.384010e-01  1.384042e+00  9.051521e-01 -4.524698e+01    10
36        population_growth    feature_ARIMA  4.733579e-01  2.272859e+00  5.601265e-01 -5.752578e+01     6
37        population_growth      feature_RNN  9.155131e-01  1.600889e+00  9.885591e-01 -6.625210e+01    13
38        population_growth  feature_LSTM_NN  8.712777e-01  1.642225e+00  9.487830e-01 -8.306054e+01    11
39        population_growth     feature_LSTM  8.810792e-01  1.486249e+00  9.464606e-01 -8.353382e+01    12
40  rural_population_growth      feature_GRU  1.704179e+00  7.623504e+01  2.040949e+00 -9.604852e+01    18
41  rural_population_growth  feature_LSTM_NN  1.524109e+00  7.807376e+01  1.876351e+00 -1.308944e+02    16
42  rural_population_growth      feature_RNN  1.971053e+00  7.826854e+01  2.326363e+00 -1.930594e+02    23
43  rural_population_growth     feature_LSTM  1.633099e+00  7.912837e+01  1.977660e+00 -2.009024e+02    17
44  rural_population_growth    feature_ARIMA  1.074182e+00  7.500817e+01  1.452544e+00 -3.396931e+02    15
45         urban_population    feature_ARIMA  6.081716e-01  1.416542e+00  7.463955e-01 -1.070559e+00     7
46         urban_population     feature_LSTM  2.815065e+00  1.415191e+01  2.958161e+00 -1.213206e+02    30
47         urban_population  feature_LSTM_NN  2.605639e+00  1.249574e+01  2.718504e+00 -1.734916e+02    27
48         urban_population      feature_RNN  4.975389e+00  4.739984e+01  5.413435e+00 -6.747598e+02    37
49         urban_population      feature_GRU  7.487876e+00  8.170120e+01  7.928412e+00 -8.074438e+02    39
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2            model  rank
0  7.091260e+10  6.058555e+23  8.002775e+10 -2.346945e+27    feature_ARIMA   4.0
2  9.766180e+10  6.576084e+23  1.145956e+11 -4.791392e+29     feature_LSTM   9.0
3  1.210523e+11  1.557443e+24  1.402378e+11 -4.688677e+29      feature_GRU  11.0
4  1.356567e+11  2.922269e+24  1.460403e+11 -6.456345e+29  feature_LSTM_NN  16.0
1  7.354832e+11  5.522964e+25  9.622243e+11 -3.410733e+31      feature_RNN  20.0
```


