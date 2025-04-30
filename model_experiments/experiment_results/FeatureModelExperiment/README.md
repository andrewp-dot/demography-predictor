
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse        r2                                       state          model    rank
202  1.510947e+10  3.059984e+21  1.749285e+10 -0.038884  Sub-Saharan Africa (excluding high income)  feature_ARIMA  3922.0
46   1.098309e+09  2.026609e+19  1.423599e+09 -0.323639                               Cote d'Ivoire  feature_ARIMA  1906.0
209  1.430135e+09  3.175031e+19  1.781874e+09 -0.325052                                    Tanzania  feature_ARIMA  2084.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
151  3.099315e+08  1.165523e+18  3.414012e+08 -7.994873e+27             Mozambique  feature_ARIMA  3196.0
139  2.757708e+08  8.585910e+17  2.930179e+08 -3.090526e+28             Mauritania  feature_ARIMA  3122.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  2350.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse         r2                       state        model    rank
446  6.882683e+09  9.942232e+20  9.971085e+09 -13.163941                    Tanzania  feature_RNN  3831.0
283  1.208696e+09  3.044674e+19  1.744906e+09 -27.338305               Cote d'Ivoire  feature_RNN  2580.0
239  3.420672e+10  2.130955e+22  4.616230e+10 -38.664133  Africa Western and Central  feature_RNN  5239.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
313  1.323240e+11  4.706337e+23  2.169440e+11 -5.619996e+32              Gabon  feature_RNN  8113.0
264  7.995765e+10  1.492755e+23  1.221784e+11 -1.584539e+33  Brunei Darussalam  feature_RNN  7732.0
376  1.823105e+09  6.828066e+19  2.613058e+09 -4.944921e+33         Mauritania  feature_RNN  4787.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse        r2          state        model    rank
920  2.733188e+09  1.374208e+20  3.707042e+09 -3.902954       Tanzania  feature_GRU  2861.0
757  3.560567e+09  1.983433e+20  4.453581e+09 -4.329597  Cote d'Ivoire  feature_GRU  3045.0
740  1.590289e+09  4.352066e+19  2.086163e+09 -6.281693   Burkina Faso  feature_GRU  2425.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2              state        model    rank
849  2.898448e+07  1.011273e+16  3.180054e+07 -3.990352e+30   Marshall Islands  feature_GRU  2457.0
738  7.179006e+08  6.864777e+18  8.285396e+08 -6.521201e+30  Brunei Darussalam  feature_GRU  3865.0
850  2.661962e+08  1.146068e+18  3.385369e+08 -8.290764e+31         Mauritania  feature_GRU  3259.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2         state         model    rank
504  7.526793e+07  1.238082e+17  1.112784e+08 -2.015677       Burundi  feature_LSTM   487.0
503  6.047805e+08  6.452870e+18  8.033001e+08 -3.058291  Burkina Faso  feature_LSTM  1555.0
577  8.922718e+09  1.145370e+21  1.070222e+10 -3.512220          Iraq  feature_LSTM  3712.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
612  4.155074e+07  1.887215e+16  4.344237e+07 -1.423135e+30   Marshall Islands  feature_LSTM  2506.0
501  3.171944e+09  1.402339e+20  3.744782e+09 -1.826363e+31  Brunei Darussalam  feature_LSTM  5117.0
613  2.665482e+08  1.026901e+18  3.204535e+08 -7.960191e+31         Mauritania  feature_LSTM  3239.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2                state            model    rank
1162  9.040271e+08  1.286894e+19  1.134415e+09 -3.114851  Trinidad and Tobago  feature_LSTM_NN  1835.0
1041  3.195958e+09  1.204612e+20  3.470754e+09 -3.703005              Hungary  feature_LSTM_NN  2861.0
994   3.162661e+09  1.295949e+20  3.599938e+09 -4.184223        Cote d'Ivoire  feature_LSTM_NN  2890.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2              state            model    rank
1024  1.087241e+09  1.817877e+19  1.348296e+09 -1.931176e+30              Gabon  feature_LSTM_NN  4203.0
975   9.153470e+08  1.115909e+19  1.056366e+09 -3.843584e+31  Brunei Darussalam  feature_LSTM_NN  4049.0
1087  1.532064e+08  2.720760e+17  1.649479e+08 -1.829739e+32         Mauritania  feature_LSTM_NN  2951.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse        r2    state           model    rank
1480  2.827532e+09  1.717436e+20  4.144201e+09 -3.083544  Ecuador  feature_GRU_NN  2902.0
1502  8.285347e+08  1.168471e+19  1.080959e+09 -3.199194    Ghana  feature_GRU_NN  1785.0
1460  4.112090e+08  2.394456e+18  4.893345e+08 -3.538476     Chad  feature_GRU_NN  1285.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1486  1.524428e+09  3.507314e+19  1.872783e+09 -9.348039e+29           Eswatini  feature_GRU_NN  4457.0
1449  1.000369e+09  1.370716e+19  1.170776e+09 -3.827926e+30  Brunei Darussalam  feature_GRU_NN  4113.0
1561  3.613487e+08  2.074467e+18  4.554639e+08 -2.516101e+31         Mauritania  feature_GRU_NN  3440.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse         r2               state           model    rank
1231  6.759557e+08  5.318042e+18  7.292571e+08  -7.090777       Cote d'Ivoire  feature_RNN_NN  1668.0
1193  4.403145e+10  3.576733e+22  5.980579e+10 -32.484311           Argentina  feature_RNN_NN  5433.0
1415  3.476978e+08  1.570936e+18  3.963513e+08 -37.495494  West Bank and Gaza  feature_RNN_NN  1657.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1261  1.598374e+11  5.679189e+23  2.385051e+11 -1.437269e+32              Gabon  feature_RNN_NN  8191.0
1324  3.074809e+09  1.507374e+20  3.882493e+09 -3.052092e+32         Mauritania  feature_RNN_NN  5145.0
1212  1.247809e+11  3.119675e+23  1.766272e+11 -4.413456e+32  Brunei Darussalam  feature_RNN_NN  8022.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse        r2         state                    model    rank
1762  5.528988e+09  4.734387e+20  6.880715e+09 -1.111257          Iraq  feature_univariate_LSTM  3223.0
1689  6.615093e+07  8.383600e+16  9.157127e+07 -2.104262       Burundi  feature_univariate_LSTM   426.0
1688  2.679572e+08  8.712568e+17  2.951729e+08 -2.529302  Burkina Faso  feature_univariate_LSTM   929.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse            r2              state                    model    rank
1735  8.459168e+08  1.197454e+19  1.094284e+09 -2.300467e+30              Gabon  feature_univariate_LSTM  4023.0
1686  1.563661e+09  3.372139e+19  1.836339e+09 -8.638407e+30  Brunei Darussalam  feature_univariate_LSTM  4482.0
1798  1.984781e+08  4.700756e+17  2.168149e+08 -4.809654e+31         Mauritania  feature_univariate_LSTM  3076.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse        r2    state                   model    rank
2236  8.673001e+09  1.013256e+21  1.006609e+10 -2.061953     Iraq  feature_univariate_RNN  3601.0
2213  1.342707e+09  2.315904e+19  1.521811e+09 -2.401096    Ghana  feature_univariate_RNN  2055.0
2163  7.413743e+07  1.174861e+17  1.084003e+08 -2.752802  Burundi  feature_univariate_RNN   509.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse            r2                  state                   model    rank
2315  2.095367e+07  5.227631e+15  2.286499e+07 -6.583401e+29  Sao Tome and Principe  feature_univariate_RNN  2391.0
2160  2.002935e+09  5.545627e+19  2.354915e+09 -2.141307e+30      Brunei Darussalam  feature_univariate_RNN  4724.0
2272  2.564701e+08  8.750473e+17  2.958153e+08 -1.195389e+31             Mauritania  feature_univariate_RNN  3191.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse        r2         state                   model    rank
1999  9.807201e+09  1.365129e+21  1.168390e+10 -1.642449          Iraq  feature_univariate_GRU  3706.0
1926  7.480658e+07  1.138429e+17  1.067065e+08 -1.992351       Burundi  feature_univariate_GRU   470.0
1925  4.980777e+08  4.235350e+18  6.507980e+08 -2.160597  Burkina Faso  feature_univariate_GRU  1363.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse            r2              state                   model    rank
1972  1.373178e+09  3.306848e+19  1.818475e+09 -1.764427e+30              Gabon  feature_univariate_GRU  4422.0
1923  2.224049e+09  6.808876e+19  2.609382e+09 -5.963579e+30  Brunei Darussalam  feature_univariate_GRU  4830.0
2035  2.877280e+08  1.151208e+18  3.392968e+08 -3.231992e+31         Mauritania  feature_univariate_GRU  3269.0
```


## Per target metrics - model comparision
```
               target                   model       mae        mse      rmse            r2  rank
0   agricultural_land           feature_ARIMA  1.046899   4.060992  1.190379 -5.786358e+25    20
1   agricultural_land         feature_LSTM_NN  3.121630  16.556435  3.335814 -1.695721e+27    60
2   agricultural_land            feature_LSTM  2.659216  13.753538  2.834605 -3.475758e+27    55
3   agricultural_land  feature_univariate_RNN  2.015450   9.051573  2.139661 -4.154811e+27    45
4   agricultural_land          feature_GRU_NN  2.318851  11.019361  2.495642 -4.253956e+27    54
..                ...                     ...       ...        ...       ...           ...   ...
95   urban_population  feature_univariate_RNN  2.815074  13.421986  2.909651 -2.308710e+02    58
96   urban_population          feature_GRU_NN  3.620825  23.650513  3.873111 -2.381178e+02    68
97   urban_population             feature_RNN  4.975389  47.399838  5.413435 -6.747598e+02    74
98   urban_population             feature_GRU  7.487876  81.701199  7.928412 -8.074438e+02    79
99   urban_population          feature_RNN_NN  6.826477  79.144182  7.283915 -1.499615e+03    77

[100 rows x 7 columns]
```


## Best model for each target
```
                    target                    model           mae           mse          rmse            r2  rank
0        agricultural_land            feature_ARIMA  1.046899e+00  4.060992e+00  1.190379e+00 -5.786358e+25    20
1              arable_land            feature_ARIMA  6.680484e-01  2.092435e+00  7.559976e-01 -2.555029e+28    14
2         death_rate_crude            feature_ARIMA  7.734279e-01  2.291279e+00  9.935892e-01 -8.438751e+00    16
3     fertility_rate_total            feature_ARIMA  1.700907e-01  6.587753e-02  2.015752e-01 -5.846050e+00     1
4                      gdp            feature_ARIMA  6.101890e+11  5.157283e+24  6.884112e+11 -1.276569e+01    91
5               gdp_growth             feature_LSTM  3.162147e+00  3.541902e+01  4.334448e+00 -1.304431e+00    61
6            net_migration  feature_univariate_LSTM  2.165471e+05  3.496900e+11  2.493452e+05 -1.721778e+02    81
7        population_growth            feature_ARIMA  4.587758e-01  1.400313e+00  5.499929e-01 -5.940815e+01    11
8  rural_population_growth            feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00 -3.891043e+01    21
9         urban_population            feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01 -1.413793e+00    12
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10 -2.519851e+27            feature_ARIMA   4.0
2  9.766180e+10  6.576084e+23  1.145956e+11 -4.791392e+29             feature_LSTM  14.0
6  1.030962e+11  1.579665e+24  1.131875e+11 -1.449619e+29           feature_GRU_NN  14.0
7  1.031156e+11  7.893257e+23  1.200436e+11 -2.887200e+29  feature_univariate_LSTM  16.0
9  1.160726e+11  9.040141e+23  1.361720e+11 -7.451274e+28   feature_univariate_RNN  16.0
3  1.210523e+11  1.557443e+24  1.402378e+11 -4.688677e+29              feature_GRU  23.0
8  1.578723e+11  1.890913e+24  1.843857e+11 -1.982959e+29   feature_univariate_GRU  27.0
4  1.473244e+11  3.344544e+24  1.547663e+11 -1.077378e+30          feature_LSTM_NN  30.0
5  5.348781e+11  1.566681e+25  6.588190e+11 -3.990882e+30           feature_RNN_NN  36.0
1  7.354832e+11  5.522964e+25  9.622243e+11 -3.410733e+31              feature_RNN  40.0
```


