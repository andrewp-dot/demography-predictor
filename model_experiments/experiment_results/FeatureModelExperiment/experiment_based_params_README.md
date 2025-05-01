
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse        r2                                       state          model    rank
202  1.510947e+10  3.059984e+21  1.749285e+10 -0.038884  Sub-Saharan Africa (excluding high income)  feature_ARIMA  4796.0
46   1.098309e+09  2.026609e+19  1.423599e+09 -0.323639                               Cote d'Ivoire  feature_ARIMA  2534.0
209  1.430135e+09  3.175031e+19  1.781874e+09 -0.325052                                    Tanzania  feature_ARIMA  2781.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
151  3.099315e+08  1.165523e+18  3.414012e+08 -7.994873e+27             Mozambique  feature_ARIMA  3657.0
139  2.757708e+08  8.585910e+17  2.930179e+08 -3.090526e+28             Mauritania  feature_ARIMA  3574.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  2440.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse        r2          state        model    rank
283  1.794117e+09  4.211038e+19  2.052087e+09 -3.081396  Cote d'Ivoire  feature_RNN  3069.0
378  1.173956e+10  2.228244e+21  1.492731e+10 -4.198862         Mexico  feature_RNN  4814.0
266  1.028183e+09  1.827701e+19  1.351927e+09 -4.486312   Burkina Faso  feature_RNN  2643.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
313  4.528198e+08  2.947142e+18  5.428770e+08 -6.792936e+30              Gabon  feature_RNN  4068.0
264  6.607323e+08  5.623598e+18  7.499066e+08 -2.131426e+31  Brunei Darussalam  feature_RNN  4325.0
376  1.784907e+08  3.962475e+17  1.990604e+08 -1.380864e+32         Mauritania  feature_RNN  3380.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse        r2          state        model    rank
920  4.348150e+09  3.541204e+20  5.950812e+09 -4.768019       Tanzania  feature_GRU  4013.0
757  5.383721e+09  4.774143e+20  6.909524e+09 -5.489099  Cote d'Ivoire  feature_GRU  4181.0
847  3.607667e+08  2.214957e+18  4.706381e+08 -8.249667           Mali  feature_GRU  1851.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2                  state        model    rank
893  1.997313e+07  5.812537e+15  2.410928e+07 -2.101302e+30  Sao Tome and Principe  feature_GRU  2524.0
849  1.750683e+07  3.695557e+15  1.922387e+07 -2.589984e+30       Marshall Islands  feature_GRU  2490.0
850  2.866210e+08  1.325962e+18  3.641384e+08 -2.273579e+32             Mauritania  feature_GRU  3779.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2                state         model    rank
503  1.745238e+08  3.360181e+17  1.833105e+08 -3.479557         Burkina Faso  feature_LSTM  1088.0
554  1.873284e+09  3.868302e+19  1.966801e+09 -4.654245                Ghana  feature_LSTM  3114.0
688  4.635341e+08  3.004877e+18  5.481687e+08 -4.887264  Trinidad and Tobago  feature_LSTM  1904.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
560  3.102220e+07  1.153093e+16  3.395727e+07 -3.763346e+29      Guinea-Bissau  feature_LSTM  2570.0
501  6.332528e+08  5.058577e+18  7.112368e+08 -7.207638e+29  Brunei Darussalam  feature_LSTM  4251.0
613  1.284151e+08  2.066115e+17  1.437405e+08 -3.662353e+31         Mauritania  feature_LSTM  3121.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2          state            model    rank
994   5.567251e+08  4.516245e+18  6.720382e+08 -1.477690  Cote d'Ivoire  feature_LSTM_NN  1902.0
1158  1.932191e+10  3.906200e+21  1.976411e+10 -2.030763       Thailand  feature_LSTM_NN  4985.0
977   1.925008e+08  5.946755e+17  2.438622e+08 -3.499639   Burkina Faso  feature_LSTM_NN  1270.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2               state            model    rank
1127  1.200267e+11  1.468326e+23  1.211770e+11 -2.076994e+30  Russian Federation  feature_LSTM_NN  8488.0
1087  1.620517e+08  2.992227e+17  1.729812e+08 -6.614748e+31          Mauritania  feature_LSTM_NN  3266.0
975   1.500263e+09  3.030242e+19  1.740759e+09 -1.006069e+32   Brunei Darussalam  feature_LSTM_NN  5149.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse        r2                       state           model    rank
1631  2.473565e+09  1.082628e+20  3.290343e+09 -2.854001                    Tanzania  feature_GRU_NN  3461.0
1424  2.159550e+10  6.775977e+21  2.603073e+10 -3.492125  Africa Western and Central  feature_GRU_NN  5251.0
1451  1.134290e+09  2.168388e+19  1.472547e+09 -3.820961                Burkina Faso  feature_GRU_NN  2723.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1498  2.884113e+09  1.199640e+20  3.463602e+09 -1.412129e+32              Gabon  feature_GRU_NN  5804.0
1449  4.407133e+09  2.632361e+20  5.130655e+09 -4.021798e+32  Brunei Darussalam  feature_GRU_NN  6145.0
1561  3.608933e+08  2.118502e+18  4.602724e+08 -1.066673e+33         Mauritania  feature_GRU_NN  3952.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse        r2          state           model    rank
1326  3.975786e+10  2.631395e+22  5.129714e+10 -4.657189         Mexico  feature_RNN_NN  5807.0
1231  3.683694e+09  2.011177e+20  4.484622e+09 -5.489178  Cote d'Ivoire  feature_RNN_NN  3852.0
1243  2.180811e+09  9.588945e+19  3.096607e+09 -5.636094        Ecuador  feature_RNN_NN  3476.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1249  4.645085e+08  3.171544e+18  5.631650e+08 -1.497395e+30           Eswatini  feature_RNN_NN  4086.0
1212  1.559887e+09  2.936682e+19  1.713675e+09 -2.043197e+30  Brunei Darussalam  feature_RNN_NN  5121.0
1324  3.396802e+08  1.908460e+18  4.368598e+08 -3.207186e+30         Mauritania  feature_RNN_NN  3866.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse        r2    state                    model    rank
1689  4.563973e+07  2.626464e+16  5.125825e+07 -2.021476  Burundi  feature_univariate_LSTM   426.0
1762  6.118157e+09  5.491478e+20  7.410478e+09 -3.601581     Iraq  feature_univariate_LSTM  4214.0
1697  3.271715e+08  1.249605e+18  3.535003e+08 -6.684496     Chad  feature_univariate_LSTM  1642.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse            r2              state                    model    rank
1735  5.293218e+08  4.124069e+18  6.421903e+08 -2.665983e+30              Gabon  feature_univariate_LSTM  4168.0
1686  1.119297e+09  1.712769e+19  1.308728e+09 -8.330614e+30  Brunei Darussalam  feature_univariate_LSTM  4852.0
1798  9.966782e+07  1.373372e+17  1.171945e+08 -4.635527e+31         Mauritania  feature_univariate_LSTM  3005.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse        r2                                   state                   model    rank
2222  1.522528e+10  2.383553e+21  1.543882e+10 -0.842258  Heavily indebted poor countries (HIPC)  feature_univariate_RNN  4752.0
2308  2.783151e+10  1.169229e+22  3.419405e+10 -1.268623                Pre-demographic dividend  feature_univariate_RNN  5340.0
2236  9.906173e+09  1.092215e+21  1.045094e+10 -1.533713                                    Iraq  feature_univariate_RNN  4456.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse            r2              state                   model    rank
2209  3.631471e+08  1.539424e+18  3.923562e+08 -1.038668e+30              Gabon  feature_univariate_RNN  3828.0
2160  7.985257e+08  8.338318e+18  9.131440e+08 -3.041539e+30  Brunei Darussalam  feature_univariate_RNN  4483.0
2272  1.178546e+08  1.548123e+17  1.244243e+08 -1.565725e+31         Mauritania  feature_univariate_RNN  3051.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse         r2    state                   model    rank
1926  5.820815e+07  4.202875e+16  6.483880e+07  -3.148590  Burundi  feature_univariate_GRU   549.0
1999  6.690477e+09  6.147116e+20  7.840381e+09 -10.870491     Iraq  feature_univariate_GRU  4450.0
1916  5.110806e+07  3.974416e+16  6.304309e+07 -11.728563   Belize  feature_univariate_GRU   746.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse            r2              state                   model    rank
1972  4.489608e+08  2.541447e+18  5.041289e+08 -6.933436e+30              Gabon  feature_univariate_GRU  4038.0
1923  9.864300e+08  1.324452e+19  1.150848e+09 -2.119153e+31  Brunei Darussalam  feature_univariate_GRU  4722.0
2035  9.320819e+07  1.618467e+17  1.272341e+08 -1.188628e+32         Mauritania  feature_univariate_GRU  3026.0
```


## Per target metrics - model comparision
```
               target                    model       mae        mse      rmse            r2  rank
0   agricultural_land            feature_ARIMA  1.046899   4.060992  1.190379 -5.786358e+25    21
1   agricultural_land   feature_univariate_GRU  2.630828  14.339557  2.793044 -1.594090e+27    52
2   agricultural_land          feature_LSTM_NN  2.655973  12.204764  2.830000 -2.508431e+27    53
3   agricultural_land  feature_univariate_LSTM  2.238162  10.102224  2.374482 -4.262471e+27    47
4   agricultural_land             feature_LSTM  3.066597  14.755432  3.264565 -4.376556e+27    56
..                ...                      ...       ...        ...       ...           ...   ...
95   urban_population              feature_RNN  4.168032  30.153092  4.442291 -2.274808e+02    74
96   urban_population          feature_LSTM_NN  3.518788  21.184525  3.658240 -2.403437e+02    66
97   urban_population           feature_GRU_NN  5.333274  44.410994  5.658633 -4.281939e+02    77
98   urban_population   feature_univariate_GRU  6.181892  55.358685  6.451734 -4.777351e+02    79
99   urban_population              feature_GRU  7.836329  93.227619  8.310895 -9.443779e+02    80

[100 rows x 7 columns]
```


## Best model for each target
```
                    target                    model           mae           mse          rmse            r2  rank
0        agricultural_land            feature_ARIMA  1.046899e+00  4.060992e+00  1.190379e+00 -5.786358e+25    21
1              arable_land            feature_ARIMA  6.680484e-01  2.092435e+00  7.559976e-01 -2.555029e+28    14
2         death_rate_crude            feature_ARIMA  7.734279e-01  2.291279e+00  9.935892e-01 -8.438751e+00    17
3     fertility_rate_total            feature_ARIMA  1.700907e-01  6.587753e-02  2.015752e-01 -5.846050e+00     1
4                      gdp            feature_ARIMA  6.101890e+11  5.157283e+24  6.884112e+11 -1.276569e+01    91
5               gdp_growth             feature_LSTM  3.155872e+00  3.603032e+01  4.387556e+00 -5.588724e-01    58
6            net_migration  feature_univariate_LSTM  2.107742e+05  2.740025e+11  2.436602e+05 -2.707323e+02    81
7        population_growth            feature_ARIMA  4.587758e-01  1.400313e+00  5.499929e-01 -5.940815e+01    11
8  rural_population_growth            feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00 -3.891043e+01    22
9         urban_population            feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01 -1.413793e+00    12
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10 -2.519851e+27            feature_ARIMA   4.0
2  1.009800e+11  1.068824e+24  1.195120e+11 -1.911432e+29             feature_LSTM  10.0
1  1.374522e+11  2.402817e+24  1.434724e+11 -8.294251e+29              feature_RNN  19.0
5  1.522702e+11  3.078286e+24  1.609114e+11 -4.117363e+28           feature_RNN_NN  19.0
6  1.288909e+11  2.526484e+24  1.417201e+11 -7.693239e+30           feature_GRU_NN  20.0
3  1.518606e+11  3.053739e+24  1.938761e+11 -1.157820e+30              feature_GRU  26.0
4  1.731564e+11  4.071189e+24  1.804884e+11 -7.331288e+29          feature_LSTM_NN  27.0
8  1.868758e+11  7.595861e+24  2.205475e+11 -7.155754e+29   feature_univariate_GRU  30.0
7  2.194070e+11  1.089803e+25  2.602544e+11 -2.805447e+29  feature_univariate_LSTM  32.0
9  6.730130e+11  2.182800e+26  8.878138e+11 -1.007446e+29   feature_univariate_RNN  33.0
```


