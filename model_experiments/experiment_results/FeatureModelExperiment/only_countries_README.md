
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse        r2                                       state          model    rank
202  1.510947e+10  3.059984e+21  1.749285e+10 -0.038884  Sub-Saharan Africa (excluding high income)  feature_ARIMA  4108.0
46   1.098309e+09  2.026609e+19  1.423599e+09 -0.323639                               Cote d'Ivoire  feature_ARIMA  2015.0
209  1.430135e+09  3.175031e+19  1.781874e+09 -0.325052                                    Tanzania  feature_ARIMA  2216.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
151  3.099315e+08  1.165523e+18  3.414012e+08 -7.994873e+27             Mozambique  feature_ARIMA  3253.0
139  2.757708e+08  8.585910e+17  2.930179e+08 -3.090526e+28             Mauritania  feature_ARIMA  3167.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  2349.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse         r2          state        model    rank
283  3.306309e+09  2.016217e+20  4.490236e+09 -19.323851  Cote d'Ivoire  feature_RNN  3517.0
446  1.253729e+10  3.101809e+21  1.761196e+10 -28.774064       Tanzania  feature_RNN  4637.0
461  2.982379e+09  1.815695e+20  4.261098e+09 -31.052800        Uruguay  feature_RNN  3583.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
313  1.210356e+11  3.975905e+23  1.993992e+11 -6.253019e+32              Gabon  feature_RNN  8165.0
264  7.285150e+10  1.223813e+23  1.106261e+11 -1.651643e+33  Brunei Darussalam  feature_RNN  7808.0
376  2.361666e+09  1.089303e+20  3.300459e+09 -3.775681e+33         Mauritania  feature_RNN  5144.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse        r2          state        model    rank
520  3.519548e+09  1.861951e+20  4.315040e+09 -3.151086  Cote d'Ivoire  feature_GRU  3142.0
683  2.089791e+09  8.132937e+19  2.851841e+09 -4.089855       Tanzania  feature_GRU  2788.0
575  1.583818e+10  3.464996e+21  1.861450e+10 -5.559199      Indonesia  feature_GRU  4351.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2              state        model    rank
550  5.263805e+08  3.098248e+18  5.566199e+08 -1.800787e+30              Gabon  feature_GRU  3648.0
501  3.788446e+08  1.541307e+18  3.925949e+08 -6.311230e+30  Brunei Darussalam  feature_GRU  3432.0
613  2.015393e+08  5.830517e+17  2.414654e+08 -1.975665e+31         Mauritania  feature_GRU  3131.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2         state         model    rank
740  6.374853e+08  7.244107e+18  8.511257e+08 -2.857260  Burkina Faso  feature_LSTM  1660.0
741  9.643958e+07  2.165160e+17  1.471541e+08 -2.943349       Burundi  feature_LSTM   608.0
814  6.827884e+09  6.256351e+20  7.909737e+09 -3.132532          Iraq  feature_LSTM  3585.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
849  3.576515e+07  1.382096e+16  3.717687e+07 -1.271426e+30   Marshall Islands  feature_LSTM  2472.0
738  2.168494e+09  6.562312e+19  2.561701e+09 -1.444146e+31  Brunei Darussalam  feature_LSTM  4940.0
850  1.974445e+08  4.860233e+17  2.204600e+08 -7.214245e+31         Mauritania  feature_LSTM  3103.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2                state            model    rank
1041  3.663201e+09  1.530568e+20  3.912249e+09 -3.906447              Hungary  feature_LSTM_NN  3127.0
994   3.798877e+09  1.892859e+20  4.350708e+09 -4.402274        Cote d'Ivoire  feature_LSTM_NN  3208.0
1162  5.520686e+08  5.586924e+18  7.474584e+08 -4.654175  Trinidad and Tobago  feature_LSTM_NN  1638.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2              state            model    rank
1024  7.939561e+08  9.522710e+18  9.758476e+08 -1.152798e+30              Gabon  feature_LSTM_NN  4030.0
975   7.097977e+08  6.348600e+18  7.967811e+08 -2.947969e+31  Brunei Darussalam  feature_LSTM_NN  3927.0
1087  1.633698e+08  3.211177e+17  1.791982e+08 -1.795135e+32         Mauritania  feature_LSTM_NN  2999.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse        r2    state           model    rank
1243  2.827532e+09  1.717436e+20  4.144201e+09 -3.083544  Ecuador  feature_GRU_NN  3056.0
1265  8.285347e+08  1.168471e+19  1.080959e+09 -3.199194    Ghana  feature_GRU_NN  1876.0
1223  4.112090e+08  2.394456e+18  4.893345e+08 -3.538476     Chad  feature_GRU_NN  1341.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1249  1.524428e+09  3.507314e+19  1.872783e+09 -9.348039e+29           Eswatini  feature_GRU_NN  4589.0
1212  1.000369e+09  1.370716e+19  1.170776e+09 -3.827926e+30  Brunei Darussalam  feature_GRU_NN  4194.0
1324  3.613487e+08  2.074467e+18  4.554639e+08 -2.516101e+31         Mauritania  feature_GRU_NN  3496.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse         r2          state           model    rank
1468  4.240716e+08  2.687739e+18  5.184420e+08  -2.121808  Cote d'Ivoire  feature_RNN_NN  1309.0
1631  1.097117e+10  1.900188e+21  1.378474e+10 -20.862859       Tanzania  feature_RNN_NN  4383.0
1625  8.305471e+09  9.607864e+20  9.801988e+09 -23.750857          Sudan  feature_RNN_NN  4179.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1498  3.683045e+10  2.569856e+22  5.069440e+10 -1.095471e+32              Gabon  feature_RNN_NN  7193.0
1561  1.210120e+09  2.215141e+19  1.488336e+09 -1.729317e+32         Mauritania  feature_RNN_NN  4438.0
1449  3.115818e+10  1.691000e+22  4.112177e+10 -3.590930e+32  Brunei Darussalam  feature_RNN_NN  7064.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse        r2         state                    model    rank
1795  3.225524e+08  1.490739e+18  3.861051e+08 -1.526074          Mali  feature_univariate_LSTM  1075.0
1762  5.602022e+09  4.801808e+20  6.929535e+09 -1.544410          Iraq  feature_univariate_LSTM  3396.0
1688  2.445384e+08  7.022662e+17  2.650054e+08 -1.925941  Burkina Faso  feature_univariate_LSTM   888.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse            r2              state                    model    rank
1735  7.519602e+08  9.282638e+18  9.634657e+08 -2.300467e+30              Gabon  feature_univariate_LSTM  4015.0
1686  1.416268e+09  2.741728e+19  1.655817e+09 -8.638407e+30  Brunei Darussalam  feature_univariate_LSTM  4504.0
1798  2.144414e+08  5.785593e+17  2.405351e+08 -4.809778e+31         Mauritania  feature_univariate_LSTM  3143.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse        r2          state                   model    rank
1942  3.042992e+09  1.383314e+20  3.719310e+09 -1.981386  Cote d'Ivoire  feature_univariate_RNN  2966.0
1999  7.225204e+09  6.924873e+20  8.321609e+09 -2.431730           Iraq  feature_univariate_RNN  3600.0
1976  1.611842e+09  2.969231e+19  1.723147e+09 -3.542702          Ghana  feature_univariate_RNN  2352.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse            r2              state                   model    rank
1972  1.054875e+09  1.929680e+19  1.389131e+09 -8.821015e+29              Gabon  feature_univariate_RNN  4302.0
1923  1.823003e+09  4.587791e+19  2.141913e+09 -3.059465e+30  Brunei Darussalam  feature_univariate_RNN  4773.0
2035  2.597221e+08  9.148901e+17  3.024749e+08 -1.692901e+31         Mauritania  feature_univariate_RNN  3252.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse        r2          state                   model    rank
2179  3.142243e+09  1.466270e+20  3.829199e+09 -1.805110  Cote d'Ivoire  feature_univariate_GRU  2984.0
2236  7.120027e+09  6.699675e+20  8.185181e+09 -2.284534           Iraq  feature_univariate_GRU  3583.0
2258  6.180072e+07  6.142919e+16  7.837994e+07 -2.523053        Liberia  feature_univariate_GRU   365.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse            r2              state                   model    rank
2209  1.081661e+09  2.027204e+19  1.423800e+09 -2.647132e+30              Gabon  feature_univariate_GRU  4351.0
2160  1.864619e+09  4.792453e+19  2.189167e+09 -9.094486e+30  Brunei Darussalam  feature_univariate_GRU  4799.0
2272  2.540770e+08  8.533068e+17  2.921164e+08 -4.919182e+31         Mauritania  feature_univariate_GRU  3237.0
```


## Per target metrics - model comparision
```
               target                    model       mae        mse      rmse            r2  rank
0   agricultural_land            feature_ARIMA  1.046899   4.060992  1.190379 -5.786358e+25    21
1   agricultural_land          feature_LSTM_NN  3.288009  17.902717  3.508237 -2.184442e+27    65
2   agricultural_land             feature_LSTM  2.908931  15.915447  3.100565 -3.036630e+27    58
3   agricultural_land           feature_GRU_NN  2.318851  11.019361  2.495642 -4.253956e+27    53
4   agricultural_land  feature_univariate_LSTM  2.246056  10.610181  2.380317 -5.177463e+27    51
..                ...                      ...       ...        ...       ...           ...   ...
95   urban_population   feature_univariate_RNN  2.736679  12.706353  2.825675 -2.399436e+02    57
96   urban_population          feature_LSTM_NN  4.098945  27.066645  4.297184 -2.777868e+02    74
97   urban_population           feature_RNN_NN  3.969581  30.139796  4.202946 -4.559104e+02    73
98   urban_population              feature_RNN  5.376233  54.390159  5.779008 -5.405634e+02    75
99   urban_population              feature_GRU  7.163137  76.426302  7.576305 -7.467364e+02    78

[100 rows x 7 columns]
```


## Best model for each target
```
                    target                   model           mae           mse          rmse            r2  rank
0        agricultural_land           feature_ARIMA  1.046899e+00  4.060992e+00  1.190379e+00 -5.786358e+25    21
1              arable_land           feature_ARIMA  6.680484e-01  2.092435e+00  7.559976e-01 -2.555029e+28    14
2         death_rate_crude           feature_ARIMA  7.734279e-01  2.291279e+00  9.935892e-01 -8.438751e+00    18
3     fertility_rate_total           feature_ARIMA  1.700907e-01  6.587753e-02  2.015752e-01 -5.846050e+00     1
4                      gdp           feature_ARIMA  6.101890e+11  5.157283e+24  6.884112e+11 -1.276569e+01    91
5               gdp_growth            feature_LSTM  3.173964e+00  3.537296e+01  4.330618e+00 -1.360806e+00    60
6            net_migration  feature_univariate_RNN  2.198793e+05  3.436483e+11  2.533170e+05 -2.774293e+02    81
7        population_growth           feature_ARIMA  4.587758e-01  1.400313e+00  5.499929e-01 -5.940815e+01     9
8  rural_population_growth           feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00 -3.891043e+01    22
9         urban_population           feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01 -1.413793e+00    11
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10 -2.519851e+27            feature_ARIMA   4.0
5  1.030962e+11  1.579665e+24  1.131875e+11 -1.449619e+29           feature_GRU_NN  13.0
3  9.784534e+10  6.769651e+23  1.133461e+11 -4.272807e+29             feature_LSTM  14.0
9  1.250859e+11  1.216825e+24  1.475544e+11 -2.997134e+29   feature_univariate_GRU  17.0
2  1.358619e+11  2.087168e+24  1.485012e+11 -1.414586e+29              feature_GRU  19.0
8  1.445288e+11  1.656324e+24  1.687135e+11 -1.045926e+29   feature_univariate_RNN  20.0
4  1.564333e+11  3.707279e+24  1.638777e+11 -1.018948e+30          feature_LSTM_NN  29.0
6  2.144452e+11  3.180909e+24  2.550508e+11 -2.835162e+30           feature_RNN_NN  32.0
7  2.197524e+11  7.043322e+24  2.584771e+11 -2.887537e+29  feature_univariate_LSTM  32.0
1  1.064526e+12  1.229740e+26  1.401499e+12 -2.876534e+31              feature_RNN  40.0
```


