
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse        r2                                       state          model    rank
202  1.510947e+10  3.059984e+21  1.749285e+10 -0.038884  Sub-Saharan Africa (excluding high income)  feature_ARIMA  4583.0
46   1.098309e+09  2.026609e+19  1.423599e+09 -0.323639                               Cote d'Ivoire  feature_ARIMA  2354.0
209  1.430135e+09  3.175031e+19  1.781874e+09 -0.325052                                    Tanzania  feature_ARIMA  2592.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
151  3.099315e+08  1.165523e+18  3.414012e+08 -7.994873e+27             Mozambique  feature_ARIMA  3434.0
139  2.757708e+08  8.585910e+17  2.930179e+08 -3.090526e+28             Mauritania  feature_ARIMA  3356.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  2408.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse        r2          state        model    rank
283  1.794117e+09  4.211038e+19  2.052087e+09 -3.081396  Cote d'Ivoire  feature_RNN  2894.0
378  1.173956e+10  2.228244e+21  1.492731e+10 -4.198862         Mexico  feature_RNN  4625.0
266  1.028183e+09  1.827701e+19  1.351927e+09 -4.486312   Burkina Faso  feature_RNN  2434.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
313  4.528198e+08  2.947142e+18  5.428770e+08 -6.792936e+30              Gabon  feature_RNN  3854.0
264  6.607323e+08  5.623598e+18  7.499066e+08 -2.131426e+31  Brunei Darussalam  feature_RNN  4135.0
376  1.784907e+08  3.962475e+17  1.990604e+08 -1.380864e+32         Mauritania  feature_RNN  3205.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse        r2          state        model    rank
920  4.348150e+09  3.541204e+20  5.950812e+09 -4.768019       Tanzania  feature_GRU  3863.0
757  5.383721e+09  4.774143e+20  6.909524e+09 -5.489099  Cote d'Ivoire  feature_GRU  4016.0
847  3.607667e+08  2.214957e+18  4.706381e+08 -8.249667           Mali  feature_GRU  1621.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2                  state        model    rank
893  1.997313e+07  5.812537e+15  2.410928e+07 -2.101302e+30  Sao Tome and Principe  feature_GRU  2484.0
849  1.750683e+07  3.695557e+15  1.922387e+07 -2.589984e+30       Marshall Islands  feature_GRU  2453.0
850  2.866210e+08  1.325962e+18  3.641384e+08 -2.273579e+32             Mauritania  feature_GRU  3545.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2                state         model    rank
503  1.745238e+08  3.360181e+17  1.833105e+08 -3.479557         Burkina Faso  feature_LSTM   914.0
554  1.873284e+09  3.868302e+19  1.966801e+09 -4.654245                Ghana  feature_LSTM  2933.0
688  4.635341e+08  3.004877e+18  5.481687e+08 -4.887264  Trinidad and Tobago  feature_LSTM  1667.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
560  3.102220e+07  1.153093e+16  3.395727e+07 -3.763346e+29      Guinea-Bissau  feature_LSTM  2534.0
501  6.332528e+08  5.058577e+18  7.112368e+08 -7.207638e+29  Brunei Darussalam  feature_LSTM  4066.0
613  1.284151e+08  2.066115e+17  1.437405e+08 -3.662353e+31         Mauritania  feature_LSTM  3018.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2          state            model    rank
994   5.567251e+08  4.516245e+18  6.720382e+08 -1.477690  Cote d'Ivoire  feature_LSTM_NN  1714.0
1158  1.932191e+10  3.906200e+21  1.976411e+10 -2.030763       Thailand  feature_LSTM_NN  4765.0
977   1.925008e+08  5.946755e+17  2.438622e+08 -3.499639   Burkina Faso  feature_LSTM_NN  1047.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2               state            model    rank
1127  1.200267e+11  1.468326e+23  1.211770e+11 -2.076994e+30  Russian Federation  feature_LSTM_NN  8352.0
1087  1.620517e+08  2.992227e+17  1.729812e+08 -6.614748e+31          Mauritania  feature_LSTM_NN  3114.0
975   1.500263e+09  3.030242e+19  1.740759e+09 -1.006069e+32   Brunei Darussalam  feature_LSTM_NN  4955.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse        r2                       state           model    rank
1631  2.473565e+09  1.082628e+20  3.290343e+09 -2.854001                    Tanzania  feature_GRU_NN  3327.0
1424  2.159550e+10  6.775977e+21  2.603073e+10 -3.492125  Africa Western and Central  feature_GRU_NN  5035.0
1451  1.134290e+09  2.168388e+19  1.472547e+09 -3.820961                Burkina Faso  feature_GRU_NN  2522.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1498  2.884113e+09  1.199640e+20  3.463602e+09 -1.412129e+32              Gabon  feature_GRU_NN  5676.0
1449  4.407133e+09  2.632361e+20  5.130655e+09 -4.021798e+32  Brunei Darussalam  feature_GRU_NN  6010.0
1561  3.608933e+08  2.118502e+18  4.602724e+08 -1.066673e+33         Mauritania  feature_GRU_NN  3743.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse        r2          state           model    rank
1326  3.975786e+10  2.631395e+22  5.129714e+10 -4.657189         Mexico  feature_RNN_NN  5567.0
1231  3.683694e+09  2.011177e+20  4.484622e+09 -5.489178  Cote d'Ivoire  feature_RNN_NN  3677.0
1243  2.180811e+09  9.588945e+19  3.096607e+09 -5.636094        Ecuador  feature_RNN_NN  3334.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1249  4.645085e+08  3.171544e+18  5.631650e+08 -1.497395e+30           Eswatini  feature_RNN_NN  3873.0
1212  1.559887e+09  2.936682e+19  1.713675e+09 -2.043197e+30  Brunei Darussalam  feature_RNN_NN  4937.0
1324  3.396802e+08  1.908460e+18  4.368598e+08 -3.207186e+30         Mauritania  feature_RNN_NN  3652.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse        r2    state                    model    rank
1762  8.254992e+09  9.024192e+20  9.499602e+09 -1.493417     Iraq  feature_univariate_LSTM  4192.0
1689  7.923278e+07  1.342865e+17  1.158913e+08 -1.733972  Burundi  feature_univariate_LSTM   591.0
1697  6.094714e+08  5.505164e+18  7.419706e+08 -4.396299     Chad  feature_univariate_LSTM  1885.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse            r2                  state                    model    rank
1686  2.133645e+09  6.222107e+19  2.494415e+09 -4.766801e+29      Brunei Darussalam  feature_univariate_LSTM  5344.0
1841  7.617790e+06  7.280285e+14  8.533664e+06 -6.130634e+29  Sao Tome and Principe  feature_univariate_LSTM  2370.0
1798  2.992901e+08  1.266708e+18  3.559117e+08 -2.783994e+30             Mauritania  feature_univariate_LSTM  3532.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse        r2         state                   model    rank
2236  5.578235e+09  4.852068e+20  6.965706e+09 -1.209131          Iraq  feature_univariate_RNN  3891.0
2163  8.616327e+07  1.672548e+17  1.293363e+08 -1.963720       Burundi  feature_univariate_RNN   633.0
2162  3.479287e+08  1.809210e+18  4.253505e+08 -3.069746  Burkina Faso  feature_univariate_RNN  1388.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse            r2                  state                   model    rank
2160  1.718108e+09  4.060435e+19  2.015052e+09 -1.022360e+29      Brunei Darussalam  feature_univariate_RNN  5055.0
2272  2.557340e+08  8.741598e+17  2.956674e+08 -4.255429e+29             Mauritania  feature_univariate_RNN  3387.0
2315  2.180053e+07  5.815688e+15  2.411704e+07 -5.127593e+29  Sao Tome and Principe  feature_univariate_RNN  2467.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse        r2    state                   model    rank
1999  5.541877e+09  4.779957e+20  6.913751e+09 -1.657638     Iraq  feature_univariate_GRU  3905.0
1926  9.235680e+07  1.902573e+17  1.379430e+08 -2.340378  Burundi  feature_univariate_GRU   679.0
1934  5.399343e+08  4.020357e+18  6.340656e+08 -3.732930     Chad  feature_univariate_GRU  1765.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse            r2                  state                   model    rank
1923  1.777584e+09  4.321041e+19  2.078711e+09 -5.781400e+29      Brunei Darussalam  feature_univariate_GRU  5123.0
2078  1.919124e+07  4.545436e+15  2.132149e+07 -5.864040e+29  Sao Tome and Principe  feature_univariate_GRU  2453.0
2035  2.750316e+08  1.042662e+18  3.229085e+08 -3.394989e+30             Mauritania  feature_univariate_GRU  3481.0
```


## Per target metrics - model comparision
```
               target                    model       mae        mse      rmse            r2  rank
0   agricultural_land            feature_ARIMA  1.046899   4.060992  1.190379 -5.786358e+25    21
1   agricultural_land          feature_LSTM_NN  2.655973  12.204764  2.830000 -2.508431e+27    53
2   agricultural_land  feature_univariate_LSTM  1.797426   8.038793  1.916743 -2.972407e+27    38
3   agricultural_land   feature_univariate_GRU  1.822989   8.257627  1.943556 -2.982662e+27    40
4   agricultural_land   feature_univariate_RNN  1.821476   8.139587  1.939923 -3.302503e+27    39
..                ...                      ...       ...        ...       ...           ...   ...
95   urban_population   feature_univariate_GRU  4.019774  25.414243  4.167277 -2.274484e+02    74
96   urban_population              feature_RNN  4.168032  30.153092  4.442291 -2.274808e+02    75
97   urban_population          feature_LSTM_NN  3.518788  21.184525  3.658240 -2.403437e+02    67
98   urban_population           feature_GRU_NN  5.333274  44.410994  5.658633 -4.281939e+02    78
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
5               gdp_growth             feature_LSTM  3.155872e+00  3.603032e+01  4.387556e+00 -5.588724e-01    57
6            net_migration  feature_univariate_LSTM  2.268813e+05  3.922303e+11  2.589013e+05 -3.019284e+02    81
7        population_growth            feature_ARIMA  4.587758e-01  1.400313e+00  5.499929e-01 -5.940815e+01    11
8  rural_population_growth            feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00 -3.891043e+01    22
9         urban_population            feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01 -1.413793e+00    12
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10 -2.519851e+27            feature_ARIMA   4.0
2  1.009800e+11  1.068824e+24  1.195120e+11 -1.911432e+29             feature_LSTM  12.0
1  1.374522e+11  2.402817e+24  1.434724e+11 -8.294251e+29              feature_RNN  19.0
6  1.288909e+11  2.526484e+24  1.417201e+11 -7.693239e+30           feature_GRU_NN  20.0
5  1.522702e+11  3.078286e+24  1.609114e+11 -4.117363e+28           feature_RNN_NN  22.0
9  1.648810e+11  3.140658e+24  1.923906e+11 -5.267024e+27   feature_univariate_RNN  23.0
3  1.518606e+11  3.053739e+24  1.938761e+11 -1.157820e+30              feature_GRU  27.0
4  1.731564e+11  4.071189e+24  1.804884e+11 -7.331288e+29          feature_LSTM_NN  29.0
8  1.921248e+11  5.039968e+24  2.246622e+11 -2.276765e+28   feature_univariate_GRU  31.0
7  2.712516e+11  9.331147e+24  3.218718e+11 -1.935230e+28  feature_univariate_LSTM  33.0
```


