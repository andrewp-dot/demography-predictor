
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse        r2                                       state          model    rank
202  1.510947e+10  3.059984e+21  1.749285e+10 -0.038884  Sub-Saharan Africa (excluding high income)  feature_ARIMA  3464.0
46   1.098309e+09  2.026609e+19  1.423599e+09 -0.323639                               Cote d'Ivoire  feature_ARIMA  1591.0
209  1.430135e+09  3.175031e+19  1.781874e+09 -0.325052                                    Tanzania  feature_ARIMA  1752.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse            r2                  state          model    rank
151  3.099315e+08  1.165523e+18  3.414012e+08 -7.994873e+27             Mozambique  feature_ARIMA  3049.0
139  2.757708e+08  8.585910e+17  2.930179e+08 -3.090526e+28             Mauritania  feature_ARIMA  2991.0
182  1.449573e+07  3.008547e+15  1.734522e+07 -5.645776e+29  Sao Tome and Principe  feature_ARIMA  2323.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse          r2               state        model    rank
320  1.625365e+08  3.808153e+17  1.951450e+08 -129.736647                Guam  feature_RNN  1433.0
454  3.145447e+09  1.017830e+20  3.190346e+09 -244.369568        Turkmenistan  feature_RNN  3344.0
467  1.343679e+09  1.826541e+19  1.351498e+09 -266.626184  West Bank and Gaza  feature_RNN  2683.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse            r2              state        model    rank
313  1.967301e+13  1.449881e+28  3.807750e+13 -7.191428e+33              Gabon  feature_RNN  9307.0
264  7.152403e+12  2.174717e+27  1.474692e+13 -1.900147e+34  Brunei Darussalam  feature_RNN  9245.0
376  5.617652e+10  1.353162e+23  1.163255e+11 -5.311331e+34         Mauritania  feature_RNN  7089.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse         r2          state        model    rank
757  3.513024e+09  1.973733e+20  4.442679e+09 -13.527154  Cote d'Ivoire  feature_GRU  2753.0
920  1.142944e+10  2.389295e+21  1.545736e+10 -19.855664       Tanzania  feature_GRU  3663.0
776  1.860438e+10  6.908490e+21  2.628401e+10 -28.842084       Ethiopia  feature_GRU  4141.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse            r2              state        model    rank
787  7.802017e+10  1.538257e+23  1.240300e+11 -4.644025e+32              Gabon  feature_GRU  7179.0
738  4.777353e+10  4.668595e+22  6.832712e+10 -1.224928e+33  Brunei Darussalam  feature_GRU  6784.0
850  2.206141e+09  8.594468e+19  2.931633e+09 -2.872110e+33         Mauritania  feature_GRU  4519.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse        r2         state         model    rank
504  1.628346e+08  2.750502e+17  1.658557e+08 -4.458107       Burundi  feature_LSTM   649.0
554  2.012376e+09  4.449010e+19  2.109269e+09 -5.091162         Ghana  feature_LSTM  2089.0
503  4.211740e+08  2.848499e+18  5.337157e+08 -6.620677  Burkina Faso  feature_LSTM  1204.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse            r2              state         model    rank
550  2.001580e+10  6.610645e+21  2.571118e+10 -4.620946e+31              Gabon  feature_LSTM  6109.0
501  1.862282e+10  5.170278e+21  2.273824e+10 -2.675249e+32  Brunei Darussalam  feature_LSTM  6016.0
613  9.970079e+08  1.555249e+19  1.247097e+09 -5.152610e+32         Mauritania  feature_LSTM  3876.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse        r2          state            model    rank
994   1.788255e+09  4.813147e+19  2.193895e+09 -4.602212  Cote d'Ivoire  feature_LSTM_NN  2072.0
1134  2.289991e+08  8.902854e+17  2.983772e+08 -6.704258   Sierra Leone  feature_LSTM_NN   870.0
977   1.454174e+09  4.075721e+19  2.018844e+09 -7.640266   Burkina Faso  feature_LSTM_NN  2013.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse            r2              state            model    rank
1035  9.555088e+08  1.607712e+19  1.267957e+09 -1.500894e+31             Guyana  feature_LSTM_NN  3843.0
975   1.003125e+10  1.457457e+21  1.207252e+10 -1.658271e+32  Brunei Darussalam  feature_LSTM_NN  5520.0
1087  9.858927e+08  1.605668e+19  1.267152e+09 -8.032661e+32         Mauritania  feature_LSTM_NN  3880.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse         r2          state           model    rank
1468  2.068650e+09  6.773285e+19  2.602561e+09  -7.244901  Cote d'Ivoire  feature_GRU_NN  2228.0
1558  8.119924e+08  1.070752e+19  1.034777e+09 -11.603903           Mali  feature_GRU_NN  1624.0
1487  6.621581e+09  8.848297e+20  9.406547e+09 -14.017808       Ethiopia  feature_GRU_NN  3242.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1498  7.908607e+08  1.136237e+19  1.065945e+09 -2.827553e+31              Gabon  feature_GRU_NN  3732.0
1449  3.314406e+09  1.622700e+20  4.028275e+09 -2.027059e+32  Brunei Darussalam  feature_GRU_NN  4784.0
1561  5.736206e+08  5.674557e+18  7.532972e+08 -5.111285e+32         Mauritania  feature_GRU_NN  3568.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse         r2          state           model    rank
1231  2.633681e+09  8.431667e+19  2.903764e+09 -11.008641  Cote d'Ivoire  feature_RNN_NN  2431.0
1394  1.197221e+10  2.504032e+21  1.582414e+10 -25.010907       Tanzania  feature_RNN_NN  3718.0
1215  1.564413e+08  5.154316e+17  2.270405e+08 -32.987986        Burundi  feature_RNN_NN  1021.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse            r2              state           model    rank
1261  1.272570e+10  2.785622e+21  1.669019e+10 -4.686926e+32              Gabon  feature_RNN_NN  5754.0
1212  1.825124e+10  5.605803e+21  2.367658e+10 -1.434857e+33  Brunei Darussalam  feature_RNN_NN  6041.0
1324  3.977872e+08  2.451176e+18  4.950947e+08 -2.723815e+33         Mauritania  feature_RNN_NN  3372.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse        r2    state                    model    rank
1784  5.197008e+07  3.603991e+16  6.003642e+07 -2.900703  Liberia  feature_univariate_LSTM   269.0
1682  6.928547e+08  9.564210e+18  9.779681e+08 -3.703314  Bolivia  feature_univariate_LSTM  1457.0
1775  3.115017e+08  1.251418e+18  3.537541e+08 -5.693477  Lao PDR  feature_univariate_LSTM   958.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse            r2              state                    model    rank
1735  1.217233e+09  2.639271e+19  1.624585e+09 -6.929287e+30              Gabon  feature_univariate_LSTM  4006.0
1686  2.136344e+09  6.464518e+19  2.542542e+09 -2.812739e+31  Brunei Darussalam  feature_univariate_LSTM  4397.0
1798  1.976231e+08  4.655580e+17  2.157686e+08 -1.519111e+32         Mauritania  feature_univariate_LSTM  2961.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse        r2          state                   model    rank
2258  5.119607e+07  3.588845e+16  5.991014e+07 -2.290541        Liberia  feature_univariate_RNN   241.0
2179  3.267871e+09  1.666077e+20  4.081766e+09 -5.135147  Cote d'Ivoire  feature_univariate_RNN  2568.0
2249  3.644250e+08  1.670503e+18  4.087184e+08 -6.279349        Lao PDR  feature_univariate_RNN  1054.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse            r2              state                   model    rank
2209  8.829289e+08  1.301107e+19  1.140662e+09 -9.756659e+30              Gabon  feature_univariate_RNN  3767.0
2160  1.678955e+09  3.938364e+19  1.984531e+09 -3.827300e+31  Brunei Darussalam  feature_univariate_RNN  4204.0
2272  1.681169e+08  3.184761e+17  1.784598e+08 -2.028220e+32         Mauritania  feature_univariate_RNN  2897.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse        r2          state                   model    rank
2021  5.220673e+07  4.085382e+16  6.392016e+07 -3.041453        Liberia  feature_univariate_GRU   287.0
1919  6.435270e+08  5.242357e+18  7.240415e+08 -4.512048        Bolivia  feature_univariate_GRU  1335.0
1942  1.609116e+09  3.978697e+19  1.994672e+09 -4.567699  Cote d'Ivoire  feature_univariate_GRU  1988.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse            r2              state                   model    rank
1972  6.776591e+08  7.388216e+18  8.595485e+08 -5.160603e+30              Gabon  feature_univariate_GRU  3600.0
1923  1.354744e+09  2.521925e+19  1.588057e+09 -1.991933e+31  Brunei Darussalam  feature_univariate_GRU  4032.0
2035  1.602974e+08  2.900064e+17  1.702966e+08 -1.085088e+32         Mauritania  feature_univariate_GRU  2876.0
```


## Per target metrics - model comparision
```
               target                    model        mae         mse       rmse            r2  rank
0   agricultural_land            feature_ARIMA   1.046899    4.060992   1.190379 -5.786358e+25    18
1   agricultural_land   feature_univariate_GRU   3.393106   20.308126   3.567232 -9.381521e+27    55
2   agricultural_land  feature_univariate_LSTM   3.491063   21.461109   3.667631 -9.773747e+27    58
3   agricultural_land   feature_univariate_RNN   3.633653   23.255614   3.819408 -1.127425e+28    59
4   agricultural_land          feature_LSTM_NN   3.708523   22.722816   3.896955 -3.208033e+28    61
..                ...                      ...        ...         ...        ...           ...   ...
95   urban_population   feature_univariate_RNN   3.332837   17.013373   3.430556 -4.990889e+02    53
96   urban_population              feature_GRU   6.406513   69.546045   6.934130 -6.128359e+02    70
97   urban_population           feature_GRU_NN   6.777501   69.889947   7.157733 -6.344390e+02    71
98   urban_population           feature_RNN_NN   7.384509   96.876625   7.860708 -1.232285e+03    73
99   urban_population              feature_RNN  15.456858  446.518918  18.703927 -5.280989e+03    78

[100 rows x 7 columns]
```


## Best model for each target
```
                    target                   model           mae           mse          rmse            r2  rank
0        agricultural_land           feature_ARIMA  1.046899e+00  4.060992e+00  1.190379e+00 -5.786358e+25    18
1              arable_land           feature_ARIMA  6.680484e-01  2.092435e+00  7.559976e-01 -2.555029e+28    11
2         death_rate_crude           feature_ARIMA  7.734279e-01  2.291279e+00  9.935892e-01 -8.438751e+00    12
3     fertility_rate_total           feature_ARIMA  1.700907e-01  6.587753e-02  2.015752e-01 -5.846050e+00     1
4                      gdp           feature_ARIMA  6.101890e+11  5.157283e+24  6.884112e+11 -1.276569e+01    90
5               gdp_growth  feature_univariate_GRU  3.261534e+00  3.626577e+01  4.465543e+00 -1.615226e+00    49
6            net_migration  feature_univariate_GRU  2.386121e+05  4.735251e+11  2.675997e+05 -1.742540e+01    81
7        population_growth           feature_ARIMA  4.587758e-01  1.400313e+00  5.499929e-01 -5.940815e+01     7
8  rural_population_growth           feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00 -3.891043e+01    19
9         urban_population           feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01 -1.413793e+00     9
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10 -2.519851e+27            feature_ARIMA   4.0
9  1.366736e+11  1.592216e+24  1.522403e+11 -1.216289e+30   feature_univariate_RNN  10.0
7  1.555296e+11  2.095904e+24  1.723424e+11 -9.063098e+29  feature_univariate_LSTM  12.0
8  1.574888e+11  2.156248e+24  1.838996e+11 -6.493390e+29   feature_univariate_GRU  14.0
2  1.798632e+11  3.258726e+24  2.114750e+11 -3.989398e+30             feature_LSTM  21.0
5  2.057256e+11  5.660935e+24  2.201515e+11 -2.168513e+31           feature_RNN_NN  26.0
4  3.054954e+11  1.445779e+25  3.636970e+11 -4.805666e+30          feature_LSTM_NN  28.0
6  4.236976e+11  1.834673e+25  5.420619e+11 -3.542305e+30           feature_GRU_NN  29.0
3  2.933885e+12  7.954463e+27  5.231172e+12 -2.175410e+31              feature_GRU  36.0
1  3.584306e+13  2.975913e+29  6.177619e+13 -3.802281e+32              feature_RNN  40.0
```


