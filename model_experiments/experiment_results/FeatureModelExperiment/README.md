
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
              mae           mse          rmse          mape            r2                  state          model    rank
142  1.637704e+06  4.258745e+13  2.063735e+06  5.483451e+13 -2.352901e+15  Micronesia, Fed. Sts.  feature_ARIMA  2242.0
138  3.271599e+06  1.479111e+14  3.845922e+06  1.305432e-01 -1.180112e+23       Marshall Islands  feature_ARIMA  2252.0
112  4.609790e+06  2.799605e+14  5.291154e+06  4.031611e-01 -2.745679e+15               Kiribati  feature_ARIMA  2267.0
```


## Model feature_ARIMA - worst states
```
              mae           mse          rmse      mape        r2         state          model    rank
162  8.983427e+11  8.918124e+24  9.443583e+11  0.210835 -7.344527  OECD members  feature_ARIMA  7055.0
90   1.100134e+12  1.347606e+25  1.160865e+12  0.361014 -6.198128   High income  feature_ARIMA  7071.0
231  1.880782e+12  4.210945e+25  2.052059e+12  0.300132 -2.772390         World  feature_ARIMA  7076.0
```


## Model feature_RNN - top states
```
              mae           mse          rmse          mape            r2                  state        model    rank
379  1.524712e+06  3.097153e+13  1.760752e+06  1.633604e+14 -1.835350e+30  Micronesia, Fed. Sts.  feature_RNN  2345.0
419  3.676164e+06  1.993925e+14  4.465429e+06  1.963960e+16 -1.779620e+29  Sao Tome and Principe  feature_RNN  2321.0
466  4.202329e+06  2.324341e+14  4.821174e+06  2.440718e+15 -1.465415e+03  Virgin Islands (U.S.)  feature_RNN  1974.0
```


## Model feature_RNN - worst states
```
              mae           mse          rmse          mape          r2         state        model    rank
468  2.659339e+12  7.608805e+25  2.758406e+12  6.173088e+14 -146.174601         World  feature_RNN  8315.0
399  2.851854e+12  8.542824e+25  2.922811e+12  6.543075e-01 -160.907347  OECD members  feature_RNN  8374.0
327  3.051521e+12  9.807109e+25  3.131631e+12  8.612802e-01 -211.834838   High income  feature_RNN  8508.0
```


## Model feature_GRU - top states
```
              mae           mse          rmse          mape            r2                  state        model    rank
703  1.208535e+07  1.933075e+15  1.390354e+07  2.708790e+15 -7.844665e+02  Virgin Islands (U.S.)  feature_GRU  1937.0
612  1.750683e+07  3.695557e+15  1.922387e+07  5.370769e-01 -2.589984e+30       Marshall Islands  feature_GRU  2490.0
656  1.997313e+07  5.812537e+15  2.410928e+07  8.778890e+16 -2.101302e+30  Sao Tome and Principe  feature_GRU  2524.0
```


## Model feature_GRU - worst states
```
              mae           mse          rmse      mape           r2                    state        model    rank
667  2.019712e+12  8.124072e+25  2.850276e+12  0.896719 -1159.981809               South Asia  feature_GRU  8908.5
668  2.019712e+12  8.124072e+25  2.850276e+12  0.896719 -1159.981809  South Asia (IDA & IBRD)  feature_GRU  8911.5
574  2.707892e+12  1.553753e+26  3.941767e+12  1.365663 -1439.350519                    India  feature_GRU  8986.0
```


## Model feature_LSTM - top states
```
              mae           mse          rmse          mape            r2                  state         model    rank
893  1.205934e+07  1.760986e+15  1.327029e+07  2.924708e+16 -4.015618e+28  Sao Tome and Principe  feature_LSTM  2368.0
901  1.600679e+07  3.204076e+15  1.790000e+07  4.169439e-01 -6.927881e+02        Solomon Islands  feature_LSTM  1937.0
773  1.796184e+07  3.966999e+15  1.991948e+07  2.978346e-01 -2.640732e+02                Eritrea  feature_LSTM  1722.0
```


## Model feature_LSTM - worst states
```
              mae           mse          rmse          mape            r2                      state         model    rank
801  1.232197e+12  2.249131e+25  1.602059e+12  24721.238288 -1.122663e+11                High income  feature_LSTM  9142.0
781  1.391358e+12  2.892502e+25  1.704600e+12   2042.335103 -3.381955e+09             European Union  feature_LSTM  9160.0
885  1.308314e+12  2.685898e+25  1.740749e+12  20268.546504 -5.693409e+11  Post-demographic dividend  feature_LSTM  9161.0
```


## Model feature_LSTM_NN - top states
```
               mae           mse          rmse          mape            r2                  state            model    rank
1138  1.621869e+07  3.056362e+15  1.748258e+07  5.215051e-01 -6.869338e+02        Solomon Islands  feature_LSTM_NN  1931.0
1025  1.443641e+07  3.079989e+15  1.755002e+07  3.597698e-01 -7.539966e+01            Gambia, The  feature_LSTM_NN  1131.0
1130  1.688532e+07  3.301421e+15  1.816989e+07  7.555418e+16 -1.725845e+28  Sao Tome and Principe  feature_LSTM_NN  2405.0
```


## Model feature_LSTM_NN - worst states
```
               mae           mse          rmse          mape            r2         state            model    rank
1110  3.164017e+12  1.035496e+26  3.221655e+12  8.697777e+02 -4.268571e+08  OECD members  feature_LSTM_NN  9249.0
1038  3.420904e+12  1.209352e+26  3.482298e+12  1.319841e+03 -2.229174e+08   High income  feature_LSTM_NN  9256.0
1179  5.096957e+12  2.810517e+26  5.301431e+12  3.945633e+19 -4.456265e+02         World  feature_LSTM_NN  8783.0
```


## Model feature_GRU_NN - top states
```
               mae           mse          rmse      mape            r2             state           model    rank
1247  1.831548e+07  6.221080e+15  2.494426e+07  0.390796 -1.349281e+03           Eritrea  feature_GRU_NN  2119.0
1297  2.359149e+07  8.068102e+15  2.840444e+07  0.958535 -1.194993e+29          Kiribati  feature_GRU_NN  2505.0
1323  2.709580e+07  8.237305e+15  2.870080e+07  0.789912 -2.447034e+31  Marshall Islands  feature_GRU_NN  2585.0
```


## Model feature_GRU_NN - worst states
```
               mae           mse          rmse       mape             r2                      state           model    rank
1359  3.200826e+12  1.085833e+26  3.295243e+12  13.672238 -104263.774739  Post-demographic dividend  feature_GRU_NN  9192.0
1347  3.429669e+12  1.243169e+26  3.525891e+12   9.062998  -33327.290661               OECD members  feature_GRU_NN  9173.0
1275  3.741964e+12  1.472719e+26  3.837638e+12  11.120573  -18206.385238                High income  feature_GRU_NN  9160.0
```


## Model feature_RNN_NN - top states
```
               mae           mse          rmse          mape            r2                  state           model    rank
1564  4.469225e+06  3.058522e+14  5.530443e+06  8.116791e+13 -6.909022e+29  Micronesia, Fed. Sts.  feature_RNN_NN  2359.0
1508  5.921184e+06  4.736073e+14  6.882016e+06  2.734552e-01 -3.469779e+29          Guinea-Bissau  feature_RNN_NN  2357.0
1603  7.844854e+06  8.549043e+14  9.246181e+06  6.563655e-01 -1.362988e+01                  Samoa  feature_RNN_NN   430.0
```


## Model feature_RNN_NN - worst states
```
               mae           mse          rmse          mape          r2         state           model    rank
1584  2.952277e+12  9.263610e+25  3.043622e+12  1.597602e+00 -915.650932  OECD members  feature_RNN_NN  8901.0
1512  3.251885e+12  1.118282e+26  3.344077e+12  1.978121e+00 -724.488324   High income  feature_RNN_NN  8861.0
1653  3.829735e+12  1.640192e+26  4.049928e+12  2.575565e+17  -28.356390         World  feature_RNN_NN  7679.0
```


## Model feature_univariate_LSTM - top states
```
               mae           mse          rmse      mape            r2             state                    model    rank
1771  2.157792e+06  7.250256e+13  2.693114e+06  4.031622 -9.063491e+25          Kiribati  feature_univariate_LSTM  2251.0
1797  3.411577e+06  1.192154e+14  3.453397e+06  0.804382 -2.461023e+29  Marshall Islands  feature_univariate_LSTM  2322.0
1745  6.350708e+06  7.665923e+14  8.757310e+06  1.392406 -2.880677e+27     Guinea-Bissau  feature_univariate_LSTM  2305.0
```


## Model feature_univariate_LSTM - worst states
```
               mae           mse          rmse          mape          r2         state                    model    rank
1821  5.735143e+12  4.524528e+26  6.726461e+12  1.751707e+00 -192.527629  OECD members  feature_univariate_LSTM  8527.0
1749  5.767793e+12  4.624302e+26  6.800223e+12  2.220509e+00 -136.031228   High income  feature_univariate_LSTM  8383.0
1890  5.934884e+12  5.272721e+26  7.261350e+12  2.707954e+16 -175.337510         World  feature_univariate_LSTM  8486.0
```


## Model feature_univariate_RNN - top states
```
               mae           mse          rmse      mape            r2             state                   model    rank
1958  2.101101e+07  6.550343e+15  2.559593e+07  0.301465 -7.556723e+02           Eritrea  feature_univariate_RNN  2027.0
2034  3.743808e+07  1.517075e+16  3.894969e+07  0.601015 -1.900103e+29  Marshall Islands  feature_univariate_RNN  2601.0
2107  3.846448e+07  2.041247e+16  4.518047e+07  0.430017 -1.159846e+02       Timor-Leste  feature_univariate_RNN  1544.0
```


## Model feature_univariate_RNN - worst states
```
               mae           mse          rmse          mape          r2         state                   model    rank
2058  2.191846e+13  8.361952e+27  2.891704e+13  1.449725e+00 -630.965877  OECD members  feature_univariate_RNN  8891.0
1986  2.274636e+13  9.068437e+27  3.011385e+13  1.768567e+00 -551.607496   High income  feature_univariate_RNN  8868.0
2127  3.077482e+13  1.740057e+28  4.171399e+13  1.538821e+16 -272.678523         World  feature_univariate_RNN  8686.0
```


## Model feature_univariate_GRU - top states
```
               mae           mse          rmse          mape            r2                  state                   model    rank
2271  5.086168e+05  4.778551e+12  6.929608e+05  1.160963e+00 -5.300571e+29       Marshall Islands  feature_univariate_GRU  2324.0
2245  3.446726e+06  1.561409e+14  3.953051e+06  4.969492e+00 -6.799863e+27               Kiribati  feature_univariate_GRU  2276.0
2275  4.852906e+06  2.777265e+14  5.271613e+06  7.898520e+13 -3.447255e+29  Micronesia, Fed. Sts.  feature_univariate_GRU  2344.0
```


## Model feature_univariate_GRU - worst states
```
               mae           mse          rmse          mape          r2         state                   model    rank
2295  4.839578e+12  3.196654e+26  5.653896e+12  1.391557e+00 -175.664514  OECD members  feature_univariate_GRU  8467.0
2223  4.846980e+12  3.245596e+26  5.697013e+12  1.780367e+00 -128.918351   High income  feature_univariate_GRU  8335.0
2364  4.765962e+12  3.434918e+26  5.860817e+12  5.331053e+16 -132.817872         World  feature_univariate_GRU  8354.0
```


## Per target metrics - model comparision
```
               target                    model       mae        mse      rmse      mape            r2  rank
0   agricultural_land            feature_ARIMA  1.046899   4.060992  1.190379  0.033024 -5.786358e+25    21
1   agricultural_land   feature_univariate_RNN  1.970232   8.234985  2.097015  0.106928 -5.181544e+27    40
2   agricultural_land  feature_univariate_LSTM  2.238162  10.102224  2.374482  0.109134 -4.262471e+27    47
3   agricultural_land   feature_univariate_GRU  2.630828  14.339557  2.793044  0.068706 -1.594090e+27    52
4   agricultural_land          feature_LSTM_NN  2.655973  12.204764  2.830000  0.141275 -2.508431e+27    53
..                ...                      ...       ...        ...       ...       ...           ...   ...
95   urban_population              feature_RNN  4.168032  30.153092  4.442291  0.075664 -2.274808e+02    67
96   urban_population             feature_LSTM  4.327347  29.631314  4.534244  0.084601 -2.118066e+02    70
97   urban_population           feature_GRU_NN  5.333274  44.410994  5.658633  0.093402 -4.281939e+02    77
98   urban_population   feature_univariate_GRU  6.181892  55.358685  6.451734  0.104901 -4.777351e+02    79
99   urban_population              feature_GRU  7.836329  93.227619  8.310895  0.134892 -9.443779e+02    80

[100 rows x 8 columns]
```


## Best model for each target
```
                    target                    model           mae           mse          rmse          mape            r2  rank
0        agricultural_land            feature_ARIMA  1.046899e+00  4.060992e+00  1.190379e+00  3.302388e-02 -5.786358e+25    21
1              arable_land            feature_ARIMA  6.680484e-01  2.092435e+00  7.559976e-01  6.360900e-02 -2.555029e+28    15
2         death_rate_crude            feature_ARIMA  7.734279e-01  2.291279e+00  9.935892e-01  1.010415e-01 -8.438751e+00    19
3     fertility_rate_total            feature_ARIMA  1.700907e-01  6.587753e-02  2.015752e-01  7.186410e-02 -5.846050e+00     1
4                      gdp            feature_ARIMA  6.101890e+11  5.157283e+24  6.884112e+11  3.326465e-01 -1.276569e+01    91
5               gdp_growth  feature_univariate_LSTM  3.302301e+00  3.602874e+01  4.456574e+00  6.603686e+12 -2.194957e+00    69
6            net_migration  feature_univariate_LSTM  2.107742e+05  2.740025e+11  2.436602e+05  4.944962e+17 -2.707323e+02    81
7        population_growth          feature_LSTM_NN  5.977667e-01  9.133658e-01  6.634935e-01  3.423307e+00 -4.285326e+01    12
8  rural_population_growth            feature_ARIMA  1.065452e+00  7.400066e+01  1.446645e+00  4.267791e+00 -3.891043e+01    27
9         urban_population            feature_ARIMA  5.459063e-01  1.149808e+00  6.724867e-01  1.144749e-02 -1.413793e+00    13
```


## Overall metrics - model comparision
```
            mae           mse          rmse          mape            r2                    model  rank
0  7.095722e+10  6.058625e+23  8.007620e+10  8.656713e+15 -2.519851e+27            feature_ARIMA   4.0
3  1.009800e+11  1.068824e+24  1.195120e+11  7.565839e+15 -1.911432e+29             feature_LSTM  10.0
1  1.374522e+11  2.402817e+24  1.434724e+11  8.622521e+14 -8.294251e+29              feature_RNN  19.0
6  1.522702e+11  3.078286e+24  1.609114e+11  1.553166e+16 -4.117363e+28           feature_RNN_NN  19.0
5  1.288909e+11  2.526484e+24  1.417201e+11  4.172961e+17 -7.693239e+30           feature_GRU_NN  20.0
2  1.518606e+11  3.053739e+24  1.938761e+11  2.189944e+15 -1.157820e+30              feature_GRU  26.0
4  1.731564e+11  4.071189e+24  1.804884e+11  3.078257e+17 -7.331288e+29          feature_LSTM_NN  27.0
9  1.868758e+11  7.595861e+24  2.205475e+11  2.579254e+17 -7.155754e+29   feature_univariate_GRU  30.0
7  2.194070e+11  1.089803e+25  2.602544e+11  5.175687e+16 -2.805447e+29  feature_univariate_LSTM  32.0
8  6.730130e+11  2.182800e+26  8.878138e+11  4.252991e+15 -1.007446e+29   feature_univariate_RNN  33.0
```


