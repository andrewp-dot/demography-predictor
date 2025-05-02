
# CompareWithStatisticalModels

**Description:** Compares BaseRNN with statistical models and BaseRNN for single feature prediction.

## Per target metrics - model comparison
```
                 target           mae           mse          rmse          mape            r2                  state  model     rank
3364  agricultural_land  5.201859e-07  2.705934e-13  5.201859e-07  2.543131e-08  0.000000e+00    Antigua and Barbuda  ARIMA    712.5
4106  agricultural_land  1.177376e-06  1.386214e-12  1.177376e-06  2.804924e-08 -2.745679e+16               Kiribati  ARIMA   4976.0
4316  agricultural_land  4.087176e-07  1.670500e-13  4.087176e-07  5.722046e-08  0.000000e+00  Micronesia, Fed. Sts.  ARIMA    709.5
4288  agricultural_land  1.186795e-05  1.408482e-10  1.186795e-05  3.051758e-07 -2.789785e+18       Marshall Islands  ARIMA   4985.0
3392  agricultural_land  4.829831e-04  2.332726e-07  4.829831e-04  4.346848e-05  0.000000e+00                  Aruba  ARIMA    732.0
...                 ...           ...           ...           ...           ...           ...                    ...    ...      ...
1391   urban_population  7.062120e+00  5.036871e+01  7.097092e+00  3.784036e-01 -2.020487e+03              St. Lucia   LSTM  17181.0
1377   urban_population  7.182285e+00  5.441989e+01  7.376984e+00  3.897081e-01 -1.169892e+03              Sri Lanka   LSTM  17159.0
215    urban_population  4.989698e+00  2.646079e+01  5.144005e+00  4.053495e-01 -1.907209e+01                Burundi   LSTM  15292.0
1265   urban_population  8.080569e+00  7.195930e+01  8.482883e+00  4.708102e-01 -1.709790e+03                 Rwanda   LSTM  17309.0
1181   urban_population  6.331862e+00  4.311766e+01  6.566404e+00  4.818958e-01 -1.961939e+03       Papua New Guinea   LSTM  17066.0

[4977 rows x 9 columns]
```


## Average metrics per model across all targets
```
           model           mae           mse          rmse      mape            r2
0          ARIMA  8.716986e+10  7.367547e+23  9.834445e+10  0.251148 -3.658308e+27
1           LSTM  1.202093e+11  1.291893e+24  1.266542e+11  0.729767 -1.847154e+29
2  ensemble_LSTM  2.981487e+11  1.690953e+25  3.548160e+11  2.723154 -2.390813e+29
```


## Best models for predicting each target:
```
                 target           mae           mse          rmse          mape        r2                state          model     rank
0     agricultural_land  5.201859e-07  2.705934e-13  5.201859e-07  2.543131e-08  0.000000  Antigua and Barbuda          ARIMA    712.5
1           arable_land  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000             Maldives          ARIMA      4.0
2      death_rate_crude  3.576453e-02  1.630092e-03  4.037440e-02  5.039838e-03 -0.164352                China          ARIMA   1156.0
3  fertility_rate_total  1.217446e-03  1.911406e-06  1.382536e-03  8.370415e-04  0.998079            St. Lucia          ARIMA     79.0
4                   gdp  1.025952e+09  1.593841e+18  1.262474e+09  2.512351e-02  0.931053         Turkmenistan           LSTM  13160.0
5     population_growth  4.037528e-02  2.444080e-03  4.943763e-02  1.065121e-02  0.023227                Niger  ensemble_LSTM    968.0
6      urban_population  9.995361e-04  1.527521e-06  1.235929e-03  1.019146e-05  0.999498              Belgium          ARIMA     71.0
```


## Overall metrics - model comparision
```
            mae           mse          rmse      mape            r2          model  rank
2  1.013597e+11  8.655177e+23  1.143846e+11  0.245527 -3.599788e+27          ARIMA   4.0
0  1.404839e+11  1.518382e+24  1.479879e+11  0.612616 -1.714926e+29           LSTM   8.0
1  3.464494e+11  1.986636e+25  4.128866e+11  2.092161 -2.705629e+29  ensemble_LSTM  12.0
```


