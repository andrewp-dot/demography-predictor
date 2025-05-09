
# FeatureModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
            mae           mse          rmse      mape        r2    state          model  rank
0  7.751624e+09  7.316812e+20  8.553837e+09  0.597858 -2.341669  Czechia  feature_ARIMA   7.0
```


## Model feature_ARIMA - worst states
```
            mae           mse          rmse      mape        r2    state          model  rank
0  7.751624e+09  7.316812e+20  8.553837e+09  0.597858 -2.341669  Czechia  feature_ARIMA   7.0
```


## Model feature_RNN - top states
```
            mae           mse          rmse      mape         r2    state        model  rank
1  1.149383e+10  1.370088e+21  1.170508e+10  0.830119 -92.811257  Czechia  feature_RNN  17.0
```


## Model feature_RNN - worst states
```
            mae           mse          rmse      mape         r2    state        model  rank
1  1.149383e+10  1.370088e+21  1.170508e+10  0.830119 -92.811257  Czechia  feature_RNN  17.0
```


## Model feature_GRU - top states
```
            mae           mse          rmse      mape          r2    state        model  rank
2  4.316734e+10  2.176362e+22  4.665149e+10  1.819772 -214.543484  Czechia  feature_GRU  36.0
```


## Model feature_GRU - worst states
```
            mae           mse          rmse      mape          r2    state        model  rank
2  4.316734e+10  2.176362e+22  4.665149e+10  1.819772 -214.543484  Czechia  feature_GRU  36.0
```


## Model feature_LSTM - top states
```
            mae           mse          rmse      mape        r2    state         model  rank
3  2.282199e+10  5.366107e+21  2.316486e+10  1.459465 -32.98468  Czechia  feature_LSTM  28.0
```


## Model feature_LSTM - worst states
```
            mae           mse          rmse      mape        r2    state         model  rank
3  2.282199e+10  5.366107e+21  2.316486e+10  1.459465 -32.98468  Czechia  feature_LSTM  28.0
```


## Model feature_LSTM_NN - top states
```
            mae           mse          rmse      mape           r2    state            model  rank
4  9.092185e+09  8.456623e+20  9.196006e+09  4.443667 -1345.307464  Czechia  feature_LSTM_NN  19.0
```


## Model feature_LSTM_NN - worst states
```
            mae           mse          rmse      mape           r2    state            model  rank
4  9.092185e+09  8.456623e+20  9.196006e+09  4.443667 -1345.307464  Czechia  feature_LSTM_NN  19.0
```


## Model feature_GRU_NN - top states
```
            mae           mse          rmse      mape          r2    state           model  rank
5  2.076854e+10  4.537622e+21  2.130170e+10  1.885037 -629.906661  Czechia  feature_GRU_NN  29.0
```


## Model feature_GRU_NN - worst states
```
            mae           mse          rmse      mape          r2    state           model  rank
5  2.076854e+10  4.537622e+21  2.130170e+10  1.885037 -629.906661  Czechia  feature_GRU_NN  29.0
```


## Model feature_RNN_NN - top states
```
            mae           mse          rmse      mape         r2    state           model  rank
6  2.681786e+10  7.947217e+21  2.819081e+10  1.558028 -26.840521  Czechia  feature_RNN_NN  30.0
```


## Model feature_RNN_NN - worst states
```
            mae           mse          rmse      mape         r2    state           model  rank
6  2.681786e+10  7.947217e+21  2.819081e+10  1.558028 -26.840521  Czechia  feature_RNN_NN  30.0
```


## Model feature_univariate_LSTM - top states
```
            mae           mse          rmse     mape          r2    state                    model  rank
7  1.354866e+10  1.858595e+21  1.363303e+10  3.28893 -347.342336  Czechia  feature_univariate_LSTM  25.0
```


## Model feature_univariate_LSTM - worst states
```
            mae           mse          rmse     mape          r2    state                    model  rank
7  1.354866e+10  1.858595e+21  1.363303e+10  3.28893 -347.342336  Czechia  feature_univariate_LSTM  25.0
```


## Model feature_univariate_RNN - top states
```
            mae           mse          rmse      mape         r2    state                   model  rank
8  5.700103e+09  3.437629e+20  5.863130e+09  1.813095 -16.048259  Czechia  feature_univariate_RNN   5.0
```


## Model feature_univariate_RNN - worst states
```
            mae           mse          rmse      mape         r2    state                   model  rank
8  5.700103e+09  3.437629e+20  5.863130e+09  1.813095 -16.048259  Czechia  feature_univariate_RNN   5.0
```


## Model feature_univariate_GRU - top states
```
            mae           mse          rmse     mape          r2    state                   model  rank
9  1.222489e+10  1.512934e+21  1.230014e+10  2.13127 -714.647121  Czechia  feature_univariate_GRU  24.0
```


## Model feature_univariate_GRU - worst states
```
            mae           mse          rmse     mape          r2    state                   model  rank
9  1.222489e+10  1.512934e+21  1.230014e+10  2.13127 -714.647121  Czechia  feature_univariate_GRU  24.0
```


## Per target metrics - model comparision
```
               target                    model       mae        mse      rmse      mape          r2  rank
0   agricultural_land            feature_ARIMA  0.344562   0.134627  0.366916  0.007550   -3.105343    15
1   agricultural_land              feature_RNN  0.337553   0.186662  0.432044  0.007423   -4.692106    16
2   agricultural_land  feature_univariate_LSTM  0.427688   0.205692  0.453533  0.009377   -5.272407    18
3   agricultural_land   feature_univariate_RNN  0.579106   0.376891  0.613914  0.012726  -10.492975    20
4   agricultural_land             feature_LSTM  0.696764   0.618730  0.786594  0.015260  -17.867658    22
..                ...                      ...       ...        ...       ...       ...         ...   ...
95   urban_population           feature_RNN_NN  1.480661   2.331681  1.526984  0.020029  -46.993621    43
96   urban_population  feature_univariate_LSTM  1.944914   3.958204  1.989524  0.026311  -80.472795    49
97   urban_population           feature_GRU_NN  4.347316  20.041622  4.476787  0.058807 -411.522175    74
98   urban_population   feature_univariate_GRU  4.634558  22.183337  4.709919  0.062704 -455.605669    76
99   urban_population              feature_GRU  5.439012  31.502404  5.612700  0.073572 -647.422580    78

[100 rows x 8 columns]
```


## Best model for each target
```
                    target                    model           mae           mse          rmse      mape        r2  rank
0        agricultural_land            feature_ARIMA  3.445616e-01  1.346271e-01  3.669157e-01  0.007550 -3.105343    15
1              arable_land             feature_LSTM  1.779633e-01  5.628673e-02  2.372482e-01  0.005519 -7.695487    12
2         death_rate_crude            feature_ARIMA  8.231560e-01  1.824483e+00  1.350734e+00  0.065429 -0.448002    38
3     fertility_rate_total  feature_univariate_LSTM  2.421784e-02  9.711144e-04  3.116271e-02  0.014020  0.731283     1
4                      gdp   feature_univariate_RNN  5.700103e+10  3.437629e+21  5.863129e+10  0.242271 -3.058636    91
5               gdp_growth              feature_GRU  2.388704e+00  1.110274e+01  3.332078e+00  0.529068  0.069410    62
6            net_migration              feature_RNN  2.674471e+03  7.940017e+06  2.817804e+03  1.286595 -2.774829    81
7        population_growth           feature_GRU_NN  6.458631e-01  5.647356e-01  7.514889e-01  1.555254  0.082223    21
8  rural_population_growth           feature_GRU_NN  5.098630e-01  6.319399e-01  7.949465e-01  1.678706  0.101785    23
9         urban_population            feature_ARIMA  1.273572e-01  2.878002e-02  1.696467e-01  0.001720  0.407613     8
```


## Overall metrics - model comparision
```
            mae           mse          rmse      mape           r2                    model  rank
8  5.700103e+09  3.437629e+20  5.863130e+09  1.813095   -16.048259   feature_univariate_RNN   5.0
0  7.751624e+09  7.316812e+20  8.553837e+09  0.597858    -2.341669            feature_ARIMA   7.0
1  1.149383e+10  1.370088e+21  1.170508e+10  0.830119   -92.811257              feature_RNN  17.0
4  9.092185e+09  8.456623e+20  9.196006e+09  4.443667 -1345.307464          feature_LSTM_NN  19.0
9  1.222489e+10  1.512934e+21  1.230014e+10  2.131270  -714.647121   feature_univariate_GRU  24.0
7  1.354866e+10  1.858595e+21  1.363303e+10  3.288930  -347.342336  feature_univariate_LSTM  25.0
3  2.282199e+10  5.366107e+21  2.316486e+10  1.459465   -32.984680             feature_LSTM  28.0
5  2.076854e+10  4.537622e+21  2.130170e+10  1.885037  -629.906661           feature_GRU_NN  29.0
6  2.681786e+10  7.947217e+21  2.819081e+10  1.558028   -26.840521           feature_RNN_NN  30.0
2  4.316734e+10  2.176362e+22  4.665149e+10  1.819772  -214.543484              feature_GRU  36.0
```


