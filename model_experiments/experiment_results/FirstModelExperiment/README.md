
# FirstModelExperiment

**Description:** Compare models for predicting all features which are used for target predictions.

## Model feature_ARIMA - top states
```
            mae           mse          rmse            r2              state          model  rank
0  3.422121e+10  1.393545e+22  3.733022e+10 -2.165851e+00             Brazil  feature_ARIMA  28.0
1  1.318128e+10  2.880513e+21  1.697208e+10 -1.529697e+01              Qatar  feature_ARIMA  23.0
2  6.764486e+08  5.244433e+18  7.565133e+08 -1.256414e+15  Brunei Darussalam  feature_ARIMA  14.0
```


## Model feature_ARIMA - worst states
```
            mae           mse          rmse            r2              state          model  rank
0  3.422121e+10  1.393545e+22  3.733022e+10 -2.165851e+00             Brazil  feature_ARIMA  28.0
1  1.318128e+10  2.880513e+21  1.697208e+10 -1.529697e+01              Qatar  feature_ARIMA  23.0
2  6.764486e+08  5.244433e+18  7.565133e+08 -1.256414e+15  Brunei Darussalam  feature_ARIMA  14.0
```


## Model feature_RNN - top states
```
            mae           mse          rmse            r2              state        model  rank
3  7.298534e+11  1.171050e+25  1.082151e+12 -4.401149e+02             Brazil  feature_RNN  51.0
4  4.247656e+11  4.096555e+24  6.400646e+11 -4.886744e+05              Qatar  feature_RNN  51.0
5  7.995765e+10  1.492755e+23  1.221784e+11 -1.584539e+33  Brunei Darussalam  feature_RNN  50.0
```


## Model feature_RNN - worst states
```
            mae           mse          rmse            r2              state        model  rank
3  7.298534e+11  1.171050e+25  1.082151e+12 -4.401149e+02             Brazil  feature_RNN  51.0
4  4.247656e+11  4.096555e+24  6.400646e+11 -4.886744e+05              Qatar  feature_RNN  51.0
5  7.995765e+10  1.492755e+23  1.221784e+11 -1.584539e+33  Brunei Darussalam  feature_RNN  50.0
```


## Model feature_GRU - top states
```
             mae           mse          rmse            r2              state        model  rank
9   5.670414e+10  4.870926e+22  6.979202e+10 -2.074560e+01             Brazil  feature_GRU  33.0
10  9.754272e+09  9.819585e+20  9.912666e+09 -1.209692e+04              Qatar  feature_GRU  26.0
11  7.179006e+08  6.864777e+18  8.285396e+08 -6.521201e+30  Brunei Darussalam  feature_GRU  18.0
```


## Model feature_GRU - worst states
```
             mae           mse          rmse            r2              state        model  rank
9   5.670414e+10  4.870926e+22  6.979202e+10 -2.074560e+01             Brazil  feature_GRU  33.0
10  9.754272e+09  9.819585e+20  9.912666e+09 -1.209692e+04              Qatar  feature_GRU  26.0
11  7.179006e+08  6.864777e+18  8.285396e+08 -6.521201e+30  Brunei Darussalam  feature_GRU  18.0
```


## Model feature_LSTM - top states
```
            mae           mse          rmse            r2              state         model  rank
6  1.214034e+11  2.094256e+23  1.447155e+11 -5.593005e+01             Brazil  feature_LSTM  43.0
7  2.339960e+10  9.148056e+21  3.028548e+10 -1.664766e+06              Qatar  feature_LSTM  34.0
8  3.171944e+09  1.402339e+20  3.744782e+09 -1.826363e+31  Brunei Darussalam  feature_LSTM  25.0
```


## Model feature_LSTM - worst states
```
            mae           mse          rmse            r2              state         model  rank
6  1.214034e+11  2.094256e+23  1.447155e+11 -5.593005e+01             Brazil  feature_LSTM  43.0
7  2.339960e+10  9.148056e+21  3.028548e+10 -1.664766e+06              Qatar  feature_LSTM  34.0
8  3.171944e+09  1.402339e+20  3.744782e+09 -1.826363e+31  Brunei Darussalam  feature_LSTM  25.0
```


## Model feature_LSTM_NN - top states
```
             mae           mse          rmse            r2              state            model  rank
12  8.060271e+10  7.580590e+22  8.706665e+10 -1.169598e+02             Brazil  feature_LSTM_NN  39.0
13  3.775809e+09  2.454809e+20  4.955183e+09 -6.083795e+02              Qatar  feature_LSTM_NN  22.0
14  1.915148e+09  5.005119e+19  2.237212e+09 -2.507907e+31  Brunei Darussalam  feature_LSTM_NN  23.0
```


## Model feature_LSTM_NN - worst states
```
             mae           mse          rmse            r2              state            model  rank
12  8.060271e+10  7.580590e+22  8.706665e+10 -1.169598e+02             Brazil  feature_LSTM_NN  39.0
13  3.775809e+09  2.454809e+20  4.955183e+09 -6.083795e+02              Qatar  feature_LSTM_NN  22.0
14  1.915148e+09  5.005119e+19  2.237212e+09 -2.507907e+31  Brunei Darussalam  feature_LSTM_NN  23.0
```


## Per target metrics - model comparision
```
                     target            model           mae           mse          rmse            r2  rank
0         agricultural_land    feature_ARIMA  2.031367e-01  9.675176e-02  2.406081e-01 -1.976349e+00     3
1         agricultural_land     feature_LSTM  2.372628e+00  9.951517e+00  2.481974e+00 -1.196192e+02    24
2         agricultural_land  feature_LSTM_NN  2.884514e+00  1.375716e+01  2.986966e+00 -1.460224e+02    28
3         agricultural_land      feature_GRU  3.558053e+00  2.697845e+01  3.686566e+00 -7.685560e+02    35
4         agricultural_land      feature_RNN  1.236074e+01  2.027336e+02  1.373700e+01 -3.016546e+03    39
5               arable_land    feature_ARIMA  3.700554e-01  3.148747e-01  4.412126e-01 -4.185840e+15     5
6               arable_land      feature_GRU  6.457809e-01  5.829414e-01  7.487246e-01 -2.173734e+31    10
7               arable_land     feature_LSTM  1.362569e+00  1.953559e+00  1.391324e+00 -6.087876e+31    17
8               arable_land  feature_LSTM_NN  1.175998e+00  1.639504e+00  1.223290e+00 -8.359691e+31    12
9               arable_land      feature_RNN  1.373755e+01  2.227918e+02  1.489658e+01 -5.281796e+33    40
10         death_rate_crude    feature_ARIMA  5.100851e-01  4.694358e-01  6.611232e-01 -1.303216e+01     9
11         death_rate_crude      feature_GRU  1.299339e+00  1.884624e+00  1.367580e+00 -6.587978e+01    16
12         death_rate_crude      feature_RNN  1.379498e+00  3.189581e+00  1.459875e+00 -2.422411e+02    18
13         death_rate_crude  feature_LSTM_NN  2.777231e+00  1.011340e+01  2.810026e+00 -6.988641e+02    27
14         death_rate_crude     feature_LSTM  3.157122e+00  1.228380e+01  3.210762e+00 -7.976065e+02    32
15     fertility_rate_total    feature_ARIMA  1.387156e-01  2.862315e-02  1.635881e-01 -5.667263e+00     1
16     fertility_rate_total      feature_GRU  3.590189e-01  1.733542e-01  3.876702e-01 -3.428324e+01     4
17     fertility_rate_total     feature_LSTM  4.164831e-01  2.382253e-01  4.323385e-01 -5.288576e+01     6
18     fertility_rate_total      feature_RNN  4.699917e-01  3.482242e-01  4.914281e-01 -6.844854e+01     7
19     fertility_rate_total  feature_LSTM_NN  5.080913e-01  3.368151e-01  5.267004e-01 -7.596810e+01     8
20                      gdp      feature_GRU  2.239142e+11  1.656603e+23  2.684331e+11 -1.354965e+01    47
21                      gdp  feature_LSTM_NN  2.876441e+11  2.536715e+23  3.141947e+11 -2.860700e+01    48
22                      gdp    feature_ARIMA  1.601962e+11  5.607065e+22  1.834191e+11 -3.080996e+01    46
23                      gdp     feature_LSTM  4.931751e+11  7.290463e+23  5.956865e+11 -1.665028e+02    49
24                      gdp      feature_RNN  4.115218e+12  5.318776e+25  6.147907e+12 -1.163411e+05    50
25               gdp_growth      feature_GRU  3.012022e+00  1.547147e+01  3.611617e+00 -2.891722e-01    31
26               gdp_growth     feature_LSTM  2.906692e+00  1.298732e+01  3.577210e+00 -7.549513e-01    29
27               gdp_growth      feature_RNN  3.320054e+00  1.669255e+01  4.043340e+00 -1.225075e+00    34
28               gdp_growth  feature_LSTM_NN  3.230372e+00  1.601718e+01  3.983130e+00 -1.347843e+00    33
29               gdp_growth    feature_ARIMA  6.808563e+00  7.659099e+01  7.397201e+00 -4.291778e+00    37
30            net_migration  feature_LSTM_NN  1.484620e+06  1.136310e+13  2.143415e+06 -1.338988e+03    41
31            net_migration      feature_GRU  6.861575e+06  3.595197e+14  1.096303e+07 -3.787807e+04    42
32            net_migration      feature_RNN  3.736935e+07  1.500905e+16  7.075268e+07 -1.585725e+06    43
33            net_migration     feature_LSTM  7.485450e+07  5.266646e+16  1.325640e+08 -5.548051e+06    45
34            net_migration    feature_ARIMA  6.694813e+07  3.648700e+16  1.103313e+08 -2.207849e+12    44
35        population_growth    feature_ARIMA  1.079358e+00  4.493404e+00  1.304434e+00  1.507893e-01    11
36        population_growth      feature_RNN  1.602787e+00  7.761214e+00  1.844634e+00 -5.059452e+00    19
37        population_growth      feature_GRU  1.693104e+00  6.744320e+00  1.893122e+00 -1.382442e+01    21
38        population_growth     feature_LSTM  1.603919e+00  5.368821e+00  1.858141e+00 -2.783691e+01    20
39        population_growth  feature_LSTM_NN  1.765934e+00  5.434290e+00  1.987283e+00 -4.471725e+01    22
40  rural_population_growth    feature_ARIMA  1.297828e+00  5.131398e+00  1.489142e+00 -2.065703e+00    15
41  rural_population_growth      feature_GRU  1.222113e+00  3.715218e+00  1.440827e+00 -8.580719e+00    14
42  rural_population_growth      feature_RNN  1.195119e+00  3.508508e+00  1.424811e+00 -9.549555e+00    13
43  rural_population_growth  feature_LSTM_NN  2.151744e+00  1.022809e+01  2.593049e+00 -6.611398e+01    23
44  rural_population_growth     feature_LSTM  2.476977e+00  1.207368e+01  2.878768e+00 -7.230860e+01    25
45         urban_population    feature_ARIMA  1.524074e-01  7.958901e-02  2.094945e-01 -6.468133e-01     2
46         urban_population  feature_LSTM_NN  2.668237e+00  8.233494e+00  2.854006e+00 -8.110837e+01    26
47         urban_population     feature_LSTM  4.470566e+00  2.358896e+01  4.790762e+00 -2.651352e+02    36
48         urban_population      feature_RNN  2.930255e+00  2.082684e+01  3.185280e+00 -4.407642e+02    30
49         urban_population      feature_GRU  1.213468e+01  1.691185e+02  1.293149e+01 -1.698729e+03    38
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2            model  rank
0  1.755918e+10  6.274529e+21  1.998980e+10 -3.718986e+14    feature_ARIMA   4.0
3  2.510826e+10  1.902061e+22  3.021744e+10 -1.930275e+30      feature_GRU   8.0
4  3.272658e+10  2.920284e+22  3.568147e+10 -7.423405e+30  feature_LSTM_NN  13.0
2  5.504569e+10  8.338832e+22  6.637055e+10 -5.406034e+30     feature_LSTM  15.0
1  4.398562e+11  5.851914e+24  6.565312e+11 -4.690235e+32      feature_RNN  20.0
```


