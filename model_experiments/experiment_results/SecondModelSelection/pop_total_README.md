
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
               mae           mse          rmse        r2                        state           model    rank
394   38868.538462  2.246997e+09  4.740250e+04  0.999730                        Niger  ensemble-arima   592.0
427     596.041667  4.464089e+05  6.681384e+02  0.999727              Solomon Islands  ensemble-arima    37.0
238  854625.384615  1.800620e+12  1.341872e+06  0.999508  Africa Eastern and Southern  ensemble-arima  2837.0
463     408.079861  2.297684e+05  4.793417e+02  0.999289                      Vanuatu  ensemble-arima    29.0
269   15990.200000  3.267200e+08  1.807540e+04  0.999012                     Cambodia  ensemble-arima   297.0
```


## Model ensemble-arima - worst states
```
              mae           mse          rmse          r2                 state           model    rank
434  2.124653e+02  6.966175e+04  2.639351e+02  -15.200594   St. Kitts and Nevis  ensemble-arima  1279.0
416  7.980142e+05  1.196820e+12  1.093993e+06  -19.329059    Russian Federation  ensemble-arima  3990.0
444  7.415794e+06  7.673303e+13  8.759739e+06  -41.410248  Syrian Arab Republic  ensemble-arima  5751.0
432  4.649685e+06  2.939709e+13  5.421908e+06 -266.124650                 Spain  ensemble-arima  5665.0
360  4.872601e+06  3.485500e+13  5.903813e+06 -435.500165               Lebanon  ensemble-arima  5742.0
```


## Model ensemble-arimax - top states
```
              mae           mse          rmse        r2                    state            model    rank
95   9.670812e+06  1.943424e+14  1.394067e+07  0.997766         IDA & IBRD total  ensemble-arimax  4558.0
146  1.102659e+07  1.711622e+14  1.308290e+07  0.996935            Middle income  ensemble-arimax  4575.0
130  8.943337e+06  1.012435e+14  1.006198e+07  0.995849      Lower middle income  ensemble-arimax  4425.0
193  5.990130e+06  4.532530e+13  6.732407e+06  0.993378               South Asia  ensemble-arimax  4197.0
194  5.990130e+06  4.532530e+13  6.732407e+06  0.993378  South Asia (IDA & IBRD)  ensemble-arimax  4197.0
```


## Model ensemble-arimax - worst states
```
              mae           mse          rmse            r2                        state            model    rank
165  4.019290e+06  3.495259e+13  5.912072e+06 -4.809188e+03  Pacific island small states  ensemble-arimax  5837.0
62   6.523108e+06  5.664554e+13  7.526323e+06 -1.007786e+04                      Eritrea  ensemble-arimax  6084.0
63   1.864880e+06  7.268571e+12  2.696029e+06 -2.160703e+05                      Estonia  ensemble-arimax  5314.0
206  3.412021e+09  3.386257e+19  5.819155e+09 -3.566389e+08                  Switzerland  ensemble-arimax  7577.0
116  1.058057e+10  4.411562e+20  2.100372e+10 -7.544312e+09                      Lao PDR  ensemble-arimax  7584.0
```


## Model simple-rnn - top states
```
              mae           mse          rmse        r2                        state       model    rank
558  2.272938e+05  7.578971e+10  2.752993e+05  0.919797                    Guatemala  simple-rnn  1750.0
485  5.747171e+05  4.485607e+11  6.697468e+05  0.744889                    Australia  simple-rnn  2611.0
503  7.719248e+05  1.137952e+12  1.066748e+06  0.722536                 Burkina Faso  simple-rnn  2955.0
631  1.747022e+06  3.889938e+12  1.972293e+06  0.531839                        Niger  simple-rnn  3590.0
639  5.206234e+04  3.718951e+09  6.098320e+04  0.488197  Pacific island small states  simple-rnn  1112.0
```


## Model simple-rnn - worst states
```
              mae           mse          rmse            r2               state       model    rank
579  2.798858e+04  8.231106e+08  2.868990e+04 -12813.750034         Isle of Man  simple-rnn  2291.0
646  1.078004e+07  1.239713e+14  1.113424e+07 -19521.415126              Poland  simple-rnn  6344.0
653  4.066490e+07  1.737464e+15  4.168289e+07 -29511.376771  Russian Federation  simple-rnn  6869.0
623  2.076987e+05  4.363813e+10  2.088974e+05 -30582.151053          Montenegro  simple-rnn  3285.0
522  2.666337e+06  7.320552e+12  2.705652e+06 -32525.471932                Cuba  simple-rnn  5407.0
```


## Model base-lstm - top states
```
               mae           mse           rmse        r2       state      model    rank
891  113481.734818  1.956584e+10  139877.955531  0.982650      Rwanda  base-lstm  1191.0
850   67714.920875  6.422162e+09   80138.394216  0.953696  Mauritania  base-lstm  1002.0
846   20238.752596  5.297154e+08   23015.547460  0.789195    Maldives  base-lstm   625.0
732  544149.214052  3.695816e+11  607932.239272  0.742265       Benin  base-lstm  2558.0
874  214406.272784  9.191700e+10  303178.160107  0.713364        Oman  base-lstm  1918.0
```


## Model base-lstm - worst states
```
              mae           mse          rmse             r2                state      model    rank
819  1.247943e+08  1.667284e+16  1.291234e+08  -37270.173779                Japan  base-lstm  7140.0
816  8.218149e+04  7.145204e+09  8.452931e+04 -111240.438130          Isle of Man  base-lstm  2808.0
898  3.154488e+06  1.003033e+13  3.167069e+06 -118052.011975      Slovak Republic  base-lstm  5552.0
883  3.074624e+07  9.717476e+14  3.117287e+07 -153025.235818               Poland  base-lstm  6787.0
908  6.306192e+04  4.181850e+09  6.466722e+04 -972533.376962  St. Kitts and Nevis  base-lstm  2669.0
```


## Model base-gru - top states
```
               mae           mse          rmse        r2                       state     model    rank
1001  6.592916e+04  7.728819e+09  8.791370e+04  0.963176          Dominican Republic  base-gru  1011.0
1128  2.859182e+05  1.114410e+11  3.338278e+05  0.901180                      Rwanda  base-gru  1936.0
1084  3.886381e+05  3.542155e+11  5.951601e+05  0.898256                        Mali  base-gru  2332.0
986   4.546292e+05  3.259613e+11  5.709302e+05  0.896537                        Chad  base-gru  2357.0
1002  4.017084e+07  2.704980e+15  5.200942e+07  0.894234  Early-demographic dividend  base-gru  5211.0
```


## Model base-gru - worst states
```
               mae           mse          rmse             r2                state     model    rank
965   3.290445e+05  1.229057e+11  3.505792e+05  -24084.495466             Barbados  base-gru  3701.0
1120  2.157318e+07  4.868600e+14  2.206490e+07  -76667.413608               Poland  base-gru  6665.0
1135  2.598493e+06  6.871498e+12  2.621354e+06  -80873.840590      Slovak Republic  base-gru  5390.0
1053  1.548027e+05  2.595337e+10  1.611005e+05 -404058.828765          Isle of Man  base-gru  3134.0
1145  4.790959e+04  2.504333e+09  5.004331e+04 -582408.654988  St. Kitts and Nevis  base-gru  2531.0
```


## Model xgboost - top states
```
               mae           mse          rmse        r2                                     state    model    rank
1274  5.368880e+06  4.680301e+13  6.841272e+06  0.992227    Heavily indebted poor countries (HIPC)  xgboost  4196.0
1186  8.435823e+06  1.125880e+14  1.061075e+07  0.969256               Africa Eastern and Southern  xgboost  4512.0
1283  1.780909e+07  5.317715e+14  2.306017e+07  0.968403                                 IDA total  xgboost  4883.0
1258  1.404690e+07  2.721760e+14  1.649776e+07  0.951780  Fragile and conflict affected situations  xgboost  4793.0
1343  3.028460e+06  1.528844e+13  3.910043e+06  0.951205                                   Nigeria  xgboost  3891.0
```


## Model xgboost - worst states
```
                mae           mse           rmse            r2                  state    model    rank
1334  111859.031250  1.261479e+10  112315.590928 -8.839893e+03             Montenegro  xgboost  2878.0
1414   63871.738281  4.286751e+09   65473.287501 -2.403673e+04  Virgin Islands (U.S.)  xgboost  2660.0
1398   84113.595703  7.384095e+09   85930.757389 -2.609444e+04                  Tonga  xgboost  2798.0
1290  100609.180176  1.127519e+10  106184.692769 -1.755389e+05            Isle of Man  xgboost  2896.0
1382  134483.739583  1.868723e+10  136701.256240 -4.345917e+06    St. Kitts and Nevis  xgboost  3033.0
```


## Model rf - top states
```
               mae           mse          rmse        r2          state model    rank
1589  1.576781e+04  3.516306e+08  1.875181e+04  0.994411         Panama    rf   331.0
1513  3.388286e+04  2.523572e+09  5.023517e+04  0.993528       Honduras    rf   636.0
1468  1.608873e+05  3.448352e+10  1.856974e+05  0.992509  Cote d'Ivoire    rf  1352.0
1580  1.790193e+06  4.675385e+12  2.162264e+06  0.985078        Nigeria    rf  3347.0
1510  5.973774e+04  4.789757e+09  6.920807e+04  0.984073          Haiti    rf   876.0
```


## Model rf - worst states
```
               mae           mse          rmse            r2               state model    rank
1594  2.054040e+06  5.722457e+12  2.392166e+06   -900.145618              Poland    rf  5120.0
1492  6.765284e+07  6.252560e+15  7.907313e+07   -954.433538      European Union    rf  6901.0
1601  8.853967e+06  1.269303e+14  1.126633e+07  -2155.024296  Russian Federation    rf  6232.0
1609  4.045856e+05  2.291959e+11  4.787440e+05  -2696.545641     Slovak Republic    rf  3857.0
1470  1.471483e+06  2.969394e+12  1.723193e+06 -13192.527040                Cuba    rf  4974.0
```


## Model lightgbm - top states
```
               mae           mse          rmse        r2                                       state     model    rank
1661  3.598709e+06  1.562109e+13  3.952352e+06  0.990751                  Africa Western and Central  lightgbm  3863.0
1861  1.001171e+07  1.472099e+14  1.213301e+07  0.985741  Sub-Saharan Africa (excluding high income)  lightgbm  4566.0
1859  1.003299e+07  1.483626e+14  1.218042e+07  0.985631                          Sub-Saharan Africa  lightgbm  4572.0
1860  1.003299e+07  1.483626e+14  1.218042e+07  0.985631   Sub-Saharan Africa (IDA & IBRD countries)  lightgbm  4572.0
1756  9.358719e+06  1.229774e+14  1.108952e+07  0.982122                                    IDA only  lightgbm  4520.0
```


## Model lightgbm - worst states
```
                mae           mse           rmse            r2                  state     model    rank
1669  232682.387374  5.420243e+10  232814.153874 -7.695736e+04                  Aruba  lightgbm  3402.0
1872  235242.583191  5.544065e+10  235458.373858 -1.959266e+05                  Tonga  lightgbm  3417.0
1888  228636.292952  5.227473e+10  228636.682947 -2.931268e+05  Virgin Islands (U.S.)  lightgbm  3388.0
1764  257080.833191  6.618776e+10  257269.823236 -1.030455e+06            Isle of Man  lightgbm  3481.0
1856  295094.735493  8.720199e+10  295299.830314 -2.027976e+07    St. Kitts and Nevis  lightgbm  3589.0
```


## Per target metrics - model comparision
```
             target            model           mae           mse          rmse            r2  rank
0  population_total   ensemble-arima  2.212649e+06  8.947748e+13  2.690339e+06 -3.512941e+00     1
1  population_total               rf  2.542539e+07  1.756521e+16  2.905972e+07 -1.114141e+02     4
2  population_total       simple-rnn  1.478877e+08  3.268141e+17  1.533154e+08 -8.542537e+02     8
3  population_total         base-gru  3.684453e+07  9.955284e+15  3.878141e+07 -5.603892e+03     5
4  population_total        base-lstm  8.312116e+07  3.864121e+16  8.853397e+07 -7.607874e+03     7
5  population_total          xgboost  1.939745e+07  8.165994e+15  2.316393e+07 -1.944929e+04     2
6  population_total         lightgbm  1.949421e+07  1.030385e+16  2.230181e+07 -9.331249e+04     3
7  population_total  ensemble-arimax  6.323247e+07  2.004456e+18  1.183307e+08 -3.333835e+07     6
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2            model  rank
1  2.555728e+06  1.050042e+14  3.106929e+06 -3.034854e+00   ensemble-arima   4.0
5  2.265237e+07  9.597812e+15  2.705595e+07 -1.646297e+04          xgboost  13.0
6  2.964558e+07  2.064097e+16  3.388443e+07 -9.968031e+01               rf  15.0
7  2.278202e+07  1.211098e+16  2.606263e+07 -7.849881e+04         lightgbm  16.0
4  4.260481e+07  1.167613e+16  4.485065e+07 -4.441516e+03         base-gru  17.0
3  9.581791e+07  4.517649e+16  1.021448e+08 -6.262269e+03        base-lstm  25.0
2  1.728076e+08  3.840784e+17  1.791426e+08 -5.779527e+02       simple-rnn  26.0
0  5.342596e+07  1.490051e+18  9.770157e+07 -2.437582e+07  ensemble-arimax  28.0
```


