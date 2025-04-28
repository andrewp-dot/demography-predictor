
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
               mae           mse          rmse        r2                        state           model    rank
394   38868.538462  2.246997e+09  4.740250e+04  0.999730                        Niger  ensemble-arima   666.0
427     596.041667  4.464089e+05  6.681384e+02  0.999727              Solomon Islands  ensemble-arima    46.0
238  854625.384615  1.800620e+12  1.341872e+06  0.999508  Africa Eastern and Southern  ensemble-arima  2661.0
463     408.079861  2.297684e+05  4.793417e+02  0.999289                      Vanuatu  ensemble-arima    36.0
269   15990.200000  3.267200e+08  1.807540e+04  0.999012                     Cambodia  ensemble-arima   367.0
```


## Model ensemble-arima - worst states
```
              mae           mse          rmse          r2                 state           model    rank
434  2.124653e+02  6.966175e+04  2.639351e+02  -15.200594   St. Kitts and Nevis  ensemble-arima  1097.0
416  7.980142e+05  1.196820e+12  1.093993e+06  -19.329059    Russian Federation  ensemble-arima  3656.0
444  7.415794e+06  7.673303e+13  8.759739e+06  -41.410248  Syrian Arab Republic  ensemble-arima  5292.0
432  4.649685e+06  2.939709e+13  5.421908e+06 -266.124650                 Spain  ensemble-arima  5262.0
360  4.872601e+06  3.485500e+13  5.903813e+06 -435.500165               Lebanon  ensemble-arima  5387.0
```


## Model ensemble-arimax - top states
```
              mae           mse          rmse        r2                    state            model    rank
95   9.670812e+06  1.943424e+14  1.394067e+07  0.997766         IDA & IBRD total  ensemble-arimax  4312.0
146  1.102659e+07  1.711622e+14  1.308290e+07  0.996935            Middle income  ensemble-arimax  4325.0
130  8.943337e+06  1.012435e+14  1.006198e+07  0.995849      Lower middle income  ensemble-arimax  4183.0
194  5.990130e+06  4.532530e+13  6.732407e+06  0.993378  South Asia (IDA & IBRD)  ensemble-arimax  3943.0
193  5.990130e+06  4.532530e+13  6.732407e+06  0.993378               South Asia  ensemble-arimax  3943.0
```


## Model ensemble-arimax - worst states
```
              mae           mse          rmse            r2                        state            model    rank
165  4.019290e+06  3.495259e+13  5.912072e+06 -4.809188e+03  Pacific island small states  ensemble-arimax  5531.0
62   6.523108e+06  5.664554e+13  7.526323e+06 -1.007786e+04                      Eritrea  ensemble-arimax  5788.0
63   1.864880e+06  7.268571e+12  2.696029e+06 -2.160703e+05                      Estonia  ensemble-arimax  5085.0
206  3.412021e+09  3.386257e+19  5.819155e+09 -3.566389e+08                  Switzerland  ensemble-arimax  7539.0
116  1.058057e+10  4.411562e+20  2.100372e+10 -7.544312e+09                      Lao PDR  ensemble-arimax  7581.0
```


## Model simple-rnn - top states
```
              mae           mse          rmse        r2              state       model    rank
512  2.788405e+05  1.203131e+11  3.468617e+05  0.961812               Chad  simple-rnn  1883.0
521  8.433143e+04  1.005535e+10  1.002764e+05 -0.185201            Croatia  simple-rnn  1542.0
485  1.788988e+06  3.366367e+12  1.834766e+06 -0.914558          Australia  simple-rnn  3643.0
535  1.808223e+05  5.098014e+10  2.257878e+05 -2.094961  Equatorial Guinea  simple-rnn  2201.0
627  1.757872e+05  4.532182e+10  2.128892e+05 -3.073198            Namibia  simple-rnn  2245.0
```


## Model simple-rnn - worst states
```
              mae           mse          rmse             r2                 state       model    rank
566  5.508391e+06  8.999631e+13  9.486639e+06  -11222.942960  Hong Kong SAR, China  simple-rnn  5846.0
614  7.795088e+05  6.661409e+11  8.161745e+05  -16760.024102             Mauritius  simple-rnn  4224.0
646  1.133517e+07  1.436238e+14  1.198432e+07  -22616.201065                Poland  simple-rnn  6117.0
671  1.795785e+04  3.601540e+08  1.897772e+04  -83756.700625   St. Kitts and Nevis  simple-rnn  2249.0
522  6.218787e+06  4.219099e+13  6.495459e+06 -187460.848574                  Cuba  simple-rnn  5772.0
```


## Model base-lstm - top states
```
              mae           mse          rmse        r2                           state      model    rank
910    396.874561  2.131453e+05    461.676627  0.930988  St. Vincent and the Grenadines  base-lstm   208.0
893   2622.878723  9.307021e+06   3050.741091  0.928727           Sao Tome and Principe  base-lstm   333.0
823   7891.068623  6.582599e+07   8113.321666 -0.882359                        Kiribati  base-lstm   881.0
892   9176.180448  9.344453e+07   9666.671028 -1.017823                           Samoa  base-lstm   917.0
810  30223.636219  1.177561e+09  34315.608041 -2.352363                         Iceland  base-lstm  1352.0
```


## Model base-lstm - worst states
```
              mae           mse          rmse             r2               state      model    rank
789  1.010270e+06  1.029684e+12  1.014733e+06  -22894.833489             Georgia  base-lstm  4416.0
898  1.857845e+06  3.481586e+12  1.865901e+06  -40975.908733     Slovak Republic  base-lstm  4871.0
883  1.607468e+07  2.609592e+14  1.615423e+07  -41093.620165              Poland  base-lstm  6314.0
890  7.181759e+07  5.186143e+15  7.201488e+07  -88090.295370  Russian Federation  base-lstm  6927.0
759  5.139951e+06  2.704651e+13  5.200625e+06 -120171.278762                Cuba  base-lstm  5614.0
```


## Model base-gru - top states
```
               mae           mse          rmse         r2    state     model    rank
1057  3.088388e+06  1.354939e+13  3.680949e+06 -10.910391   Jordan  base-gru  4501.0
1125  1.309933e+06  2.483626e+12  1.575952e+06 -18.416357    Qatar  base-gru  3942.0
1111  2.109178e+06  6.538694e+12  2.557087e+06 -19.390383     Oman  base-gru  4326.0
1084  1.095296e+07  1.485214e+14  1.218693e+07 -41.661088     Mali  base-gru  5504.0
1139  1.032651e+07  1.367695e+14  1.169485e+07 -44.503719  Somalia  base-gru  5483.0
```


## Model base-gru - worst states
```
               mae           mse          rmse             r2                state     model    rank
1135  5.378912e+06  3.120037e+13  5.585729e+06 -367215.103915      Slovak Republic  base-gru  5673.0
1120  4.635800e+07  2.379741e+15  4.878259e+07 -374749.442927               Poland  base-gru  6804.0
1018  1.477322e+09  3.030349e+18  1.740790e+09 -463056.890142       European Union  base-gru  7469.0
1145  4.202340e+04  1.991758e+09  4.462911e+04 -463203.778430  St. Kitts and Nevis  base-gru  2552.0
996   1.219398e+07  1.762845e+14  1.327722e+07 -783261.403329                 Cuba  base-gru  6211.0
```


## Model xgboost - top states
```
               mae           mse          rmse        r2                                     state    model    rank
1274  5.368880e+06  4.680301e+13  6.841272e+06  0.992227    Heavily indebted poor countries (HIPC)  xgboost  3933.0
1186  8.435823e+06  1.125880e+14  1.061075e+07  0.969256               Africa Eastern and Southern  xgboost  4279.0
1283  1.780909e+07  5.317715e+14  2.306017e+07  0.968403                                 IDA total  xgboost  4721.0
1258  1.404690e+07  2.721760e+14  1.649776e+07  0.951780  Fragile and conflict affected situations  xgboost  4579.0
1343  3.028460e+06  1.528844e+13  3.910043e+06  0.951205                                   Nigeria  xgboost  3651.0
```


## Model xgboost - worst states
```
                mae           mse           rmse            r2                  state    model    rank
1334  111859.031250  1.261479e+10  112315.590928 -8.839893e+03             Montenegro  xgboost  2884.0
1414   63871.738281  4.286751e+09   65473.287501 -2.403673e+04  Virgin Islands (U.S.)  xgboost  2684.0
1398   84113.595703  7.384095e+09   85930.757389 -2.609444e+04                  Tonga  xgboost  2789.0
1290  100609.180176  1.127519e+10  106184.692769 -1.755389e+05            Isle of Man  xgboost  2924.0
1382  134483.739583  1.868723e+10  136701.256240 -4.345917e+06    St. Kitts and Nevis  xgboost  3089.0
```


## Model rf - top states
```
               mae           mse          rmse        r2       state model   rank
1634  13777.104615  2.763577e+08  16624.008543  0.999436        Togo    rf  341.0
1554  60256.004615  5.283314e+09  72686.407358  0.999104  Madagascar    rf  863.0
1507  28033.958182  1.225634e+09  35009.055455  0.998657      Guinea    rf  559.0
1630  10555.565000  1.651960e+08  12852.858606  0.998656  Tajikistan    rf  299.0
1480  45690.566154  4.096917e+09  64007.167048  0.995578     Ecuador    rf  811.0
```


## Model rf - worst states
```
               mae           mse          rmse           r2        state model    rank
1439  2.412248e+04  9.288756e+08  3.047746e+04  -181.029151     Barbados    rf  1963.0
1488  3.925591e+07  2.266311e+15  4.760579e+07  -304.272001    Euro area    rf  6430.0
1493  1.071752e+05  1.553648e+10  1.246454e+05  -383.067567         Fiji    rf  2691.0
1530  1.612499e+07  3.366481e+14  1.834797e+07  -751.557196        Japan    rf  6157.0
1527  6.658750e+03  6.714049e+07  8.193930e+03 -1044.289112  Isle of Man    rf  1915.0
```


## Model lightgbm - top states
```
               mae           mse          rmse        r2                                       state     model    rank
1661  3.598709e+06  1.562109e+13  3.952352e+06  0.990751                  Africa Western and Central  lightgbm  3599.0
1861  1.001171e+07  1.472099e+14  1.213301e+07  0.985741  Sub-Saharan Africa (excluding high income)  lightgbm  4323.0
1860  1.003299e+07  1.483626e+14  1.218042e+07  0.985631   Sub-Saharan Africa (IDA & IBRD countries)  lightgbm  4329.0
1859  1.003299e+07  1.483626e+14  1.218042e+07  0.985631                          Sub-Saharan Africa  lightgbm  4329.0
1756  9.358719e+06  1.229774e+14  1.108952e+07  0.982122                                    IDA only  lightgbm  4276.0
```


## Model lightgbm - worst states
```
                mae           mse           rmse            r2                  state     model    rank
1669  232682.387374  5.420243e+10  232814.153874 -7.695736e+04                  Aruba  lightgbm  3366.0
1872  235242.583191  5.544065e+10  235458.373858 -1.959266e+05                  Tonga  lightgbm  3381.0
1888  228636.292952  5.227473e+10  228636.682947 -2.931268e+05  Virgin Islands (U.S.)  lightgbm  3362.0
1764  257080.833191  6.618776e+10  257269.823236 -1.030455e+06            Isle of Man  lightgbm  3448.0
1856  295094.735493  8.720199e+10  295299.830314 -2.027976e+07    St. Kitts and Nevis  lightgbm  3555.0
```


## Per target metrics - model comparision
```
             target            model           mae           mse          rmse            r2  rank
0  population_total   ensemble-arima  2.212649e+06  8.947748e+13  2.690339e+06 -3.512941e+00     1
1  population_total               rf  1.456636e+07  3.849493e+15  1.741097e+07 -1.363932e+01     2
2  population_total       simple-rnn  1.875287e+08  4.489091e+17  1.968007e+08 -1.912212e+03     7
3  population_total        base-lstm  1.853889e+08  3.576794e+17  1.906440e+08 -1.931380e+03     6
4  population_total          xgboost  1.939745e+07  8.165994e+15  2.316393e+07 -1.944929e+04     3
5  population_total         base-gru  6.615800e+08  6.354869e+18  7.718169e+08 -2.145659e+04     8
6  population_total         lightgbm  1.949421e+07  1.030385e+16  2.230181e+07 -9.331249e+04     4
7  population_total  ensemble-arimax  6.323247e+07  2.004456e+18  1.183307e+08 -3.333835e+07     5
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2            model  rank
1  2.555728e+06  1.050042e+14  3.106929e+06 -3.034854e+00   ensemble-arima   4.0
6  1.702478e+07  4.523354e+15  2.034712e+07 -1.353924e+01               rf   8.0
5  2.265237e+07  9.597812e+15  2.705595e+07 -1.646297e+04          xgboost  15.0
7  2.278202e+07  1.211098e+16  2.606263e+07 -7.849881e+04         lightgbm  18.0
3  2.159235e+08  4.201770e+17  2.220767e+08 -1.531429e+03        base-lstm  20.0
2  2.190826e+08  5.275334e+17  2.298538e+08 -1.698327e+03       simple-rnn  24.0
0  5.342596e+07  1.490051e+18  9.770157e+07 -2.437582e+07  ensemble-arimax  25.0
4  7.733673e+08  7.468493e+18  9.026739e+08 -1.941704e+04         base-gru  30.0
```


