
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
               mae           mse           rmse        r2                                   state  model    rank
89   643539.036691  6.452739e+11  803289.407770  0.999893  Heavily indebted poor countries (HIPC)  ARIMA  2535.0
156    4594.370801  2.915400e+07    5399.444764  0.999743                               Nicaragua  ARIMA   185.0
157   43475.303300  2.738167e+09   52327.501567  0.999670                                   Niger  ARIMA   667.0
```


## Model ARIMA - worst states
```
              mae           mse          rmse          r2                state  model    rank
123  2.097826e+06  5.548798e+12  2.355589e+06  -68.489339              Lebanon  ARIMA  4928.0
195  3.448471e+06  1.587272e+13  3.984059e+06 -143.231810                Spain  ARIMA  5431.0
197  8.275954e+02  1.097134e+06  1.047442e+03 -254.150314  St. Kitts and Nevis  ARIMA  1696.0
```


## Model ARIMAX - top states
```
              mae           mse          rmse        r2                                         state   model    rank
427  7.891424e+02  8.331946e+05  9.127950e+02  0.999490                               Solomon Islands  ARIMAX    49.0
359  2.780104e+06  1.073488e+13  3.276413e+06  0.998339  Least developed countries: UN classification  ARIMAX  3708.0
394  1.003942e+05  1.715216e+10  1.309662e+05  0.997936                                         Niger  ARIMAX  1135.0
```


## Model ARIMAX - worst states
```
              mae           mse          rmse           r2                 state   model    rank
329  1.710518e+06  4.314267e+12  2.077081e+06  -537.056299  Hong Kong SAR, China  ARIMAX  5003.0
457  5.864923e+06  5.244867e+13  7.242145e+06  -860.252097  United Arab Emirates  ARIMAX  5929.0
342  7.603469e+03  9.238063e+07  9.611485e+03 -1437.245011           Isle of Man  ARIMAX  2009.0
```


## Model RNN - top states
```
               mae           mse          rmse        r2         state model    rank
558  227293.752614  7.578971e+10  2.752993e+05  0.919797     Guatemala   RNN  1895.0
485  574717.090514  4.485607e+11  6.697468e+05  0.744889     Australia   RNN  2723.0
503  771924.822971  1.137952e+12  1.066748e+06  0.722536  Burkina Faso   RNN  3081.0
```


## Model RNN - worst states
```
              mae           mse          rmse            r2               state model    rank
653  4.066490e+07  1.737464e+15  4.168289e+07 -29511.376771  Russian Federation   RNN  6868.0
623  2.076987e+05  4.363813e+10  2.088974e+05 -30582.151053          Montenegro   RNN  3385.0
522  2.666337e+06  7.320552e+12  2.705652e+06 -32525.471932                Cuba   RNN  5476.0
```


## Model LSTM - top states
```
               mae           mse           rmse        r2       state model    rank
891  113481.734818  1.956584e+10  139877.955531  0.982650      Rwanda  LSTM  1261.0
850   67714.920875  6.422162e+09   80138.394216  0.953696  Mauritania  LSTM  1056.0
846   20238.752596  5.297154e+08   23015.547460  0.789195    Maldives  LSTM   707.0
```


## Model LSTM - worst states
```
              mae           mse          rmse             r2                state model    rank
898  3.154488e+06  1.003033e+13  3.167069e+06 -118052.011975      Slovak Republic  LSTM  5603.0
883  3.074624e+07  9.717476e+14  3.117287e+07 -153025.235818               Poland  LSTM  6794.0
908  6.306192e+04  4.181850e+09  6.466722e+04 -972533.376962  St. Kitts and Nevis  LSTM  2704.0
```


## Model GRU - top states
```
                mae           mse           rmse        r2               state model    rank
1001   65929.157157  7.728819e+09   87913.699505  0.963176  Dominican Republic   GRU  1074.0
1128  285918.248270  1.114410e+11  333827.845655  0.901180              Rwanda   GRU  2058.0
1084  388638.053463  3.542155e+11  595160.089365  0.898256                Mali   GRU  2426.0
```


## Model GRU - worst states
```
               mae           mse          rmse             r2                state model    rank
1135  2.598493e+06  6.871498e+12  2.621354e+06  -80873.840590      Slovak Republic   GRU  5453.0
1053  1.548027e+05  2.595337e+10  1.611005e+05 -404058.828765          Isle of Man   GRU  3207.0
1145  4.790959e+04  2.504333e+09  5.004331e+04 -582408.654988  St. Kitts and Nevis   GRU  2560.0
```


## Model XGBoost - top states
```
               mae           mse          rmse        r2                                   state    model    rank
1274  5.368880e+06  4.680301e+13  6.841272e+06  0.992227  Heavily indebted poor countries (HIPC)  XGBoost  4191.0
1186  8.435823e+06  1.125880e+14  1.061075e+07  0.969256             Africa Eastern and Southern  XGBoost  4533.0
1283  1.780909e+07  5.317715e+14  2.306017e+07  0.968403                               IDA total  XGBoost  4909.0
```


## Model XGBoost - worst states
```
                mae           mse           rmse            r2                state    model    rank
1398   84113.595703  7.384095e+09   85930.757389 -2.609444e+04                Tonga  XGBoost  2839.0
1290  100609.180176  1.127519e+10  106184.692769 -1.755389e+05          Isle of Man  XGBoost  2947.0
1382  134483.739583  1.868723e+10  136701.256240 -4.345917e+06  St. Kitts and Nevis  XGBoost  3094.0
```


## Model random_forest - top states
```
                mae           mse           rmse        r2          state          model    rank
1589   15767.805846  3.516306e+08   18751.814912  0.994411         Panama  random_forest   376.0
1513   33882.859058  2.523572e+09   50235.166269  0.993528       Honduras  random_forest   667.0
1468  160887.340675  3.448352e+10  185697.375731  0.992509  Cote d'Ivoire  random_forest  1435.0
```


## Model random_forest - worst states
```
               mae           mse          rmse            r2               state          model    rank
1601  8.853967e+06  1.269303e+14  1.126633e+07  -2155.024296  Russian Federation  random_forest  6235.0
1609  4.045856e+05  2.291959e+11  4.787440e+05  -2696.545641     Slovak Republic  random_forest  3924.0
1470  1.471483e+06  2.969394e+12  1.723193e+06 -13192.527040                Cuba  random_forest  5037.0
```


## Model LightGBM - top states
```
               mae           mse          rmse        r2                                       state     model    rank
1661  3.598709e+06  1.562109e+13  3.952352e+06  0.990751                  Africa Western and Central  LightGBM  3904.0
1861  1.001171e+07  1.472099e+14  1.213301e+07  0.985741  Sub-Saharan Africa (excluding high income)  LightGBM  4565.0
1859  1.003299e+07  1.483626e+14  1.218042e+07  0.985631                          Sub-Saharan Africa  LightGBM  4572.0
```


## Model LightGBM - worst states
```
                mae           mse           rmse            r2                  state     model    rank
1888  228636.292952  5.227473e+10  228636.682947 -2.931268e+05  Virgin Islands (U.S.)  LightGBM  3491.0
1764  257080.833191  6.618776e+10  257269.823236 -1.030455e+06            Isle of Man  LightGBM  3590.0
1856  295094.735493  8.720199e+10  295299.830314 -2.027976e+07    St. Kitts and Nevis  LightGBM  3705.0
```


## Per target metrics - model comparision
```
             target          model           mae           mse          rmse            r2  rank
0  population_total          ARIMA  1.795531e+06  3.121514e+13  2.270705e+06     -2.120259     1
1  population_total         ARIMAX  6.999138e+06  9.793895e+14  8.643719e+06    -17.538239     2
2  population_total  random_forest  2.542539e+07  1.756521e+16  2.905972e+07   -111.414059     5
3  population_total            RNN  1.478877e+08  3.268141e+17  1.533154e+08   -854.253684     8
4  population_total            GRU  3.684453e+07  9.955284e+15  3.878141e+07  -5603.891923     6
5  population_total           LSTM  8.312116e+07  3.864121e+16  8.853397e+07  -7607.873775     7
6  population_total        XGBoost  1.939745e+07  8.165994e+15  2.316393e+07 -19449.285323     3
7  population_total       LightGBM  1.949421e+07  1.030385e+16  2.230181e+07 -93312.489894     4
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2          model  rank
0  2.071033e+06  3.656528e+13  2.620776e+06     -1.804440          ARIMA   4.0
1  8.120780e+06  1.150009e+15  1.002770e+07    -13.435932         ARIMAX   8.0
5  2.265237e+07  9.597812e+15  2.705595e+07 -16462.971860        XGBoost  17.0
6  2.964558e+07  2.064097e+16  3.388443e+07    -99.680314  random_forest  19.0
7  2.278202e+07  1.211098e+16  2.606263e+07 -78498.805717       LightGBM  20.0
4  4.260481e+07  1.167613e+16  4.485065e+07  -4441.516331            GRU  21.0
3  9.581791e+07  4.517649e+16  1.021448e+08  -6262.269084           LSTM  27.0
2  1.728076e+08  3.840784e+17  1.791426e+08   -577.952715            RNN  28.0
```


