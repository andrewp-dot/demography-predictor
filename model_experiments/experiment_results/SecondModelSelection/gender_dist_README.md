
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
          mae       mse      rmse        r2        state  model   rank
211  0.015000  0.000421  0.020521  0.977578  Timor-Leste  ARIMA  257.0
226  0.025886  0.000928  0.030471  0.968880      Vanuatu  ARIMA  510.0
178  0.004758  0.000055  0.007414  0.966273      Romania  ARIMA   21.0
```


## Model ARIMA - worst states
```
          mae       mse      rmse          r2                   state  model    rank
217  0.050368  0.004025  0.063446 -513.437759            Turkmenistan  ARIMA  2890.0
24   0.215491  0.067460  0.259731 -668.480003  Bosnia and Herzegovina  ARIMA  4697.0
78   0.238066  0.080901  0.284431 -990.636206                 Georgia  ARIMA  4882.0
```


## Model ARIMAX - top states
```
          mae       mse      rmse        r2                     state   model  rank
263  0.009733  0.000184  0.013581  0.965374                    Brazil  ARIMAX  86.0
455  0.006567  0.000097  0.009572  0.955419                    Uganda  ARIMAX  40.0
412  0.009897  0.000140  0.011829  0.950121  Pre-demographic dividend  ARIMAX  76.0
```


## Model ARIMAX - worst states
```
           mae         mse       rmse             r2             state   model    rank
342   0.517685    0.436414   0.648562   -2339.575254       Isle of Man  ARIMAX  6158.0
280   1.027438    2.196977   1.084198   -3372.887059  Congo, Dem. Rep.  ARIMAX  7184.0
299  22.782853  689.051011  26.249781 -885914.544165           Eritrea  ARIMAX  7584.0
```


## Model RNN - top states
```
          mae       mse      rmse        r2    state model    rank
628  0.148530  0.051484  0.226602  0.719811    Nepal   RNN  2774.0
687  0.258897  0.085501  0.292404  0.661520    Tonga   RNN  3292.0
597  0.291893  0.105845  0.324997  0.407740  Lebanon   RNN  3487.0
```


## Model RNN - worst states
```
          mae       mse      rmse             r2         state model    rank
691  0.512589  0.285823  0.533545  -36526.345587  Turkmenistan   RNN  5941.0
610  0.850555  0.767565  0.876024  -60954.608433          Mali   RNN  6841.0
606  1.520662  2.573371  1.603308 -134784.904727    Madagascar   RNN  7408.0
```


## Model LSTM - top states
```
          mae       mse      rmse        r2                   state model    rank
908  0.090292  0.012070  0.109759  0.835199     St. Kitts and Nevis  LSTM  1870.0
717  0.020820  0.000571  0.023731  0.607413     Antigua and Barbuda  LSTM   480.0
746  0.088861  0.010857  0.100051  0.559356  Caribbean small states  LSTM  1891.0
```


## Model LSTM - worst states
```
          mae       mse      rmse            r2         state model    rank
847  0.657547  0.466126  0.666337 -37016.064513          Mali  LSTM  6371.0
928  0.634090  0.406192  0.636414 -51909.152124  Turkmenistan  LSTM  6272.0
843  1.098421  1.285408  1.133738 -67325.040906    Madagascar  LSTM  7192.0
```


## Model GRU - top states
```
           mae       mse      rmse        r2                state model    rank
1145  0.090772  0.012535  0.111830  0.828857  St. Kitts and Nevis   GRU  1893.0
1061  0.124120  0.026506  0.162785  0.716033          Korea, Rep.   GRU  2383.0
964   0.161869  0.031441  0.176896  0.394671           Bangladesh   GRU  2669.0
```


## Model GRU - worst states
```
           mae       mse      rmse            r2         state model    rank
1165  0.429913  0.191586  0.436612 -24483.099787  Turkmenistan   GRU  5633.0
1084  0.688970  0.491688  0.700280 -39046.087497          Mali   GRU  6455.0
1080  0.930812  0.939254  0.968685 -49194.477587    Madagascar   GRU  6977.0
```


## Model XGBoost - top states
```
           mae       mse      rmse        r2       state    model   rank
1206  0.012078  0.000196  0.013986  0.984874       Benin  XGBoost  107.0
1389  0.018461  0.000578  0.024040  0.978821    Suriname  XGBoost  346.0
1198  0.020061  0.000884  0.029734  0.953393  Azerbaijan  XGBoost  447.0
```


## Model XGBoost - worst states
```
           mae       mse      rmse          r2      state    model    rank
1309  0.347963  0.184390  0.429406 -219.575803    Lesotho  XGBoost  5271.0
1325  0.312272  0.160626  0.400782 -251.390190  Mauritius  XGBoost  5177.0
1404  0.127659  0.020328  0.142572 -412.961134    Ukraine  XGBoost  3887.0
```


## Model random_forest - top states
```
           mae       mse      rmse        r2          state          model   rank
1657  0.010893  0.000155  0.012447  0.978027         Israel  random_forest   76.0
1508  0.014587  0.000370  0.019246  0.964811  Guinea-Bissau  random_forest  240.0
1591  0.012272  0.000197  0.014027  0.957321       Paraguay  random_forest  124.0
```


## Model random_forest - worst states
```
           mae       mse      rmse          r2         state          model    rank
1613  0.176675  0.043124  0.207663 -481.336421       Somalia  random_forest  4385.0
1558  0.074328  0.007793  0.088280 -617.907904          Mali  random_forest  3294.0
1639  0.073285  0.006048  0.077768 -771.902298  Turkmenistan  random_forest  3196.0
```


## Model LightGBM - top states
```
           mae       mse      rmse        r2       state     model   rank
1867  0.012221  0.000232  0.015228  0.963746  Tajikistan  LightGBM  142.0
1742  0.014156  0.000235  0.015318  0.931563        Guam  LightGBM  185.0
1877  0.013169  0.000234  0.015293  0.892944      Uganda  LightGBM  187.0
```


## Model LightGBM - worst states
```
           mae       mse      rmse          r2    state     model    rank
1768  0.997540  1.699446  1.303628 -361.282460   Jordan  LightGBM  7034.0
1795  0.073287  0.006037  0.077701 -478.463715     Mali  LightGBM  3149.0
1678  0.729577  0.675125  0.821660 -867.400151  Belgium  LightGBM  6554.0
```


## Per target metrics - model comparision
```
               target          model       mae       mse      rmse           r2  rank
0   population_female        XGBoost  0.149675  0.105588  0.185706   -10.355222     1
1   population_female       LightGBM  0.165890  0.138393  0.195994   -13.530153     3
2   population_female  random_forest  0.218305  0.452014  0.247069   -16.151870     7
3   population_female          ARIMA  0.187231  0.228593  0.225463   -17.903565     5
4   population_female            GRU  0.731105  1.326647  0.761868  -906.950224    12
5   population_female           LSTM  0.748053  1.102718  0.774127 -1315.414691    13
6   population_female            RNN  0.917243  1.546056  0.952018 -1805.284201    16
7   population_female         ARIMAX  0.325619  3.210540  0.383907 -3773.928662     9
8     population_male        XGBoost  0.149958  0.105221  0.185871   -10.560817     2
9     population_male       LightGBM  0.165890  0.138393  0.195994   -13.530154     4
10    population_male  random_forest  0.218337  0.452031  0.247076   -16.156741     8
11    population_male          ARIMA  0.195532  0.296123  0.233157   -17.112842     6
12    population_male           LSTM  0.636409  0.932722  0.662322  -950.966584    11
13    population_male            GRU  0.749724  1.316380  0.779522 -1060.654022    14
14    population_male            RNN  0.781984  1.418513  0.829988 -1691.327587    15
15    population_male         ARIMAX  0.345230  3.251145  0.407238 -3818.681809    10
```


## Overall metrics - model comparision
```
        mae       mse      rmse           r2          model  rank
5  0.152435  0.102863  0.188660    -9.786589        XGBoost   4.0
7  0.164130  0.135818  0.193887   -10.233497       LightGBM   8.0
0  0.194362  0.274776  0.232729   -13.581815          ARIMA  12.0
6  0.220588  0.459863  0.249659   -15.561229  random_forest  16.0
3  0.703759  1.037625  0.730663 -1072.765528           LSTM  23.0
1  0.272545  1.383705  0.323066 -1425.358241         ARIMAX  24.0
4  0.758816  1.368895  0.790422  -962.848544            GRU  25.0
2  0.855707  1.499688  0.897700 -1724.578089            RNN  32.0
```


