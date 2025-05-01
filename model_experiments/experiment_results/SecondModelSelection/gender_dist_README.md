
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
          mae       mse      rmse        r2        state  model   rank
211  0.015000  0.000421  0.020521  0.977578  Timor-Leste  ARIMA  257.0
226  0.025886  0.000928  0.030471  0.968880      Vanuatu  ARIMA  507.0
178  0.004758  0.000055  0.007414  0.966273      Romania  ARIMA   21.0
```


## Model ARIMA - worst states
```
          mae       mse      rmse          r2                   state  model    rank
217  0.050368  0.004025  0.063446 -513.437759            Turkmenistan  ARIMA  2881.0
24   0.215491  0.067460  0.259731 -668.480003  Bosnia and Herzegovina  ARIMA  4673.0
78   0.238066  0.080901  0.284431 -990.636206                 Georgia  ARIMA  4852.0
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
342   0.517685    0.436414   0.648562   -2339.575254       Isle of Man  ARIMAX  6085.0
280   1.027438    2.196977   1.084198   -3372.887059  Congo, Dem. Rep.  ARIMAX  7081.0
299  22.782853  689.051011  26.249781 -885914.544165           Eritrea  ARIMAX  7584.0
```


## Model RNN - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model RNN - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model LSTM - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model LSTM - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model GRU - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model GRU - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model XGBoost - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model XGBoost - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model random_forest - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model random_forest - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model LightGBM - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model LightGBM - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Per target metrics - model comparision
```
               target                      model       mae       mse      rmse           r2  rank
0   population_female        gender_dist_XGBoost  0.149675  0.105588  0.185706   -10.355222     1
1   population_female       gender_dist_LightGBM  0.165890  0.138393  0.195994   -13.530153     3
2   population_female  gender_dist_random_forest  0.218305  0.452014  0.247069   -16.151870     7
3   population_female                      ARIMA  0.187231  0.228593  0.225463   -17.903565     5
4   population_female           gender_dist_LSTM  0.710960  1.425313  0.741063 -1125.246774    11
5   population_female            gender_dist_GRU  0.781937  1.784101  0.836499 -1470.623202    13
6   population_female            gender_dist_RNN  0.924856  2.207752  0.968636 -1592.126708    15
7   population_female                     ARIMAX  0.325619  3.210540  0.383907 -3773.928662     9
8     population_male        gender_dist_XGBoost  0.149958  0.105221  0.185871   -10.560817     2
9     population_male       gender_dist_LightGBM  0.165890  0.138393  0.195994   -13.530154     4
10    population_male  gender_dist_random_forest  0.218337  0.452031  0.247076   -16.156741     8
11    population_male                      ARIMA  0.195532  0.296123  0.233157   -17.112842     6
12    population_male           gender_dist_LSTM  0.732298  1.837157  0.755388  -969.090816    12
13    population_male            gender_dist_RNN  0.909044  1.853983  0.960082 -1660.619765    14
14    population_male            gender_dist_GRU  1.154152  4.011211  1.183700 -2643.753297    16
15    population_male                     ARIMAX  0.345230  3.251145  0.407238 -3818.681809    10
```


## Overall metrics - model comparision
```
        mae       mse      rmse           r2                      model  rank
5  0.152435  0.102863  0.188660    -9.786589        gender_dist_XGBoost   4.0
7  0.164130  0.135818  0.193887   -10.233497       gender_dist_LightGBM   8.0
0  0.194362  0.274776  0.232729   -13.581815                      ARIMA  12.0
6  0.220588  0.459863  0.249659   -15.561229  gender_dist_random_forest  16.0
1  0.272545  1.383705  0.323066 -1425.358241                     ARIMAX  21.0
3  0.728606  1.665476  0.755860 -1044.067835           gender_dist_LSTM  23.0
2  0.930610  2.064451  0.979180 -1539.268361            gender_dist_RNN  28.0
4  0.985829  2.952317  1.029541 -2114.947706            gender_dist_GRU  32.0
```


