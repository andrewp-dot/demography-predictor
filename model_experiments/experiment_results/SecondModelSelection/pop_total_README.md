
# TargetModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model ARIMA - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model ARIMAX - top states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
```


## Model ARIMAX - worst states
```
Empty DataFrame
Columns: [mae, mse, rmse, r2, state, model, rank]
Index: []
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
             target          model           mae           mse          rmse            r2  rank
0  population_total          ARIMA  1.795531e+06  3.121514e+13  2.270705e+06     -2.120259     1
1  population_total         ARIMAX  6.999138e+06  9.793895e+14  8.643719e+06    -17.538239     2
2  population_total  random_forest  2.542539e+07  1.756521e+16  2.905972e+07   -111.414059     5
3  population_total           LSTM  1.606577e+08  2.972176e+17  1.638162e+08  -1422.915947     7
4  population_total            RNN  1.772618e+08  3.759103e+17  1.821395e+08  -1720.323557     8
5  population_total            GRU  1.379280e+08  2.190517e+17  1.423971e+08  -2694.522372     6
6  population_total        XGBoost  1.939745e+07  8.165994e+15  2.316393e+07 -19449.285323     3
7  population_total       LightGBM  1.949421e+07  1.030385e+16  2.230181e+07 -93312.489894     4
```


## Overall metrics - model comparision
```
            mae           mse          rmse            r2          model  rank
0  2.071033e+06  3.656528e+13  2.620776e+06     -1.804440          ARIMA   4.0
1  8.120780e+06  1.150009e+15  1.002770e+07    -13.435932         ARIMAX   8.0
5  2.265237e+07  9.597812e+15  2.705595e+07 -16462.971860        XGBoost  17.0
6  2.964558e+07  2.064097e+16  3.388443e+07    -99.680314  random_forest  18.0
7  2.278202e+07  1.211098e+16  2.606263e+07 -78498.805717       LightGBM  19.0
4  1.610048e+08  2.573702e+17  1.662284e+08  -2081.634824            GRU  24.0
3  1.874660e+08  3.492160e+17  1.911600e+08  -1171.852099           LSTM  25.0
2  2.067736e+08  4.416824e+17  2.124683e+08  -1316.513221            RNN  29.0
```


