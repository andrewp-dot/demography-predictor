
# SecondModelSelection

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
                          target          model       mae       mse      rmse          r2  rank
0           population_ages_0-14       LightGBM  0.449832  0.505705  0.533965   -2.089682     3
1           population_ages_0-14        XGBoost  0.478217  0.667427  0.606794   -2.497383     5
2           population_ages_0-14  random_forest  0.731907  1.371745  0.846763  -14.960379    11
3           population_ages_0-14          ARIMA  0.774603  1.770384  0.954558  -17.061588    12
4           population_ages_0-14         ARIMAX  1.066918  3.790711  1.280779  -60.137786    17
5           population_ages_0-14           LSTM  1.687159  5.650604  1.808610  -65.256108    22
6           population_ages_0-14            RNN  1.626553  4.481320  1.765476  -88.372686    21
7           population_ages_0-14            GRU  2.271102  8.196237  2.418588 -108.087261    24
8          population_ages_15-64        XGBoost  0.544665  1.033470  0.687576   -3.305675     8
9          population_ages_15-64       LightGBM  0.606528  1.126280  0.735565   -4.340259     9
10         population_ages_15-64          ARIMA  0.860648  1.964225  1.054725   -5.423368    13
11         population_ages_15-64  random_forest  1.026569  2.791326  1.195392  -19.449460    15
12         population_ages_15-64            RNN  1.234860  2.699417  1.319509  -30.729315    18
13         population_ages_15-64            GRU  1.553919  4.055350  1.656459  -39.669030    20
14         population_ages_15-64           LSTM  1.961220  7.817834  2.141462  -48.130883    23
15         population_ages_15-64         ARIMAX  1.286266  4.769994  1.532711  -97.014285    19
16  population_ages_65_and_above  random_forest  0.455044  0.720366  0.544857   -2.737803     4
17  population_ages_65_and_above          ARIMA  0.505753  0.859120  0.624398   -3.834502     6
18  population_ages_65_and_above        XGBoost  0.295210  0.380312  0.373121   -4.008964     1
19  population_ages_65_and_above       LightGBM  0.297900  0.377071  0.359557   -4.307343     2
20  population_ages_65_and_above         ARIMAX  0.542179  0.979602  0.669532   -8.629867     7
21  population_ages_65_and_above            GRU  0.922059  1.682950  0.979121  -67.098128    14
22  population_ages_65_and_above            RNN  1.032048  1.861657  1.110412 -113.214114    16
23  population_ages_65_and_above           LSTM  0.705735  0.905420  0.756784 -118.821971    10
```


## Overall metrics - model comparision
```
        mae       mse      rmse         r2          model  rank
5  0.469907  0.773085  0.594448  -3.162772        XGBoost   6.0
7  0.474313  0.730706  0.571284  -3.477708       LightGBM   6.0
0  0.756170  1.665824  0.931751  -7.856792          ARIMA  14.0
6  0.748777  1.724899  0.879608 -11.892065  random_forest  14.0
1  0.969567  3.015138  1.168805 -49.691072         ARIMAX  20.0
2  1.302317  3.052082  1.407487 -68.266003            RNN  26.0
3  1.421084  4.666500  1.546407 -64.632170           LSTM  27.0
4  1.606564  4.749886  1.715321 -64.959586            GRU  31.0
```


