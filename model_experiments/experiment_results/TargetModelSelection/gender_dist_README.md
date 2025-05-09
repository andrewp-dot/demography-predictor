
# TargetModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
          mae       mse      rmse      mape        r2       state              model   rank
415  0.004758  0.000055  0.007414  0.000095  0.966273     Romania  gender_dist_ARIMA   21.0
366  0.006854  0.000065  0.008058  0.000137  0.750756  Low income  gender_dist_ARIMA  108.0
422  0.008432  0.000081  0.008984  0.000169 -3.450543      Serbia  gender_dist_ARIMA  880.0
```


## Model ARIMA - worst states
```
          mae        mse      rmse      mape        r2                 state              model    rank
457  1.227538   2.654877  1.629378  0.026810 -9.551554  United Arab Emirates  gender_dist_ARIMA  6525.0
414  1.487304   3.575423  1.832278  0.032057 -1.711774                 Qatar  gender_dist_ARIMA  6152.0
400  5.513655  35.562899  5.938605  0.116741 -6.862125                  Oman  gender_dist_ARIMA  6684.0
```


## Model ARIMAX - top states
```
          mae       mse      rmse      mape        r2       state               model    rank
157  0.006131  0.000050  0.007072  0.000123  0.936524       Niger  gender_dist_ARIMAX    34.0
218  0.006567  0.000097  0.009572  0.000131  0.955419      Uganda  gender_dist_ARIMAX    40.0
132  0.008824  0.000117  0.010820  0.000176 -5.132455  Madagascar  gender_dist_ARIMAX  1008.0
```


## Model ARIMAX - worst states
```
           mae         mse       rmse      mape             r2                 state               model    rank
92    2.622882    8.215851   2.860427  0.052432    -154.413176  Hong Kong SAR, China  gender_dist_ARIMAX  7199.0
163   3.407659   14.891171   3.856428  0.073036      -2.292090                  Oman  gender_dist_ARIMAX  6337.0
62   22.782853  689.051011  26.249781  0.455787 -885914.544165               Eritrea  gender_dist_ARIMAX  7584.0
```


## Model RNN - top states
```
          mae       mse      rmse      mape          r2         state            model    rank
488  0.174557  0.036938  0.186477  0.003466    0.409271  Bahamas, The  gender_dist_RNN  2717.0
615  0.176133  0.040698  0.186490  0.003566   -3.024693        Mexico  gender_dist_RNN  3365.0
548  0.183981  0.054490  0.189284  0.003595 -275.885995        France  gender_dist_RNN  4322.0
```


## Model RNN - worst states
```
          mae        mse      rmse      mape         r2     state            model    rank
651  7.406509  57.167668  7.541602  0.200488 -42.358741     Qatar  gender_dist_RNN  7040.0
609  7.482870  61.654106  7.837089  0.158184 -19.511208  Maldives  gender_dist_RNN  6906.0
637  8.020299  70.695620  8.386375  0.177672 -14.629148      Oman  gender_dist_RNN  6859.0
```


## Model LSTM - top states
```
          mae       mse      rmse      mape         r2   state             model    rank
785  0.045632  0.002590  0.050604  0.000912 -12.161395  France  gender_dist_LSTM  2091.0
946  0.057227  0.004558  0.062438  0.001139   0.353574  Israel  gender_dist_LSTM  1418.0
844  0.082955  0.007094  0.083923  0.001663  -3.032785  Malawi  gender_dist_LSTM  2361.0
```


## Model LSTM - worst states
```
          mae        mse      rmse      mape         r2     state             model    rank
888  6.894418  51.734376  6.917873  0.155194 -38.237867     Qatar  gender_dist_LSTM  7014.0
874  7.151270  56.263901  7.464505  0.150720 -11.438633      Oman  gender_dist_LSTM  6793.0
846  7.482131  60.164501  7.747253  0.154481 -19.015643  Maldives  gender_dist_LSTM  6900.0
```


## Model GRU - top states
```
           mae       mse      rmse      mape         r2    state            model    rank
999   0.070776  0.006766  0.078045  0.001419  -0.091402  Denmark  gender_dist_GRU  1755.0
1172  0.101018  0.011043  0.102946  0.002009 -10.860935  Uruguay  gender_dist_GRU  2944.0
996   0.115655  0.016105  0.116053  0.002305  -3.690388     Cuba  gender_dist_GRU  2874.0
```


## Model GRU - worst states
```
           mae         mse      rmse      mape         r2     state            model    rank
1083  8.063829   71.588479  8.433093  0.165357 -22.816194  Maldives  gender_dist_GRU  6943.0
1111  8.862070   88.334980  9.262537  0.182566 -18.528798      Oman  gender_dist_GRU  6909.0
1125  9.048767  110.166365  9.375131  0.173267 -82.555530     Qatar  gender_dist_GRU  7177.0
```


## Model XGBoost - top states
```
           mae       mse      rmse      mape       r2       state                model   rank
1317  0.002509  0.000010  0.003135  0.000050  0.48509  Madagascar  gender_dist_XGBoost  163.0
1341  0.009744  0.000156  0.012492  0.000195  0.78652   Nicaragua  gender_dist_XGBoost  144.0
1403  0.011561  0.000167  0.012911  0.000231  0.92363      Uganda  gender_dist_XGBoost  111.0
```


## Model XGBoost - worst states
```
           mae       mse      rmse      mape         r2     state                model    rank
1348  1.306597  2.468171  1.571041  0.028554   0.454345      Oman  gender_dist_XGBoost  5610.0
1299  1.462631  3.725644  1.930181  0.030120 -26.013428    Kuwait  gender_dist_XGBoost  6802.0
1320  1.786278  4.652843  2.157041  0.037456  -0.547917  Maldives  gender_dist_XGBoost  5970.0
```


## Model random_forest - top states
```
           mae       mse      rmse      mape        r2             state                      model   rank
1560  0.002733  0.000018  0.004202  0.000055  0.952525  Marshall Islands  gender_dist_random_forest   20.0
1554  0.004776  0.000039  0.006209  0.000096 -1.019499        Madagascar  gender_dist_random_forest  530.0
1461  0.005084  0.000046  0.006775  0.000102  0.267165             Chile  gender_dist_random_forest  227.0
```


## Model random_forest - worst states
```
           mae        mse      rmse      mape         r2                 state                      model    rank
1642  4.197349  18.859766  4.342783  0.092022 -73.956322  United Arab Emirates  gender_dist_random_forest  7108.0
1599  4.336094  19.975525  4.469399  0.113880 -14.150410                 Qatar  gender_dist_random_forest  6806.0
1585  5.156881  31.103328  5.577036  0.112197  -5.876219                  Oman  gender_dist_random_forest  6636.0
```


## Model LightGBM - top states
```
           mae       mse      rmse      mape        r2                   state                 model    rank
1844  0.007023  0.000077  0.008790  0.000141 -3.260252                  Serbia  gender_dist_LightGBM   856.0
1683  0.006898  0.000118  0.010882  0.000138 -0.175090  Bosnia and Herzegovina  gender_dist_LightGBM   376.0
1791  0.009556  0.000120  0.010967  0.000191 -5.300077              Madagascar  gender_dist_LightGBM  1020.0
```


## Model LightGBM - worst states
```
           mae       mse      rmse      mape        r2     state                 model    rank
1836  1.410774  2.478913  1.574456  0.037220 -0.880128     Qatar  gender_dist_LightGBM  5945.0
1822  2.159406  6.215612  2.493113  0.047133 -0.374126      Oman  gender_dist_LightGBM  5969.0
1794  2.409171  8.606962  2.933762  0.050533 -1.863381  Maldives  gender_dist_LightGBM  6270.0
```


## Per target metrics - model comparision
```
               target                      model       mae       mse      rmse      mape           r2  rank
0   population_female        gender_dist_XGBoost  0.149675  0.105588  0.185706  0.003198   -10.355222     1
1   population_female       gender_dist_LightGBM  0.165890  0.138393  0.195994  0.003608   -13.530153     3
2   population_female          gender_dist_ARIMA  0.187231  0.228593  0.225463  0.004065   -17.903565     5
3   population_female  gender_dist_random_forest  0.218305  0.452014  0.247069  0.005196   -16.151870     7
4   population_female         gender_dist_ARIMAX  0.325619  3.210540  0.383907  0.006885 -3773.928662     9
5   population_female           gender_dist_LSTM  0.576750  1.074527  0.608266  0.012550  -729.883667    11
6   population_female            gender_dist_GRU  0.607817  1.282896  0.658883  0.013203  -971.530480    12
7   population_female            gender_dist_RNN  0.940487  2.197384  0.978588  0.020429 -1703.330407    14
8     population_male        gender_dist_XGBoost  0.149958  0.105221  0.185871  0.002906   -10.560817     2
9     population_male       gender_dist_LightGBM  0.165890  0.138393  0.195994  0.003183   -13.530154     4
10    population_male          gender_dist_ARIMA  0.195532  0.296123  0.233157  0.003711   -17.112842     6
11    population_male  gender_dist_random_forest  0.218337  0.452031  0.247076  0.003982   -16.156741     8
12    population_male         gender_dist_ARIMAX  0.345230  3.251145  0.407238  0.006719 -3818.681809    10
13    population_male           gender_dist_LSTM  0.789953  1.890580  0.811041  0.014964 -1122.795298    13
14    population_male            gender_dist_RNN  0.999808  1.997260  1.047221  0.019365 -2122.966137    15
15    population_male            gender_dist_GRU  1.067526  3.746765  1.091656  0.020132 -2283.311900    16
```


## Overall metrics - model comparision
```
        mae       mse      rmse      mape           r2                      model  rank
5  0.152435  0.102863  0.188660  0.003103    -9.786589        gender_dist_XGBoost   4.0
7  0.164130  0.135818  0.193887  0.003360   -10.233497       gender_dist_LightGBM   8.0
1  0.194362  0.274776  0.232729  0.003949   -13.581815          gender_dist_ARIMA  12.0
6  0.220588  0.459863  0.249659  0.004634   -15.561229  gender_dist_random_forest  16.0
0  0.272545  1.383705  0.323066  0.005545 -1425.358241         gender_dist_ARIMAX  21.0
3  0.690887  1.513531  0.717819  0.013913  -925.760349           gender_dist_LSTM  23.0
4  0.844248  2.546971  0.883037  0.016798 -1673.681444            gender_dist_GRU  29.0
2  0.984102  2.138364  1.027802  0.020173 -1828.977594            gender_dist_RNN  31.0
```


