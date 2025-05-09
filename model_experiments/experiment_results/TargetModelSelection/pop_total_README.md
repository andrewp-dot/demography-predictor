
# TargetModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
            mae            mse        rmse      mape        r2     state            model   rank
463  338.572024  156493.386966  395.592451  0.001129  0.999516   Vanuatu  pop_total_ARIMA   20.0
247  377.024696  321782.422768  567.258691  0.003543  0.543123     Aruba  pop_total_ARIMA  409.0
349  512.143648  334479.514710  578.342040  0.004272  0.990435  Kiribati  pop_total_ARIMA   81.0
```


## Model ARIMA - worst states
```
              mae           mse          rmse      mape        r2                     state            model    rank
412  1.971954e+07  6.188462e+14  2.487662e+07  0.021357  0.926042  Pre-demographic dividend  pop_total_ARIMA  4916.0
332  2.695053e+07  9.545081e+14  3.089511e+07  0.004222  0.989027          IDA & IBRD total  pop_total_ARIMA  4906.0
383  2.969309e+07  1.187083e+15  3.445407e+07  0.005352  0.978743             Middle income  pop_total_ARIMA  4990.0
```


## Model ARIMAX - top states
```
            mae            mse        rmse      mape         r2                  state             model    rank
197  263.678017   77909.149719  279.122105  0.005524 -17.118615    St. Kitts and Nevis  pop_total_ARIMAX  1152.0
52   310.835468  146012.980922  382.116449  0.004426   0.883837               Dominica  pop_total_ARIMAX   231.0
229  318.656715  149736.348391  386.957812  0.003001   0.160361  Virgin Islands (U.S.)  pop_total_ARIMAX   503.0
```


## Model ARIMAX - worst states
```
              mae           mse          rmse      mape        r2                       state             model    rank
231  1.399873e+08  2.541845e+16  1.594317e+08  0.018513  0.748517                       World  pop_total_ARIMAX  5538.0
54   1.508492e+08  3.764648e+16  1.940270e+08  0.045952 -0.471996  Early-demographic dividend  pop_total_ARIMAX  5874.0
130  2.294878e+08  8.572218e+16  2.927835e+08  0.080464 -2.514406         Lower middle income  pop_total_ARIMAX  6165.0
```


## Model RNN - top states
```
              mae           mse          rmse      mape            r2                state          model    rank
579   1008.840227  1.602782e+06   1266.010181  0.012036    -23.953205          Isle of Man  pop_total_RNN  1289.0
671  10218.807505  1.107842e+08  10525.406493  0.214125 -25763.059101  St. Kitts and Nevis  pop_total_RNN  2130.0
484  24819.980698  6.658112e+08  25803.317278  0.234273   -944.340167                Aruba  pop_total_RNN  2218.0
```


## Model RNN - worst states
```
              mae           mse          rmse      mape          r2                state          model    rank
602  4.013757e+09  1.681714e+19  4.100870e+09  0.662931 -193.047739  Low & middle income  pop_total_RNN  7276.0
569  4.162898e+09  1.808172e+19  4.252261e+09  0.662358 -206.868824     IDA & IBRD total  pop_total_RNN  7297.0
705  5.032300e+09  2.640996e+19  5.139062e+09  0.675346 -260.293050                World  pop_total_RNN  7342.0
```


## Model LSTM - top states
```
             mae           mse         rmse      mape         r2             state           model    rank
763  3046.902727  1.444916e+07  3801.205723  0.042840 -10.495297          Dominica  pop_total_LSTM  1194.0
849  8248.166808  7.254629e+07  8517.410932  0.182829 -11.016572  Marshall Islands  pop_total_LSTM  1281.0
842  7530.540511  8.947733e+07  9459.245866  0.012406   0.630514        Luxembourg  pop_total_LSTM   581.0
```


## Model LSTM - worst states
```
              mae           mse          rmse      mape          r2                state           model    rank
839  3.841114e+09  1.514709e+19  3.891926e+09  0.635894 -173.777583  Low & middle income  pop_total_LSTM  7253.0
806  3.966845e+09  1.614740e+19  4.018383e+09  0.632595 -184.631760     IDA & IBRD total  pop_total_LSTM  7266.0
942  4.500498e+09  2.090932e+19  4.572671e+09  0.604699 -205.871196                World  pop_total_LSTM  7296.0
```


## Model GRU - top states
```
               mae           mse          rmse      mape        r2       state          model   rank
979   10109.157078  1.158565e+08  10763.663720  0.018101  0.528604  Cabo Verde  pop_total_GRU  652.0
970    9933.174031  1.606731e+08  12675.689390  0.013461  0.612633      Bhutan  pop_total_GRU  629.0
1083  11523.327401  1.919473e+08  13854.505992  0.023956  0.923613    Maldives  pop_total_GRU  457.0
```


## Model GRU - worst states
```
               mae           mse          rmse      mape          r2                state          model    rank
1076  3.067815e+09  9.852464e+18  3.138864e+09  0.506514 -112.684546  Low & middle income  pop_total_GRU  7163.0
1043  3.136976e+09  1.030297e+19  3.209824e+09  0.498918 -117.443716     IDA & IBRD total  pop_total_GRU  7172.0
1179  3.686880e+09  1.433620e+19  3.786318e+09  0.494201 -140.838567                World  pop_total_GRU  7215.0
```


## Model XGBoost - top states
```
               mae           mse          rmse      mape         r2    state              model    rank
1267  13637.218750  2.178701e+08  14760.422768  0.112703 -35.075328  Grenada  pop_total_XGBoost  1593.0
1256  13592.173077  2.279839e+08  15099.136071  0.014876  -4.635848     Fiji  pop_total_XGBoost  1233.0
1272  17215.677885  4.710676e+08  21704.092465  0.022327   0.004959   Guyana  pop_total_XGBoost   895.0
```


## Model XGBoost - worst states
```
               mae           mse          rmse      mape          r2                      state              model    rank
1279  5.188959e+08  3.647987e+17  6.039857e+08  0.109482  -12.312881                  IBRD only  pop_total_XGBoost  6573.0
1416  5.152972e+08  3.650493e+17  6.041931e+08  0.067946   -2.611700                      World  pop_total_XGBoost  6321.0
1302  5.110565e+08  3.758991e+17  6.131061e+08  0.223563 -144.340563  Late-demographic dividend  pop_total_XGBoost  7051.0
```


## Model random_forest - top states
```
             mae            mse        rmse      mape         r2                  state                    model    rank
1619  284.027086   84971.336237  291.498433  0.005952 -18.761003    St. Kitts and Nevis  pop_total_random_forest  1172.0
1564  720.434353  809719.254997  899.844017  0.006469   0.418086  Micronesia, Fed. Sts.  pop_total_random_forest   466.0
1651  897.487025  990582.174726  995.279948  0.008447  -4.554638  Virgin Islands (U.S.)  pop_total_random_forest   977.0
```


## Model random_forest - worst states
```
               mae           mse          rmse      mape         r2                state                    model    rank
1550  3.443941e+08  1.775536e+17  4.213711e+08  0.055598  -1.048736  Low & middle income  pop_total_random_forest  6108.0
1517  5.320381e+08  3.700508e+17  6.083180e+08  0.083248  -3.254132     IDA & IBRD total  pop_total_random_forest  6378.0
1653  1.693920e+09  2.970439e+18  1.723496e+09  0.227515 -28.388728                World  pop_total_random_forest  6874.0
```


## Model LightGBM - top states
```
               mae           mse          rmse      mape        r2   state               model   rank
1730   5186.000350  4.069716e+07   6379.432221  0.005685 -0.006049    Fiji  pop_total_LightGBM  719.0
1681  17820.530534  4.174443e+08  20431.454406  0.023854 -0.006417  Bhutan  pop_total_LightGBM  891.0
1746  15638.210936  6.046667e+08  24589.972140  0.019946 -0.277244  Guyana  pop_total_LightGBM  965.0
```


## Model LightGBM - worst states
```
               mae           mse          rmse      mape         r2          state               model    rank
1805  5.243486e+08  3.332808e+17  5.773048e+08  0.095726  -4.968132  Middle income  pop_total_LightGBM  6434.0
1753  8.122626e+08  8.029277e+17  8.960623e+08  0.172253 -28.301858      IBRD only  pop_total_LightGBM  6796.0
1890  8.702801e+08  8.584615e+17  9.265320e+08  0.115987  -7.493388          World  pop_total_LightGBM  6560.0
```


## Per target metrics - model comparision
```
             target                    model           mae           mse          rmse      mape            r2  rank
0  population_total          pop_total_ARIMA  1.795531e+06  3.121514e+13  2.270705e+06  0.020145     -2.120259     1
1  population_total         pop_total_ARIMAX  6.999138e+06  9.793895e+14  8.643719e+06  0.040141    -17.538239     2
2  population_total       pop_total_LightGBM  1.949421e+07  1.030385e+16  2.230181e+07  0.262730 -93312.489894     3
3  population_total        pop_total_XGBoost  1.939745e+07  8.165994e+15  2.316393e+07  0.130787 -19449.285323     4
4  population_total  pop_total_random_forest  2.542539e+07  1.756521e+16  2.905972e+07  0.059581   -111.414059     5
5  population_total            pop_total_GRU  1.531744e+08  2.789681e+17  1.574853e+08  0.299194  -4771.006174     6
6  population_total           pop_total_LSTM  1.948120e+08  4.230988e+17  1.980739e+08  0.350874  -1681.420387     7
7  population_total            pop_total_RNN  2.047126e+08  4.837588e+17  2.103042e+08  0.458597  -2289.759424     8
```


## Overall metrics - model comparision
```
            mae           mse          rmse      mape            r2                    model  rank
1  2.071033e+06  3.656528e+13  2.620776e+06  0.020286     -1.804440          pop_total_ARIMA   4.0
0  8.120780e+06  1.150009e+15  1.002770e+07  0.039378    -13.435932         pop_total_ARIMAX   8.0
5  2.265237e+07  9.597812e+15  2.705595e+07  0.120810 -16462.971860        pop_total_XGBoost  17.0
6  2.964558e+07  2.064097e+16  3.388443e+07  0.061969    -99.680314  pop_total_random_forest  18.0
7  2.278202e+07  1.211098e+16  2.606263e+07  0.225166 -78498.805717       pop_total_LightGBM  19.0
4  1.789344e+08  3.277970e+17  1.839712e+08  0.303337  -3713.213548            pop_total_GRU  24.0
3  2.272827e+08  4.971094e+17  2.310974e+08  0.369576  -1370.611547           pop_total_LSTM  25.0
2  2.387422e+08  5.683885e+17  2.452683e+08  0.471827  -1781.705767            pop_total_RNN  29.0
```


