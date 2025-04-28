
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
          mae       mse      rmse        r2             state           model   rank
320  0.003767  0.000019  0.004295  0.994600              Guam  ensemble-arima   13.0
280  0.001928  0.000006  0.002412  0.991057  Congo, Dem. Rep.  ensemble-arima    5.0
263  0.005991  0.000074  0.008611  0.986081            Brazil  ensemble-arima   41.0
415  0.004273  0.000033  0.005737  0.979801           Romania  ensemble-arima   21.0
398  0.025622  0.000967  0.031099  0.966139            Norway  ensemble-arima  543.0
```


## Model ensemble-arima - worst states
```
          mae       mse      rmse         r2        state           model    rank
469  0.053263  0.004620  0.067971 -24.972786  Yemen, Rep.  ensemble-arima  2592.0
342  0.058126  0.005069  0.071194 -26.183718  Isle of Man  ensemble-arima  2652.0
373  0.014878  0.000370  0.019233 -28.376462         Mali  ensemble-arima  1581.0
376  0.570173  0.536698  0.732596 -57.690231   Mauritania  ensemble-arima  5951.0
433  0.557886  0.474183  0.688609 -78.916658    Sri Lanka  ensemble-arima  5903.0
```


## Model ensemble-arimax - top states
```
          mae       mse      rmse        r2                                   state            model   rank
89   0.006682  0.000056  0.007497  0.969717  Heavily indebted poor countries (HIPC)  ensemble-arimax   48.0
21   0.024847  0.000866  0.029424  0.933091                                   Benin  ensemble-arimax  509.0
28   0.019038  0.000481  0.021927  0.921362                                Bulgaria  ensemble-arimax  352.0
211  0.033144  0.001769  0.042058  0.905815                             Timor-Leste  ensemble-arimax  812.0
212  0.023061  0.000651  0.025523  0.868914                                    Togo  ensemble-arimax  460.0
```


## Model ensemble-arimax - worst states
```
             mae           mse          rmse            r2             state            model    rank
185     0.335314  1.240065e-01      0.352146 -6.836711e+03            Serbia  ensemble-arimax  5298.0
43      2.310704  5.976625e+00      2.444714 -9.177274e+03  Congo, Dem. Rep.  ensemble-arimax  7484.0
147     9.135506  3.021440e+02     17.382289 -3.396187e+04           Moldova  ensemble-arimax  7567.0
62     16.842543  3.773152e+02     19.424602 -4.851146e+05           Eritrea  ensemble-arimax  7580.0
177  7074.918130  2.506504e+08  15831.941813 -1.901054e+08             Qatar  ensemble-arimax  7584.0
```


## Model simple-rnn - top states
```
          mae       mse      rmse        r2                     state       model    rank
628  0.148530  0.051484  0.226602  0.719811                     Nepal  simple-rnn  2814.0
687  0.258897  0.085501  0.292404  0.661520                     Tonga  simple-rnn  3287.0
597  0.291893  0.105845  0.324997  0.407740                   Lebanon  simple-rnn  3510.0
510  0.177515  0.047557  0.217675  0.351319  Central African Republic  simple-rnn  2963.0
488  0.244941  0.073460  0.256455 -0.174806              Bahamas, The  simple-rnn  3395.0
```


## Model simple-rnn - worst states
```
          mae       mse      rmse             r2         state       model    rank
665  1.129892  1.393907  1.176989  -15589.700665       Somalia  simple-rnn  7232.0
477  1.295529  1.787519  1.334538  -19385.976644       Albania  simple-rnn  7323.0
691  0.512589  0.285823  0.533545  -36526.345587  Turkmenistan  simple-rnn  5953.0
610  0.850555  0.767565  0.876024  -60954.608433          Mali  simple-rnn  6838.0
606  1.520662  2.573371  1.603308 -134784.904727    Madagascar  simple-rnn  7405.0
```


## Model base-lstm - top states
```
          mae       mse      rmse        r2                   state      model    rank
908  0.090292  0.012070  0.109759  0.835199     St. Kitts and Nevis  base-lstm  1888.0
717  0.020820  0.000571  0.023731  0.607413     Antigua and Barbuda  base-lstm   486.0
746  0.088861  0.010857  0.100051  0.559356  Caribbean small states  base-lstm  1919.0
892  0.166591  0.033130  0.174100  0.535894                   Samoa  base-lstm  2713.0
798  0.275713  0.097648  0.312464  0.316556                  Guyana  base-lstm  3485.0
```


## Model base-lstm - worst states
```
          mae       mse      rmse            r2         state      model    rank
896  0.377617  0.149433  0.377747  -8238.747480        Serbia  base-lstm  5426.0
902  0.980318  1.016774  1.002951 -11371.506222       Somalia  base-lstm  7031.0
847  0.657547  0.466126  0.666337 -37016.064513          Mali  base-lstm  6362.0
928  0.634090  0.406192  0.636414 -51909.152124  Turkmenistan  base-lstm  6266.0
843  1.098421  1.285408  1.133738 -67325.040906    Madagascar  base-lstm  7189.0
```


## Model base-gru - top states
```
           mae       mse      rmse        r2                state     model    rank
1145  0.090772  0.012535  0.111830  0.828857  St. Kitts and Nevis  base-gru  1923.0
1061  0.124120  0.026506  0.162785  0.716033          Korea, Rep.  base-gru  2439.0
964   0.161869  0.031441  0.176896  0.394671           Bangladesh  base-gru  2733.0
1035  0.249619  0.090119  0.285690  0.369246               Guyana  base-gru  3354.0
1089  0.089484  0.008937  0.094492  0.116233               Mexico  base-gru  1996.0
```


## Model base-gru - worst states
```
           mae       mse      rmse            r2         state     model    rank
1133  0.359803  0.149486  0.363376  -8241.663801        Serbia  base-gru  5400.0
1139  0.979671  1.001187  0.998386 -11197.172134       Somalia  base-gru  7021.0
1165  0.429913  0.191586  0.436612 -24483.099787  Turkmenistan  base-gru  5634.0
1084  0.688970  0.491688  0.700280 -39046.087497          Mali  base-gru  6444.0
1080  0.930812  0.939254  0.968685 -49194.477587    Madagascar  base-gru  6969.0
```


## Model xgboost - top states
```
           mae       mse      rmse        r2                        state    model    rank
1206  0.012078  0.000196  0.013986  0.984874                        Benin  xgboost   137.0
1389  0.018461  0.000578  0.024040  0.978821                     Suriname  xgboost   343.0
1198  0.020061  0.000884  0.029734  0.953393                   Azerbaijan  xgboost   457.0
1350  0.026254  0.000921  0.030343  0.952472  Pacific island small states  xgboost   542.0
1277  0.037486  0.003734  0.061062  0.929373         Hong Kong SAR, China  xgboost  1110.0
```


## Model xgboost - worst states
```
           mae       mse      rmse          r2      state    model    rank
1263  0.100300  0.014970  0.122337 -182.499000    Georgia  xgboost  3604.0
1224  0.077619  0.012497  0.111786 -198.527135      Chile  xgboost  3427.0
1309  0.347963  0.184390  0.429406 -219.575803    Lesotho  xgboost  5274.0
1325  0.312272  0.160626  0.400782 -251.390190  Mauritius  xgboost  5176.0
1404  0.127659  0.020328  0.142572 -412.961134    Ukraine  xgboost  3931.0
```


## Model rf - top states
```
           mae       mse      rmse        r2             state model   rank
1657  0.010893  0.000155  0.012447  0.978027            Israel    rf  110.0
1508  0.014587  0.000370  0.019246  0.964811     Guinea-Bissau    rf  265.0
1591  0.012272  0.000197  0.014027  0.957321          Paraguay    rf  151.0
1560  0.002733  0.000018  0.004202  0.952525  Marshall Islands    rf   24.0
1471  0.012713  0.000202  0.014219  0.936100            Cyprus    rf  168.0
```


## Model rf - worst states
```
           mae       mse      rmse          r2         state model    rank
1506  0.149327  0.033462  0.182928 -203.845014     Guatemala    rf  4161.0
1487  0.200613  0.056592  0.237891 -244.900222      Ethiopia    rf  4533.0
1613  0.176675  0.043124  0.207663 -481.336421       Somalia    rf  4422.0
1558  0.074328  0.007793  0.088280 -617.907904          Mali    rf  3326.0
1639  0.073285  0.006048  0.077768 -771.902298  Turkmenistan    rf  3231.0
```


## Model lightgbm - top states
```
           mae       mse      rmse        r2                        state     model    rank
1867  0.012221  0.000232  0.015228  0.963746                   Tajikistan  lightgbm   162.0
1742  0.014156  0.000235  0.015318  0.931563                         Guam  lightgbm   205.0
1877  0.013169  0.000234  0.015293  0.892944                       Uganda  lightgbm   205.0
1824  0.039501  0.002343  0.048401  0.879141  Pacific island small states  lightgbm   969.0
1839  0.045472  0.003008  0.054844  0.878141                       Rwanda  lightgbm  1121.0
```


## Model lightgbm - worst states
```
           mae       mse      rmse          r2            state     model    rank
1849  0.204583  0.054344  0.233118 -318.363799  Solomon Islands  lightgbm  4532.0
1737  0.156386  0.026614  0.163137 -325.215700          Georgia  lightgbm  4118.0
1768  0.997540  1.699446  1.303628 -361.282460           Jordan  lightgbm  7031.0
1795  0.073287  0.006037  0.077701 -478.463715             Mali  lightgbm  3186.0
1678  0.729577  0.675125  0.821660 -867.400151          Belgium  lightgbm  6555.0
```


## Per target metrics - model comparision
```
               target            model        mae           mse       rmse             r2  rank
0   population_female   ensemble-arima   0.163777  2.100040e-01   0.198509      -2.951926     3
1   population_female          xgboost   0.149675  1.055876e-01   0.185706     -10.355222     1
2   population_female         lightgbm   0.165890  1.383926e-01   0.195994     -13.530153     5
3   population_female               rf   0.218305  4.520139e-01   0.247069     -16.151870     7
4   population_female         base-gru   0.731105  1.326647e+00   0.761868    -906.950224    10
5   population_female        base-lstm   0.748053  1.102718e+00   0.774127   -1315.414691    11
6   population_female       simple-rnn   0.917243  1.546056e+00   0.952018   -1805.284201    14
7   population_female  ensemble-arimax  30.238622  1.057600e+06  67.287719 -804473.192254    15
8     population_male   ensemble-arima   0.164117  2.103296e-01   0.198891      -3.042964     4
9     population_male          xgboost   0.149958  1.052209e-01   0.185871     -10.560817     2
10    population_male         lightgbm   0.165890  1.383926e-01   0.195994     -13.530154     6
11    population_male               rf   0.218337  4.520309e-01   0.247076     -16.156741     8
12    population_male        base-lstm   0.636409  9.327216e-01   0.662322    -950.966584     9
13    population_male         base-gru   0.749724  1.316380e+00   0.779522   -1060.654022    12
14    population_male       simple-rnn   0.781984  1.418513e+00   0.829988   -1691.327587    13
15    population_male  ensemble-arimax  30.238622  1.057600e+06  67.287719 -804473.192247    16
```


## Overall metrics - model comparision
```
         mae           mse       rmse             r2            model  rank
5   0.152435  1.028627e-01   0.188660      -9.786589          xgboost   5.0
7   0.164130  1.358177e-01   0.193887     -10.233497         lightgbm   9.0
1   0.167004  2.205109e-01   0.202825      -3.034367   ensemble-arima  10.0
6   0.220588  4.598631e-01   0.249659     -15.561229               rf  16.0
3   0.703759  1.037625e+00   0.730663   -1072.765528        base-lstm  21.0
4   0.758816  1.368895e+00   0.790422    -962.848544         base-gru  23.0
2   0.855707  1.499688e+00   0.897700   -1724.578089       simple-rnn  28.0
0  29.555098  1.035961e+06  65.822687 -786680.232731  ensemble-arimax  32.0
```


