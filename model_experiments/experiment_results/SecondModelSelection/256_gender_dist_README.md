
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
          mae       mse      rmse        r2             state           model   rank
320  0.003767  0.000019  0.004295  0.994600              Guam  ensemble-arima   10.0
280  0.001928  0.000006  0.002412  0.991057  Congo, Dem. Rep.  ensemble-arima    5.0
263  0.005991  0.000074  0.008611  0.986081            Brazil  ensemble-arima   38.0
415  0.004273  0.000033  0.005737  0.979801           Romania  ensemble-arima   20.0
398  0.025622  0.000967  0.031099  0.966139            Norway  ensemble-arima  591.0
```


## Model ensemble-arima - worst states
```
          mae       mse      rmse         r2        state           model    rank
469  0.053263  0.004620  0.067971 -24.972786  Yemen, Rep.  ensemble-arima  2596.0
342  0.058126  0.005069  0.071194 -26.183718  Isle of Man  ensemble-arima  2659.0
373  0.014878  0.000370  0.019233 -28.376462         Mali  ensemble-arima  1577.0
376  0.570173  0.536698  0.732596 -57.690231   Mauritania  ensemble-arima  5591.0
433  0.557886  0.474183  0.688609 -78.916658    Sri Lanka  ensemble-arima  5584.0
```


## Model ensemble-arimax - top states
```
          mae       mse      rmse        r2                                   state            model   rank
89   0.006682  0.000056  0.007497  0.969717  Heavily indebted poor countries (HIPC)  ensemble-arimax   44.0
21   0.024847  0.000866  0.029424  0.933091                                   Benin  ensemble-arimax  560.0
28   0.019038  0.000481  0.021927  0.921362                                Bulgaria  ensemble-arimax  381.0
211  0.033144  0.001769  0.042058  0.905815                             Timor-Leste  ensemble-arimax  869.0
212  0.023061  0.000651  0.025523  0.868914                                    Togo  ensemble-arimax  502.0
```


## Model ensemble-arimax - worst states
```
             mae           mse          rmse            r2             state            model    rank
185     0.335314  1.240065e-01      0.352146 -6.836711e+03            Serbia  ensemble-arimax  5292.0
43      2.310704  5.976625e+00      2.444714 -9.177274e+03  Congo, Dem. Rep.  ensemble-arimax  6828.0
147     9.135506  3.021440e+02     17.382289 -3.396187e+04           Moldova  ensemble-arimax  7502.0
62     16.842543  3.773152e+02     19.424602 -4.851146e+05           Eritrea  ensemble-arimax  7559.0
177  7074.918130  2.506504e+08  15831.941813 -1.901054e+08             Qatar  ensemble-arimax  7584.0
```


## Model simple-rnn - top states
```
          mae       mse      rmse        r2                state       model    rank
628  0.335449  0.149496  0.369106  0.186406                Nepal  simple-rnn  3852.0
537  0.191222  0.050117  0.216894 -0.390036              Estonia  simple-rnn  3322.0
687  0.821140  1.037928  0.985092 -3.108915                Tonga  simple-rnn  5276.0
671  0.479740  0.326870  0.534622 -3.462869  St. Kitts and Nevis  simple-rnn  4903.0
567  0.268024  0.098493  0.292686 -4.453033              Hungary  simple-rnn  4258.0
```


## Model simple-rnn - worst states
```
          mae         mse       rmse            r2         state       model    rank
665  5.756642   57.490053   7.462683 -6.430192e+05       Somalia  simple-rnn  7343.0
691  2.123947    6.404153   2.466497 -8.184305e+05  Turkmenistan  simple-rnn  6927.0
610  2.829761   13.172787   3.546764 -1.046107e+06          Mali  simple-rnn  7125.0
513  8.467512  120.378910  10.807032 -1.922041e+06         Chile  simple-rnn  7507.0
606  6.163154   64.566951   7.905466 -3.381833e+06    Madagascar  simple-rnn  7380.0
```


## Model base-lstm - top states
```
          mae       mse      rmse        r2                  state      model    rank
926  0.029549  0.001268  0.035066  0.933835                Tunisia  base-lstm   713.0
778  0.041325  0.002120  0.045238  0.728703  Europe & Central Asia  base-lstm  1070.0
815  0.036918  0.001947  0.043597  0.694951                Ireland  base-lstm  1006.0
801  0.059638  0.005155  0.069909  0.619474            High income  base-lstm  1539.0
947  0.173922  0.038954  0.197219  0.603743            New Zealand  base-lstm  2920.0
```


## Model base-lstm - worst states
```
          mae       mse      rmse            r2       state      model    rank
847  0.176733  0.047046  0.177464  -3735.140178        Mali  base-lstm  4508.0
896  0.246189  0.068047  0.248152  -3751.105003      Serbia  base-lstm  4862.0
902  0.627235  0.409435  0.636403  -4578.487722     Somalia  base-lstm  5923.0
750  0.773975  0.701541  0.836610 -11200.230270       Chile  base-lstm  6143.0
843  0.545365  0.330828  0.569573 -17326.837022  Madagascar  base-lstm  5888.0
```


## Model base-gru - top states
```
           mae       mse      rmse        r2       state     model    rank
1083  0.906871  1.126866  1.018748  0.625112    Maldives  base-gru  4599.0
1085  0.866126  0.843383  0.914931 -0.499203       Malta  base-gru  4860.0
1136  0.247548  0.078763  0.265205 -0.510909    Slovenia  base-gru  3627.0
1058  0.300618  0.133947  0.330952 -1.136243  Kazakhstan  base-gru  4079.0
961   0.210233  0.062016  0.224605 -2.269187  Azerbaijan  base-gru  3777.0
```


## Model base-gru - worst states
```
           mae       mse      rmse             r2        state     model    rank
992   2.387073  6.839070  2.612111  -38675.639896  Congo, Rep.  base-gru  6952.0
1139  1.641713  3.844222  1.954931  -42996.214184      Somalia  base-gru  6731.0
987   2.369686  7.316605  2.704314 -116820.306254        Chile  base-gru  6986.0
1080  1.874286  4.423568  2.095402 -231692.984325   Madagascar  base-gru  6813.0
1084  1.716988  3.740742  1.928586 -297067.398003         Mali  base-gru  6766.0
```


## Model xgboost - top states
```
           mae       mse      rmse        r2                        state    model    rank
1206  0.012078  0.000196  0.013986  0.984874                        Benin  xgboost   179.0
1389  0.018461  0.000578  0.024040  0.978821                     Suriname  xgboost   377.0
1198  0.020061  0.000884  0.029734  0.953393                   Azerbaijan  xgboost   501.0
1350  0.026254  0.000921  0.030343  0.952472  Pacific island small states  xgboost   586.0
1277  0.037486  0.003734  0.061062  0.929373         Hong Kong SAR, China  xgboost  1163.0
```


## Model xgboost - worst states
```
           mae       mse      rmse          r2      state    model    rank
1263  0.100300  0.014970  0.122337 -182.499000    Georgia  xgboost  3545.0
1224  0.077619  0.012497  0.111786 -198.527135      Chile  xgboost  3364.0
1309  0.347963  0.184390  0.429406 -219.575803    Lesotho  xgboost  5187.0
1325  0.312272  0.160626  0.400782 -251.390190  Mauritius  xgboost  5106.0
1404  0.127659  0.020328  0.142572 -412.961134    Ukraine  xgboost  3869.0
```


## Model rf - top states
```
           mae       mse      rmse        r2       state model   rank
1630  0.007097  0.000067  0.008180  0.989534  Tajikistan    rf   45.0
1591  0.008154  0.000085  0.009221  0.981554    Paraguay    rf   72.0
1657  0.011260  0.000174  0.013201  0.975280      Israel    rf  164.0
1547  0.008970  0.000131  0.011431  0.974038     Liberia    rf  120.0
1588  0.015634  0.000368  0.019182  0.965004    Pakistan    rf  314.0
```


## Model rf - worst states
```
           mae       mse      rmse          r2        state model    rank
1527  0.087759  0.012241  0.110638  -64.650014  Isle of Man    rf  3288.0
1562  0.159896  0.042786  0.206845  -66.228447    Mauritius    rf  4160.0
1480  0.115446  0.023165  0.152200  -67.162244      Ecuador    rf  3711.0
1613  0.071194  0.008241  0.090777  -91.169561      Somalia    rf  3088.0
1554  0.037824  0.002331  0.048249 -121.079152   Madagascar    rf  2415.0
```


## Model lightgbm - top states
```
           mae       mse      rmse        r2                        state     model    rank
1867  0.012221  0.000232  0.015228  0.963746                   Tajikistan  lightgbm   208.0
1742  0.014156  0.000235  0.015318  0.931563                         Guam  lightgbm   253.0
1877  0.013169  0.000234  0.015293  0.892944                       Uganda  lightgbm   255.0
1824  0.039501  0.002343  0.048401  0.879141  Pacific island small states  lightgbm  1029.0
1839  0.045472  0.003008  0.054844  0.878141                       Rwanda  lightgbm  1168.0
```


## Model lightgbm - worst states
```
           mae       mse      rmse          r2            state     model    rank
1849  0.204583  0.054344  0.233118 -318.363799  Solomon Islands  lightgbm  4486.0
1737  0.156386  0.026614  0.163137 -325.215700          Georgia  lightgbm  4052.0
1768  0.997540  1.699446  1.303628 -361.282460           Jordan  lightgbm  6070.0
1795  0.073287  0.006037  0.077701 -478.463715             Mali  lightgbm  3102.0
1678  0.729577  0.675125  0.821660 -867.400151          Belgium  lightgbm  5949.0
```


## Per target metrics - model comparision
```
               target            model        mae           mse       rmse             r2  rank
0   population_female   ensemble-arima   0.163777  2.100040e-01   0.198509      -2.951926     5
1   population_female               rf   0.141921  1.240689e-01   0.174462      -4.918993     2
2   population_female          xgboost   0.149675  1.055876e-01   0.185706     -10.355222     3
3   population_female         lightgbm   0.165890  1.383926e-01   0.195994     -13.530153     7
4   population_female        base-lstm   0.452214  7.281156e-01   0.480557    -267.453708    10
5   population_female         base-gru   1.663351  9.691710e+00   1.885242   -5923.004518    12
6   population_female       simple-rnn   5.158517  6.818797e+01   6.271131  -72345.529588    14
7   population_female  ensemble-arimax  30.238622  1.057600e+06  67.287719 -804473.192254    15
8     population_male   ensemble-arima   0.164117  2.103296e-01   0.198891      -3.042964     6
9     population_male               rf   0.141808  1.245555e-01   0.174073      -4.988188     1
10    population_male          xgboost   0.149958  1.052209e-01   0.185871     -10.560817     4
11    population_male         lightgbm   0.165890  1.383926e-01   0.195994     -13.530154     8
12    population_male        base-lstm   0.436859  7.059586e-01   0.463077    -296.984026     9
13    population_male         base-gru   1.478441  4.351191e+00   1.657056   -4885.468194    11
14    population_male       simple-rnn   3.190430  2.924660e+01   4.144942  -33395.122818    13
15    population_male  ensemble-arimax  30.238622  1.057600e+06  67.287719 -804473.192247    16
```


## Overall metrics - model comparision
```
         mae           mse       rmse             r2            model  rank
6   0.144907  1.268835e-01   0.177776      -5.276541               rf   6.0
5   0.152435  1.028627e-01   0.188660      -9.786589          xgboost   8.0
1   0.167004  2.205109e-01   0.202825      -3.034367   ensemble-arima  13.0
7   0.164130  1.358177e-01   0.193887     -10.233497         lightgbm  13.0
3   0.448173  7.334177e-01   0.476716    -281.161207        base-lstm  20.0
4   1.675642  7.330938e+00   1.894493   -5942.166687         base-gru  24.0
2   4.530162  5.363885e+01   5.675663  -58579.641072       simple-rnn  28.0
0  29.555098  1.035961e+06  65.822687 -786680.232731  ensemble-arimax  32.0
```


