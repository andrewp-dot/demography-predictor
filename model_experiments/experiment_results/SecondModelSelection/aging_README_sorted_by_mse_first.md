
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arimas - top states
```
          mae       mse      rmse        r2                                     state            model   rank
160  0.112085  0.020175  0.137482  0.813108                           North Macedonia  ensemble-arimas  154.0
190  0.102024  0.021371  0.121550  0.655303                           Solomon Islands  ensemble-arimas  220.0
229  0.147458  0.036931  0.176735 -0.532786                     Virgin Islands (U.S.)  ensemble-arimas  664.0
80   0.148184  0.039338  0.180412  0.780259                                     Ghana  ensemble-arimas  284.0
73   0.148041  0.049473  0.209375 -1.405051  Fragile and conflict affected situations  ensemble-arimas  866.0
```


## Model ensemble-arimas - worst states
```
           mae          mse       rmse             r2                 state            model    rank
114   4.518803    52.187029   5.960727     -22.920064                Kuwait  ensemble-arimas  5354.0
31    4.189480    77.135096   6.048098     -60.249011            Cabo Verde  ensemble-arimas  5535.0
135   6.240273    80.861029   7.440831     -78.892305              Maldives  ensemble-arimas  5728.0
92   12.478125   238.507282  14.261997    -823.207273  Hong Kong SAR, China  ensemble-arimas  6481.0
62   67.007888  8864.620110  77.321071 -255384.632098               Eritrea  ensemble-arimas  6636.0
```


## Model simple-rnn - top states
```
          mae       mse      rmse            r2        state       model    rank
299  0.843782  1.025519  0.942063   -155.828932      Eritrea  simple-rnn  3830.0
320  1.442920  2.442415  1.547646    -29.431753         Guam  simple-rnn  4161.0
237  1.461476  3.391078  1.705216 -32004.663373  Afghanistan  simple-rnn  4793.0
347  1.990135  5.937243  2.336374    -25.339518   Kazakhstan  simple-rnn  4533.0
448  1.856501  6.026997  2.218207   -789.180554  Timor-Leste  simple-rnn  4912.0
```


## Model simple-rnn - worst states
```
           mae         mse       rmse           r2                 state       model    rank
444  12.443364  233.585676  14.611289  -150.667952  Syrian Arab Republic  simple-rnn  6304.0
303  12.677518  233.947615  14.341630 -3482.282484             Euro area  simple-rnn  6566.0
316  15.123380  333.257588  17.106730 -5366.760684               Germany  simple-rnn  6584.0
343  15.318307  343.819428  17.333729 -1217.688592                 Italy  simple-rnn  6526.0
345  17.748988  463.721483  20.057189 -1429.196733                 Japan  simple-rnn  6540.0
```


## Model base-lstm - top states
```
          mae       mse      rmse         r2                        state      model    rank
625  0.083623  0.008793  0.091698 -17.834035                   Mozambique  base-lstm  1087.0
687  0.111919  0.016308  0.124670   0.618061                        Tonga  base-lstm   244.0
475  0.159075  0.045776  0.169261   0.757209  Africa Eastern and Southern  base-lstm   321.0
571  0.215690  0.049737  0.217845   0.469045                     IDA only  base-lstm   608.0
520  0.194461  0.055428  0.222107  -0.834538                Cote d'Ivoire  base-lstm   893.0
```


## Model base-lstm - worst states
```
          mae        mse      rmse           r2                 state      model    rank
622  4.415307  33.287517  4.751145   -17.930161              Mongolia  base-lstm  5231.0
597  4.720172  33.529508  5.008600   -17.612967               Lebanon  base-lstm  5260.0
489  5.673261  56.755678  6.167536 -1092.859675               Bahrain  base-lstm  5878.0
694  7.073491  82.285968  7.515355  -670.785322  United Arab Emirates  base-lstm  6045.0
651  7.438682  93.867382  7.990421  -231.570807                 Qatar  base-lstm  6005.0
```


## Model base-gru - top states
```
          mae       mse      rmse         r2                                         state     model    rank
740  0.104600  0.016074  0.108846  -2.097880                                  Burkina Faso  base-gru   744.0
833  0.153535  0.043267  0.196076  -0.493989  Least developed countries: UN classification  base-gru   705.0
773  0.205828  0.047769  0.211976  -3.007769                                       Eritrea  base-gru  1107.0
912  0.194163  0.049774  0.204516 -14.087392     Sub-Saharan Africa (IDA & IBRD countries)  base-gru  1353.5
911  0.194163  0.049774  0.204516 -14.087392                            Sub-Saharan Africa  base-gru  1356.5
```


## Model base-gru - worst states
```
           mae         mse       rmse            r2     state     model    rank
819   8.277390  111.117359   9.227575    -32.172213     Japan  base-gru  5834.0
921   7.828381  111.789159   9.125206   -144.831023  Thailand  base-gru  6025.0
790   9.025807  121.502653  10.338476   -796.991974   Germany  base-gru  6312.0
804   9.120758  125.859793  10.363454 -18870.516379   Hungary  base-gru  6462.0
762  10.754914  175.500105  12.275610   -398.296231   Denmark  base-gru  6387.0
```


## Model xgboost - top states
```
           mae       mse      rmse        r2                     state    model   rank
1156  0.066031  0.007102  0.079711 -1.450954                Tajikistan  xgboost  617.0
1097  0.051121  0.007839  0.079051 -7.272347                Montenegro  xgboost  899.0
1127  0.065874  0.009473  0.088145  0.758235        Russian Federation  xgboost  105.0
1033  0.095355  0.014013  0.117512  0.632408                    Guinea  xgboost  212.0
1123  0.093337  0.014693  0.111433 -0.365955  Pre-demographic dividend  xgboost  484.0
```


## Model xgboost - worst states
```
           mae        mse      rmse         r2                state    model    rank
1050  1.460149   6.203216  1.931999  -3.283813   Iran, Islamic Rep.  xgboost  4038.0
988   1.520549   7.230104  1.837023  -3.119486                China  xgboost  4057.0
1162  1.541107   7.447862  1.868952  -7.004887  Trinidad and Tobago  xgboost  4203.0
963   1.777455   7.545408  2.179129 -58.053500              Bahrain  xgboost  4637.0
1056  2.583080  12.315211  3.062156  -3.672046                Japan  xgboost  4565.0
```


## Model rf - top states
```
           mae       mse      rmse        r2        state model   rank
1185  0.027077  0.001560  0.034645 -0.575492  Afghanistan    rf  469.0
1247  0.058249  0.006621  0.069778  0.814293      Eritrea    rf   67.0
1398  0.084629  0.010535  0.100873  0.707828        Tonga    rf  153.0
1393  0.081071  0.012560  0.105733 -1.393573   Tajikistan    rf  630.0
1370  0.079079  0.012842  0.109640  0.405791       Serbia    rf  261.0
```


## Model rf - worst states
```
           mae        mse      rmse         r2                 state model    rank
1200  1.992615  10.417204  2.266193 -57.110423               Bahrain    rf  4752.0
1293  2.405600  10.667565  2.914797  -2.505650                 Japan    rf  4410.0
1392  2.497899  12.397835  3.055020  -1.817238  Syrian Arab Republic    rf  4414.0
1287  2.634478  20.368525  3.485656 -12.042372    Iran, Islamic Rep.    rf  4891.0
1225  3.629814  36.317781  4.245979 -28.340814                 China    rf  5210.0
```


## Model lightgbm - top states
```
           mae       mse      rmse        r2                                   state     model   rank
1422  0.037956  0.002790  0.044733 -1.929763                             Afghanistan  lightgbm  670.0
1573  0.046013  0.003561  0.052323  0.557443                              Mozambique  lightgbm  186.0
1484  0.067160  0.007975  0.079998  0.413726                                 Eritrea  lightgbm  238.0
1424  0.071319  0.008094  0.089623 -1.599079              Africa Western and Central  lightgbm  653.0
1511  0.084769  0.012580  0.100289  0.730421  Heavily indebted poor countries (HIPC)  lightgbm  137.0
```


## Model lightgbm - worst states
```
           mae        mse      rmse        r2                 state     model    rank
1524  1.579758   7.166475  2.075493 -3.501681    Iran, Islamic Rep.  lightgbm  4145.0
1462  1.538029   7.462555  1.892982 -4.414601                 China  lightgbm  4137.0
1636  1.702838   7.508845  2.090165 -7.298901   Trinidad and Tobago  lightgbm  4288.0
1629  2.375887  11.283122  2.912794 -1.573377  Syrian Arab Republic  lightgbm  4335.0
1530  3.078361  15.983221  3.463545 -4.558224                 Japan  lightgbm  4749.0
```


## Per target metrics - model comparision
```
                          target            model       mae         mse      rmse           r2  rank
0           population_ages_0-14         lightgbm  0.449832    0.505705  0.533965    -2.089682     4
1           population_ages_0-14          xgboost  0.478217    0.667427  0.606794    -2.497383     5
2           population_ages_0-14               rf  0.495484    0.696867  0.607374    -4.192226     6
3           population_ages_0-14        base-lstm  2.241070    9.402685  2.400987  -181.185117    16
4           population_ages_0-14         base-gru  3.425102   24.526580  3.995410  -645.639147    17
5           population_ages_0-14  ensemble-arimas  1.674727   60.287081  1.977694 -1127.636009    13
6           population_ages_0-14       simple-rnn  7.395302  107.437462  9.106146 -3323.398015    21
7          population_ages_15-64          xgboost  0.544665    1.033470  0.687576    -3.305675     7
8          population_ages_15-64         lightgbm  0.606528    1.126280  0.735565    -4.340259     8
9          population_ages_15-64               rf  0.692411    1.807058  0.841045    -8.369454    10
10         population_ages_15-64        base-lstm  2.092871    7.891896  2.263884   -42.320253    15
11         population_ages_15-64         base-gru  4.502130   42.637104  5.132958  -218.564025    18
12         population_ages_15-64  ensemble-arimas  1.892892   64.122585  2.241989 -2217.177694    14
13         population_ages_15-64       simple-rnn  7.017004   98.061145  8.848335  -896.400219    20
14  population_ages_65_and_above         lightgbm  0.297900    0.377071  0.359557    -4.307343     2
15  population_ages_65_and_above               rf  0.317113    0.379437  0.395173    -1.134953     3
16  population_ages_65_and_above          xgboost  0.295210    0.380312  0.373121    -4.008964     1
17  population_ages_65_and_above  ensemble-arimas  0.629898    1.256158  0.761840   -11.284446     9
18  population_ages_65_and_above        base-lstm  0.925757    1.712699  0.963143   -16.642477    11
19  population_ages_65_and_above         base-gru  1.619219    7.327652  1.730971   -72.612345    12
20  population_ages_65_and_above       simple-rnn  5.879421   56.801457  7.064451 -5599.659126    19
```


## Overall metrics - model comparision
```
        mae        mse      rmse           r2            model  rank
4  0.469907   0.773085  0.594448    -3.162772          xgboost   6.0
6  0.474313   0.730706  0.571284    -3.477708         lightgbm   6.0
5  0.527944   1.059187  0.647563    -4.679068               rf  12.0
2  1.714147   6.136379  1.843400   -65.930570        base-lstm  18.0
0  1.195244  17.666922  1.427050  -430.936536  ensemble-arimas  19.0
3  3.237727  26.113308  3.694253  -300.603000         base-gru  23.0
1  7.200624  96.417500  8.915308 -3037.640703       simple-rnn  28.0
```


