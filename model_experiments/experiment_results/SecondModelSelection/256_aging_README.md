
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
          mae       mse      rmse        r2               state           model    rank
259  0.277230  0.136327  0.317043  0.949151              Bhutan  ensemble-arima  1022.0
338  0.043808  0.004980  0.062891  0.946293           Indonesia  ensemble-arima    18.0
295  0.178961  0.046040  0.212277  0.943811             Ecuador  ensemble-arima   414.0
426  0.098822  0.019187  0.120513  0.942912        Small states  ensemble-arima   103.0
467  0.066466  0.008926  0.080386  0.917491  West Bank and Gaza  ensemble-arima    47.0
```


## Model ensemble-arima - worst states
```
          mae       mse      rmse         r2                       state           model    rank
361  0.424703  0.281701  0.521969 -61.087523                     Lesotho  ensemble-arima  3281.0
330  1.272058  2.823852  1.528139 -61.409078                     Hungary  ensemble-arima  5141.0
296  1.111168  2.110862  1.320316 -67.254335            Egypt, Arab Rep.  ensemble-arima  4974.0
380  1.438625  3.477969  1.680949 -68.918634  Middle East & North Africa  ensemble-arima  5259.0
265  0.523723  0.558305  0.663370 -86.225393                    Bulgaria  ensemble-arima  3817.0
```


## Model ensemble-arimax - top states
```
          mae       mse      rmse        r2             state            model    rank
168  0.204322  0.095617  0.271720  0.858752  Papua New Guinea  ensemble-arimax   768.0
22   0.555398  0.548058  0.622878  0.822267            Bhutan  ensemble-arimax  2365.0
160  0.112085  0.020175  0.137482  0.813108   North Macedonia  ensemble-arimax   188.0
14   0.495715  0.389683  0.601109  0.795908      Bahamas, The  ensemble-arimax  2189.0
80   0.148184  0.039338  0.180412  0.780259             Ghana  ensemble-arimax   361.0
```


## Model ensemble-arimax - worst states
```
           mae          mse       rmse             r2                 state            model    rank
153   1.719036     5.118961   1.923757    -667.830412               Namibia  ensemble-arimax  5721.0
208   1.117657     2.393342   1.300861    -823.036269            Tajikistan  ensemble-arimax  5290.0
92   12.478125   238.507282  14.261997    -823.207273  Hong Kong SAR, China  ensemble-arimax  7429.0
43    2.991219    13.049206   3.144580   -1357.230961      Congo, Dem. Rep.  ensemble-arimax  6354.0
62   67.007888  8864.620110  77.321071 -255384.632098               Eritrea  ensemble-arimax  7584.0
```


## Model simple-rnn - top states
```
          mae        mse      rmse         r2        state       model    rank
484  2.799639  12.716601  3.239450 -15.970624        Aruba  simple-rnn  5784.0
584  1.990135   5.937243  2.336374 -25.339518   Kazakhstan  simple-rnn  5449.0
557  1.442920   2.442415  1.547646 -29.431753         Guam  simple-rnn  5034.0
597  5.973790  50.102419  6.891956 -30.058211      Lebanon  simple-rnn  6418.0
650  7.078946  88.906200  8.667938 -32.007432  Puerto Rico  simple-rnn  6650.0
```


## Model simple-rnn - worst states
```
          mae         mse       rmse            r2                     state       model    rank
649  5.927769   65.100409   7.915088 -25922.801003  Pre-demographic dividend  simple-rnn  7059.0
474  1.461476    3.391078   1.705216 -32004.663373               Afghanistan  simple-rnn  5697.0
567  8.873892  121.119185  10.110457 -41986.988650                   Hungary  simple-rnn  7386.0
502  9.659976  145.280006  11.120135 -50784.058387                  Bulgaria  simple-rnn  7470.0
664  4.462108   32.908075   5.560211 -76073.471840           Solomon Islands  simple-rnn  6823.0
```


## Model base-lstm - top states
```
          mae       mse      rmse        r2                        state      model    rank
802  0.625284  0.775522  0.733402  0.783189                     Honduras  base-lstm  2702.0
822  0.500984  0.372783  0.513870  0.761831                        Kenya  base-lstm  2094.0
712  0.159075  0.045776  0.169261  0.757209  Africa Eastern and Southern  base-lstm   398.0
795  0.242714  0.101586  0.300067  0.750263                    Guatemala  base-lstm   964.0
731  0.862102  1.314382  0.930519  0.745943                       Belize  base-lstm  3168.0
```


## Model base-lstm - worst states
```
          mae        mse      rmse           r2                   state      model    rank
919  1.371751   2.977583  1.445154  -840.833563              Tajikistan  base-lstm  5467.0
739  3.144168  12.412346  3.226206 -1069.451337                Bulgaria  base-lstm  6353.0
726  5.673261  56.755678  6.167536 -1092.859675                 Bahrain  base-lstm  6826.0
804  3.012525  10.625626  3.150925 -1478.576278                 Hungary  base-lstm  6333.0
735  2.534330   7.264703  2.604222 -1687.278979  Bosnia and Herzegovina  base-lstm  6128.0
```


## Model base-gru - top states
```
           mae       mse      rmse        r2             state     model    rank
1080  0.206136  0.080002  0.259790  0.822334        Madagascar  base-gru   720.0
1086  0.342700  0.160725  0.398627  0.715205  Marshall Islands  base-gru  1419.0
1166  0.341163  0.266974  0.432738  0.666042            Uganda  base-gru  1669.0
1032  0.939999  1.862764  1.142913  0.466570         Guatemala  base-gru  3556.0
1059  0.841171  0.977597  0.877726  0.284029             Kenya  base-gru  3237.0
```


## Model base-gru - worst states
```
           mae         mse       rmse            r2                   state     model    rank
1167  5.786114   47.757761   6.421652  -2146.119829                 Ukraine  base-gru  6870.0
1169  8.252896  102.208117   9.431328  -2658.622755          United Kingdom  base-gru  7234.0
972   4.083115   23.373296   4.537570  -2762.538154  Bosnia and Herzegovina  base-gru  6653.0
976   6.659659   61.753720   7.386348  -4047.116681                Bulgaria  base-gru  7026.0
1041  9.120758  125.859793  10.363454 -18870.516379                 Hungary  base-gru  7410.0
```


## Model xgboost - top states
```
           mae       mse      rmse        r2                           state    model    rank
1277  0.095492  0.018289  0.123232  0.963380            Hong Kong SAR, China  xgboost    96.0
1276  0.221089  0.108515  0.294814  0.961653                        Honduras  xgboost   834.0
1412  0.120551  0.026937  0.147502  0.934830                   Venezuela, RB  xgboost   177.0
1222  0.150813  0.039619  0.174325  0.883901  Central Europe and the Baltics  xgboost   305.0
1269  0.253591  0.170390  0.322330  0.877268                       Guatemala  xgboost  1075.0
```


## Model xgboost - worst states
```
           mae       mse      rmse          r2                   state    model    rank
1322  1.093151  2.555874  1.432657  -33.725124                   Malta  xgboost  4941.0
1209  0.322165  0.172161  0.396504  -42.527709  Bosnia and Herzegovina  xgboost  2680.0
1368  1.335802  3.546833  1.679952  -44.617259            Saudi Arabia  xgboost  5189.0
1200  1.777455  7.545408  2.179129  -58.053500                 Bahrain  xgboost  5559.0
1375  0.174843  0.067988  0.213433 -125.152021         Solomon Islands  xgboost  2019.0
```


## Model rf - top states
```
           mae       mse      rmse        r2                                              state model   rank
1506  0.109704  0.028880  0.141646  0.974105                                          Guatemala    rf  144.0
1541  0.163932  0.055099  0.230950  0.937356  Latin America & Caribbean (excluding high income)    rf  442.0
1586  0.081723  0.016053  0.111785  0.935398                                 Other small states    rf   79.0
1542  0.161665  0.059991  0.237381  0.935179  Latin America & the Caribbean (IDA & IBRD coun...    rf  467.0
1540  0.184547  0.081429  0.254311  0.929689                          Latin America & Caribbean    rf  634.0
```


## Model rf - worst states
```
           mae        mse      rmse          r2             state model    rank
1517  0.556141   0.600179  0.686981  -51.302552  IDA & IBRD total    rf  3858.0
1437  1.992615  10.417204  2.266193  -57.110423           Bahrain    rf  5680.0
1481  1.086571   2.120928  1.265630  -74.430640  Egypt, Arab Rep.    rf  4944.0
1575  0.555378   0.749856  0.722434  -98.004281           Namibia    rf  4037.0
1429  0.830834   1.239253  0.952534 -120.584654        Arab World    rf  4573.0
```


## Model lightgbm - top states
```
           mae       mse      rmse        r2                                         state     model   rank
1743  0.215374  0.091489  0.251068  0.967919                                     Guatemala  lightgbm  689.0
1886  0.103952  0.020413  0.123916  0.961818                                 Venezuela, RB  lightgbm  111.0
1882  0.156949  0.034645  0.184393  0.910487                           Upper middle income  lightgbm  301.0
1781  0.120407  0.025290  0.141626  0.903993  Least developed countries: UN classification  lightgbm  172.0
1665  0.155779  0.051363  0.199692  0.900555                           Antigua and Barbuda  lightgbm  380.0
```


## Model lightgbm - worst states
```
           mae       mse      rmse          r2                     state     model    rank
1718  0.736650  1.024884  0.827439  -32.522989          Egypt, Arab Rep.  lightgbm  4216.0
1885  0.449947  0.283634  0.487302  -38.971739                   Vanuatu  lightgbm  3210.0
1834  0.269468  0.100348  0.311773  -60.665860  Pre-demographic dividend  lightgbm  2378.0
1666  0.528032  0.547365  0.586589  -64.123434                Arab World  lightgbm  3699.0
1849  0.151645  0.035491  0.180948 -142.246271           Solomon Islands  lightgbm  1846.0
```


## Per target metrics - model comparision
```
                          target            model       mae         mse      rmse           r2  rank
0           population_ages_0-14         lightgbm  0.449832    0.505705  0.533965    -2.089682     5
1           population_ages_0-14          xgboost  0.478217    0.667427  0.606794    -2.497383     6
2           population_ages_0-14               rf  0.495484    0.696867  0.607374    -4.192226     7
3           population_ages_0-14   ensemble-arima  0.656464    1.359651  0.814877    -6.946118    11
4           population_ages_0-14        base-lstm  2.241070    9.402685  2.400987  -181.185117    19
5           population_ages_0-14         base-gru  3.425102   24.526580  3.995410  -645.639147    20
6           population_ages_0-14  ensemble-arimax  1.674727   60.287081  1.977694 -1127.636009    16
7           population_ages_0-14       simple-rnn  7.395302  107.437462  9.106146 -3323.398015    24
8          population_ages_15-64          xgboost  0.544665    1.033470  0.687576    -3.305675     8
9          population_ages_15-64         lightgbm  0.606528    1.126280  0.735565    -4.340259     9
10         population_ages_15-64   ensemble-arima  0.834058    2.029728  1.028809    -5.558373    13
11         population_ages_15-64               rf  0.692411    1.807058  0.841045    -8.369454    12
12         population_ages_15-64        base-lstm  2.092871    7.891896  2.263884   -42.320253    18
13         population_ages_15-64         base-gru  4.502130   42.637104  5.132958  -218.564025    21
14         population_ages_15-64       simple-rnn  7.017004   98.061145  8.848335  -896.400219    23
15         population_ages_15-64  ensemble-arimax  1.892892   64.122585  2.241989 -2217.177694    17
16  population_ages_65_and_above               rf  0.317113    0.379437  0.395173    -1.134953     3
17  population_ages_65_and_above   ensemble-arima  0.386520    0.509008  0.489265    -3.444971     4
18  population_ages_65_and_above          xgboost  0.295210    0.380312  0.373121    -4.008964     1
19  population_ages_65_and_above         lightgbm  0.297900    0.377071  0.359557    -4.307343     2
20  population_ages_65_and_above  ensemble-arimax  0.629898    1.256158  0.761840   -11.284446    10
21  population_ages_65_and_above        base-lstm  0.925757    1.712699  0.963143   -16.642477    14
22  population_ages_65_and_above         base-gru  1.619219    7.327652  1.730971   -72.612345    15
23  population_ages_65_and_above       simple-rnn  5.879421   56.801457  7.064451 -5599.659126    22
```


## Overall metrics - model comparision
```
        mae        mse      rmse           r2            model  rank
5  0.469907   0.773085  0.594448    -3.162772          xgboost   6.0
7  0.474313   0.730706  0.571284    -3.477708         lightgbm   6.0
6  0.527944   1.059187  0.647563    -4.679068               rf  12.0
1  0.662969   1.390415  0.825622    -5.468615   ensemble-arima  16.0
3  1.714147   6.136379  1.843400   -65.930570        base-lstm  22.0
0  1.195244  17.666922  1.427050  -430.936536  ensemble-arimax  23.0
4  3.237727  26.113308  3.694253  -300.603000         base-gru  27.0
2  7.200624  96.417500  8.915308 -3037.640703       simple-rnn  32.0
```


