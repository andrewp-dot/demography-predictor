
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ensemble-arima - top states
```
          mae       mse      rmse        r2               state           model   rank
259  0.277230  0.136327  0.317043  0.949151              Bhutan  ensemble-arima  968.0
338  0.043808  0.004980  0.062891  0.946293           Indonesia  ensemble-arima   16.0
295  0.178961  0.046040  0.212277  0.943811             Ecuador  ensemble-arima  398.0
426  0.098822  0.019187  0.120513  0.942912        Small states  ensemble-arima   89.0
467  0.066466  0.008926  0.080386  0.917491  West Bank and Gaza  ensemble-arima   43.0
```


## Model ensemble-arima - worst states
```
          mae       mse      rmse         r2                       state           model    rank
361  0.424703  0.281701  0.521969 -61.087523                     Lesotho  ensemble-arima  3472.0
330  1.272058  2.823852  1.528139 -61.409078                     Hungary  ensemble-arima  6302.0
296  1.111168  2.110862  1.320316 -67.254335            Egypt, Arab Rep.  ensemble-arima  5992.0
380  1.438625  3.477969  1.680949 -68.918634  Middle East & North Africa  ensemble-arima  6519.0
265  0.523723  0.558305  0.663370 -86.225393                    Bulgaria  ensemble-arima  4144.0
```


## Model ensemble-arimax - top states
```
          mae       mse      rmse        r2             state            model    rank
168  0.204322  0.095617  0.271720  0.858752  Papua New Guinea  ensemble-arimax   703.0
22   0.555398  0.548058  0.622878  0.822267            Bhutan  ensemble-arimax  2399.0
160  0.112085  0.020175  0.137482  0.813108   North Macedonia  ensemble-arimax   171.0
14   0.495715  0.389683  0.601109  0.795908      Bahamas, The  ensemble-arimax  2165.0
80   0.148184  0.039338  0.180412  0.780259             Ghana  ensemble-arimax   336.0
```


## Model ensemble-arimax - worst states
```
           mae          mse       rmse             r2                 state            model    rank
153   1.719036     5.118961   1.923757    -667.830412               Namibia  ensemble-arimax  6951.0
208   1.117657     2.393342   1.300861    -823.036269            Tajikistan  ensemble-arimax  6180.0
92   12.478125   238.507282  14.261997    -823.207273  Hong Kong SAR, China  ensemble-arimax  7573.0
43    2.991219    13.049206   3.144580   -1357.230961      Congo, Dem. Rep.  ensemble-arimax  7454.0
62   67.007888  8864.620110  77.321071 -255384.632098               Eritrea  ensemble-arimax  7584.0
```


## Model RNN - top states
```
          mae       mse      rmse        r2                           state model    rank
565  0.833097  1.228834  0.927726  0.754899                        Honduras   RNN  3526.0
710  0.233231  0.084487  0.286763  0.595908                     New Zealand   RNN   927.0
534  0.549611  0.637874  0.721922  0.505780                     El Salvador   RNN  2776.0
508  0.393075  0.215022  0.452249  0.405069                          Canada   RNN  1756.0
673  0.371067  0.277969  0.493001  0.404321  St. Vincent and the Grenadines   RNN  1867.0
```


## Model RNN - worst states
```
          mae       mse      rmse           r2            state model    rank
479  1.973919  5.545804  2.097560  -710.123801           Angola   RNN  7084.0
474  0.339973  0.126383  0.351037  -939.057379      Afghanistan   RNN  2978.0
631  1.076350  1.726221  1.178741 -1150.117559            Niger   RNN  5924.0
682  1.474445  3.279030  1.567263 -1155.546908       Tajikistan   RNN  6626.0
664  0.817539  1.002656  0.884941 -4023.244168  Solomon Islands   RNN  5162.0
```


## Model LSTM - top states
```
          mae       mse      rmse        r2     state model    rank
769  0.360529  0.201748  0.402274  0.808446   Ecuador  LSTM  1396.0
915  0.246044  0.079364  0.280379  0.767009  Suriname  LSTM   819.0
731  0.898385  1.482235  1.058393  0.581302    Belize  LSTM  3897.0
852  0.544720  0.335171  0.564253  0.519771    Mexico  LSTM  2262.0
822  0.675522  0.860409  0.793211  0.428671     Kenya  LSTM  3183.0
```


## Model LSTM - worst states
```
           mae         mse       rmse           r2                   state model    rank
931  11.072154  139.884736  11.135324  -439.499574    United Arab Emirates  LSTM  7557.0
940   4.098937   20.753158   4.118071  -582.293446   Virgin Islands (U.S.)  LSTM  7518.0
735   1.286369    2.115042   1.451115  -653.446585  Bosnia and Herzegovina  LSTM  6328.0
901   0.310930    0.134763   0.332600 -1662.322688         Solomon Islands  LSTM  2929.0
888  10.854022  133.694320  10.931977 -1765.148953                   Qatar  LSTM  7573.0
```


## Model GRU - top states
```
           mae       mse      rmse        r2                   state model    rank
1039  0.199642  0.049078  0.220859  0.892255                Honduras   GRU   475.0
983   0.260964  0.107845  0.314112  0.868836  Caribbean small states   GRU   902.0
1119  0.285071  0.137990  0.337911  0.857373             Philippines   GRU  1047.0
1032  0.325543  0.169204  0.341925  0.850811               Guatemala   GRU  1188.0
1117  0.365050  0.218685  0.422044  0.814855                Paraguay   GRU  1458.0
```


## Model GRU - worst states
```
           mae       mse      rmse          r2                  state model    rank
995   1.033084  1.679813  1.077872 -376.009735                Croatia   GRU  5752.0
1138  0.334592  0.124232  0.342388 -378.485779        Solomon Islands   GRU  2934.0
1177  1.559376  2.480257  1.561449 -423.091562  Virgin Islands (U.S.)   GRU  6525.0
976   0.776282  1.181063  0.851974 -524.763441               Bulgaria   GRU  5154.0
1156  1.149760  1.783694  1.191561 -591.553795             Tajikistan   GRU  5999.0
```


## Model xgboost - top states
```
           mae       mse      rmse        r2                           state    model    rank
1277  0.095492  0.018289  0.123232  0.963380            Hong Kong SAR, China  xgboost    80.0
1276  0.221089  0.108515  0.294814  0.961653                        Honduras  xgboost   771.0
1412  0.120551  0.026937  0.147502  0.934830                   Venezuela, RB  xgboost   161.0
1222  0.150813  0.039619  0.174325  0.883901  Central Europe and the Baltics  xgboost   273.0
1269  0.253591  0.170390  0.322330  0.877268                       Guatemala  xgboost  1011.0
```


## Model xgboost - worst states
```
           mae       mse      rmse          r2                   state    model    rank
1322  1.093151  2.555874  1.432657  -33.725124                   Malta  xgboost  5993.0
1209  0.322165  0.172161  0.396504  -42.527709  Bosnia and Herzegovina  xgboost  2862.0
1368  1.335802  3.546833  1.679952  -44.617259            Saudi Arabia  xgboost  6410.0
1200  1.777455  7.545408  2.179129  -58.053500                 Bahrain  xgboost  6920.0
1375  0.174843  0.067988  0.213433 -125.152021         Solomon Islands  xgboost  2247.0
```


## Model rf - top states
```
           mae       mse      rmse        r2        state model    rank
1626  0.108968  0.017242  0.131173  0.956690     Suriname    rf    96.0
1506  0.174561  0.109195  0.247621  0.954882    Guatemala    rf   593.0
1593  0.166803  0.044388  0.198764  0.938440  Philippines    rf   349.0
1487  0.117199  0.025178  0.132678  0.928969     Ethiopia    rf   134.0
1444  0.275920  0.206447  0.391192  0.919391       Bhutan    rf  1187.0
```


## Model rf - worst states
```
           mae       mse      rmse          r2              state model    rank
1483  0.926711  1.705297  1.087629 -153.364113  Equatorial Guinea    rf  5637.0
1575  0.779940  1.324350  0.944506 -157.873618            Namibia    rf  5234.0
1481  1.597699  4.588670  1.848373 -181.177721   Egypt, Arab Rep.    rf  6798.0
1630  0.702151  0.788216  0.796727 -273.885373         Tajikistan    rf  4788.0
1429  1.568058  4.065656  1.781565 -349.390565         Arab World    rf  6761.0
```


## Model lightgbm - top states
```
           mae       mse      rmse        r2                                         state     model   rank
1743  0.215374  0.091489  0.251068  0.967919                                     Guatemala  lightgbm  639.0
1886  0.103952  0.020413  0.123916  0.961818                                 Venezuela, RB  lightgbm   95.0
1882  0.156949  0.034645  0.184393  0.910487                           Upper middle income  lightgbm  281.0
1781  0.120407  0.025290  0.141626  0.903993  Least developed countries: UN classification  lightgbm  156.0
1665  0.155779  0.051363  0.199692  0.900555                           Antigua and Barbuda  lightgbm  361.0
```


## Model lightgbm - worst states
```
           mae       mse      rmse          r2                     state     model    rank
1718  0.736650  1.024884  0.827439  -32.522989          Egypt, Arab Rep.  lightgbm  4741.0
1885  0.449947  0.283634  0.487302  -38.971739                   Vanuatu  lightgbm  3389.0
1834  0.269468  0.100348  0.311773  -60.665860  Pre-demographic dividend  lightgbm  2585.0
1666  0.528032  0.547365  0.586589  -64.123434                Arab World  lightgbm  3990.0
1849  0.151645  0.035491  0.180948 -142.246271           Solomon Islands  lightgbm  2062.0
```


## Per target metrics - model comparision
```
                          target            model       mae        mse      rmse           r2  rank
0           population_ages_0-14         lightgbm  0.449832   0.505705  0.533965    -2.089682     4
1           population_ages_0-14          xgboost  0.478217   0.667427  0.606794    -2.497383     6
2           population_ages_0-14   ensemble-arima  0.656464   1.359651  0.814877    -6.946118    11
3           population_ages_0-14               rf  0.731907   1.371745  0.846763   -14.960379    13
4           population_ages_0-14             LSTM  1.258235   2.961044  1.327640   -38.478883    18
5           population_ages_0-14              GRU  1.032458   2.035067  1.102606   -48.452562    16
6           population_ages_0-14              RNN  1.719437   5.085545  1.873695   -62.701022    21
7           population_ages_0-14  ensemble-arimax  1.674727  60.287081  1.977694 -1127.636009    20
8          population_ages_15-64          xgboost  0.544665   1.033470  0.687576    -3.305675     8
9          population_ages_15-64         lightgbm  0.606528   1.126280  0.735565    -4.340259     9
10         population_ages_15-64   ensemble-arima  0.834058   2.029728  1.028809    -5.558373    14
11         population_ages_15-64              GRU  1.038094   2.136933  1.128540   -17.537314    17
12         population_ages_15-64               rf  1.026569   2.791326  1.195392   -19.449460    15
13         population_ages_15-64             LSTM  1.926743   8.223467  2.023543   -32.524107    24
14         population_ages_15-64              RNN  1.769856   5.027429  1.916310   -58.992910    22
15         population_ages_15-64  ensemble-arimax  1.892892  64.122585  2.241989 -2217.177694    23
16  population_ages_65_and_above               rf  0.455044   0.720366  0.544857    -2.737803     5
17  population_ages_65_and_above   ensemble-arima  0.386520   0.509008  0.489265    -3.444971     3
18  population_ages_65_and_above          xgboost  0.295210   0.380312  0.373121    -4.008964     1
19  population_ages_65_and_above         lightgbm  0.297900   0.377071  0.359557    -4.307343     2
20  population_ages_65_and_above  ensemble-arimax  0.629898   1.256158  0.761840   -11.284446    10
21  population_ages_65_and_above              GRU  0.523149   0.603195  0.567575   -26.990299     7
22  population_ages_65_and_above             LSTM  1.326672   3.924366  1.383892  -108.493627    19
23  population_ages_65_and_above              RNN  0.668340   0.972692  0.727008  -132.926165    12
```


## Overall metrics - model comparision
```
        mae        mse      rmse          r2            model  rank
5  0.469907   0.773085  0.594448   -3.162772          xgboost   6.0
7  0.474313   0.730706  0.571284   -3.477708         lightgbm   6.0
1  0.662969   1.390415  0.825622   -5.468615   ensemble-arima  12.0
6  0.748777   1.724899  0.879608  -11.892065               rf  17.0
4  0.843680   1.546331  0.915101  -25.179205              GRU  19.0
2  1.403268   3.794362  1.529983  -73.651774              RNN  27.0
0  1.195244  17.666922  1.427050 -430.936536  ensemble-arimax  28.0
3  1.474452   4.927603  1.550890  -50.803958             LSTM  29.0
```


