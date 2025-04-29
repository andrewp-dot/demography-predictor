
# SecondModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
          mae       mse      rmse        r2             state  model   rank
138  0.139473  0.055287  0.213824  0.941702  Marshall Islands  ARIMA  340.0
189  0.104888  0.021745  0.130602  0.931735      Small states  ARIMA  103.0
101  0.091964  0.016615  0.117486  0.885249         Indonesia  ARIMA   79.0
```


## Model ARIMA - worst states
```
          mae       mse      rmse          r2                   state  model    rank
208  0.402916  0.545162  0.492349 -112.507477              Tajikistan  ARIMA  3701.0
24   0.875710  1.168447  1.065820 -304.505013  Bosnia and Herzegovina  ARIMA  5436.0
28   1.029777  2.021145  1.214741 -321.956306                Bulgaria  ARIMA  5980.0
```


## Model ARIMAX - top states
```
          mae       mse      rmse        r2                  state   model   rank
379  0.138623  0.036994  0.165593  0.911228  Micronesia, Fed. Sts.  ARIMAX  220.0
405  0.158829  0.066626  0.230941  0.881266       Papua New Guinea  ARIMAX  448.0
393  0.236446  0.089112  0.296348  0.847003              Nicaragua  ARIMAX  770.0
```


## Model ARIMAX - worst states
```
          mae        mse      rmse           r2             state   model    rank
390  1.757277   5.410387  1.965948  -751.869570           Namibia  ARIMAX  7017.0
299  7.394824  97.626173  8.502990 -2787.681026           Eritrea  ARIMAX  7576.0
280  5.851259  55.994665  6.199047 -5463.611603  Congo, Dem. Rep.  ARIMAX  7572.0
```


## Model RNN - top states
```
          mae       mse      rmse        r2        state model    rank
565  0.833097  1.228834  0.927726  0.754899     Honduras   RNN  3537.0
710  0.233231  0.084487  0.286763  0.595908  New Zealand   RNN   871.0
534  0.549611  0.637874  0.721922  0.505780  El Salvador   RNN  2736.0
```


## Model RNN - worst states
```
          mae       mse      rmse           r2            state model    rank
631  1.076350  1.726221  1.178741 -1150.117559            Niger   RNN  5942.0
682  1.474445  3.279030  1.567263 -1155.546908       Tajikistan   RNN  6615.0
664  0.817539  1.002656  0.884941 -4023.244168  Solomon Islands   RNN  5165.0
```


## Model LSTM - top states
```
          mae       mse      rmse        r2     state model    rank
769  0.360529  0.201748  0.402274  0.808446   Ecuador  LSTM  1375.0
915  0.246044  0.079364  0.280379  0.767009  Suriname  LSTM   774.0
731  0.898385  1.482235  1.058393  0.581302    Belize  LSTM  3906.0
```


## Model LSTM - worst states
```
           mae         mse       rmse           r2                   state model    rank
735   1.286369    2.115042   1.451115  -653.446585  Bosnia and Herzegovina  LSTM  6321.0
901   0.310930    0.134763   0.332600 -1662.322688         Solomon Islands  LSTM  2899.0
888  10.854022  133.694320  10.931977 -1765.148953                   Qatar  LSTM  7578.0
```


## Model GRU - top states
```
           mae       mse      rmse        r2                   state model    rank
1039  0.199642  0.049078  0.220859  0.892255                Honduras   GRU   452.0
983   0.260964  0.107845  0.314112  0.868836  Caribbean small states   GRU   863.0
1119  0.285071  0.137990  0.337911  0.857373             Philippines   GRU  1012.0
```


## Model GRU - worst states
```
           mae       mse      rmse          r2                  state model    rank
1177  1.559376  2.480257  1.561449 -423.091562  Virgin Islands (U.S.)   GRU  6536.0
976   0.776282  1.181063  0.851974 -524.763441               Bulgaria   GRU  5151.0
1156  1.149760  1.783694  1.191561 -591.553795             Tajikistan   GRU  6013.0
```


## Model XGBoost - top states
```
           mae       mse      rmse        r2                 state    model   rank
1277  0.095492  0.018289  0.123232  0.963380  Hong Kong SAR, China  XGBoost   76.0
1276  0.221089  0.108515  0.294814  0.961653              Honduras  XGBoost  729.0
1412  0.120551  0.026937  0.147502  0.934830         Venezuela, RB  XGBoost  150.0
```


## Model XGBoost - worst states
```
           mae       mse      rmse          r2            state    model    rank
1368  1.335802  3.546833  1.679952  -44.617259     Saudi Arabia  XGBoost  6406.0
1200  1.777455  7.545408  2.179129  -58.053500          Bahrain  XGBoost  6948.0
1375  0.174843  0.067988  0.213433 -125.152021  Solomon Islands  XGBoost  2235.0
```


## Model random_forest - top states
```
           mae       mse      rmse        r2        state          model   rank
1626  0.108968  0.017242  0.131173  0.956690     Suriname  random_forest   95.0
1506  0.174561  0.109195  0.247621  0.954882    Guatemala  random_forest  572.0
1593  0.166803  0.044388  0.198764  0.938440  Philippines  random_forest  338.0
```


## Model random_forest - worst states
```
           mae       mse      rmse          r2             state          model    rank
1481  1.597699  4.588670  1.848373 -181.177721  Egypt, Arab Rep.  random_forest  6819.0
1630  0.702151  0.788216  0.796727 -273.885373        Tajikistan  random_forest  4782.0
1429  1.568058  4.065656  1.781565 -349.390565        Arab World  random_forest  6774.0
```


## Model LightGBM - top states
```
           mae       mse      rmse        r2                state     model   rank
1743  0.215374  0.091489  0.251068  0.967919            Guatemala  LightGBM  605.0
1886  0.103952  0.020413  0.123916  0.961818        Venezuela, RB  LightGBM   88.0
1882  0.156949  0.034645  0.184393  0.910487  Upper middle income  LightGBM  273.0
```


## Model LightGBM - worst states
```
           mae       mse      rmse          r2                     state     model    rank
1834  0.269468  0.100348  0.311773  -60.665860  Pre-demographic dividend  LightGBM  2556.0
1666  0.528032  0.547365  0.586589  -64.123434                Arab World  LightGBM  3966.0
1849  0.151645  0.035491  0.180948 -142.246271           Solomon Islands  LightGBM  2056.0
```


## Per target metrics - model comparision
```
                          target          model       mae       mse      rmse          r2  rank
0           population_ages_0-14       LightGBM  0.449832  0.505705  0.533965   -2.089682     3
1           population_ages_0-14        XGBoost  0.478217  0.667427  0.606794   -2.497383     5
2           population_ages_0-14  random_forest  0.731907  1.371745  0.846763  -14.960379    12
3           population_ages_0-14          ARIMA  0.774603  1.770384  0.954558  -17.061588    13
4           population_ages_0-14           LSTM  1.258235  2.961044  1.327640  -38.478883    19
5           population_ages_0-14            GRU  1.032458  2.035067  1.102606  -48.452562    16
6           population_ages_0-14         ARIMAX  1.066918  3.790711  1.280779  -60.137786    18
7           population_ages_0-14            RNN  1.719437  5.085545  1.873695  -62.701022    22
8          population_ages_15-64        XGBoost  0.544665  1.033470  0.687576   -3.305675     9
9          population_ages_15-64       LightGBM  0.606528  1.126280  0.735565   -4.340259    10
10         population_ages_15-64          ARIMA  0.860648  1.964225  1.054725   -5.423368    14
11         population_ages_15-64            GRU  1.038094  2.136933  1.128540  -17.537314    17
12         population_ages_15-64  random_forest  1.026569  2.791326  1.195392  -19.449460    15
13         population_ages_15-64           LSTM  1.926743  8.223467  2.023543  -32.524107    24
14         population_ages_15-64            RNN  1.769856  5.027429  1.916310  -58.992910    23
15         population_ages_15-64         ARIMAX  1.286266  4.769994  1.532711  -97.014285    20
16  population_ages_65_and_above  random_forest  0.455044  0.720366  0.544857   -2.737803     4
17  population_ages_65_and_above          ARIMA  0.505753  0.859120  0.624398   -3.834502     6
18  population_ages_65_and_above        XGBoost  0.295210  0.380312  0.373121   -4.008964     1
19  population_ages_65_and_above       LightGBM  0.297900  0.377071  0.359557   -4.307343     2
20  population_ages_65_and_above         ARIMAX  0.542179  0.979602  0.669532   -8.629867     8
21  population_ages_65_and_above            GRU  0.523149  0.603195  0.567575  -26.990299     7
22  population_ages_65_and_above           LSTM  1.326672  3.924366  1.383892 -108.493627    21
23  population_ages_65_and_above            RNN  0.668340  0.972692  0.727008 -132.926165    11
```


## Overall metrics - model comparision
```
        mae       mse      rmse         r2          model  rank
5  0.469907  0.773085  0.594448  -3.162772        XGBoost   6.0
7  0.474313  0.730706  0.571284  -3.477708       LightGBM   6.0
6  0.748777  1.724899  0.879608 -11.892065  random_forest  15.0
0  0.756170  1.665824  0.931751  -7.856792          ARIMA  16.0
4  0.843680  1.546331  0.915101 -25.179205            GRU  17.0
1  0.969567  3.015138  1.168805 -49.691072         ARIMAX  24.0
2  1.403268  3.794362  1.529983 -73.651774            RNN  29.0
3  1.474452  4.927603  1.550890 -50.803958           LSTM  31.0
```


