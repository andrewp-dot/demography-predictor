
# TargetModelSelection

**Description:** Compares models to predict the target variable(s) using past data and future known (ground truth) data.

## Model ARIMA - top states
```
          mae       mse      rmse      mape        r2                  state        model   rank
358  0.083903  0.010015  0.098916  0.003542  0.873007                 Latvia  aging_ARIMA   60.0
466  0.085102  0.014350  0.109410  0.002875  0.467024  Virgin Islands (U.S.)  aging_ARIMA  248.0
338  0.091964  0.016615  0.117486  0.004935  0.885249              Indonesia  aging_ARIMA   76.0
```


## Model ARIMA - worst states
```
          mae        mse      rmse      mape         r2               state        model    rank
360  2.635783   9.740583  2.930078  0.112753  -4.096844             Lebanon  aging_ARIMA  6491.0
254  2.410142   9.867229  2.943133  0.099927 -40.817966            Barbados  aging_ARIMA  7040.0
339  3.104064  17.961393  3.705867  0.112491 -19.432922  Iran, Islamic Rep.  aging_ARIMA  7054.0
```


## Model ARIMAX - top states
```
          mae       mse      rmse      mape        r2      state         model   rank
226  0.087906  0.014645  0.101893  0.002848  0.415131    Vanuatu  aging_ARIMAX  267.0
184  0.101689  0.027499  0.140111  0.003759  0.767104    Senegal  aging_ARIMAX  183.0
96   0.122052  0.029583  0.145875  0.006621  0.627018  IDA blend  aging_ARIMAX  287.0
```


## Model ARIMAX - worst states
```
          mae        mse      rmse      mape           r2                 state         model    rank
220  4.741639  35.228739  5.345486  0.649289  -209.158381  United Arab Emirates  aging_ARIMAX  7489.0
43   5.851259  55.994665  6.199047  0.147129 -5463.611603      Congo, Dem. Rep.  aging_ARIMAX  7575.0
62   7.394824  97.626173  8.502990  0.271358 -2787.681026               Eritrea  aging_ARIMAX  7579.0
```


## Model RNN - top states
```
          mae       mse      rmse      mape          r2        state      model    rank
631  0.273258  0.155013  0.290036  0.009802  -17.645664        Niger  aging_RNN  2266.0
552  0.297770  0.110093  0.313890  0.014904   -0.361040      Georgia  aging_RNN  1299.0
706  0.356618  0.142939  0.375921  0.052159 -146.853373  Yemen, Rep.  aging_RNN  2834.0
```


## Model RNN - worst states
```
          mae        mse      rmse      mape         r2                 state      model    rank
681  3.211019  18.205239  3.766237  0.146508  -4.049764  Syrian Arab Republic  aging_RNN  6631.0
600  3.656057  19.383663  3.886067  0.150664 -20.195912                 Libya  aging_RNN  7082.0
582  4.090325  26.512246  4.486538  0.212549 -41.076264                 Japan  aging_RNN  7263.0
```


## Model LSTM - top states
```
          mae       mse      rmse      mape         r2         state       model    rank
773  0.373059  0.185056  0.392218  0.061884 -30.207773       Eritrea  aging_LSTM  2706.0
795  0.346755  0.162894  0.392898  0.041198  -0.393770     Guatemala  aging_LSTM  1567.0
740  0.397557  0.176981  0.409179  0.070204 -21.813940  Burkina Faso  aging_LSTM  2706.0
```


## Model LSTM - worst states
```
          mae         mse      rmse      mape          r2                 state       model    rank
726  4.842368   41.297505  5.437949  0.199738 -563.609242               Bahrain  aging_LSTM  7549.0
931  7.480573   94.232581  8.069755  0.630235 -397.205642  United Arab Emirates  aging_LSTM  7547.0
888  7.867303  106.680131  8.772357  0.742373 -286.243259                 Qatar  aging_LSTM  7532.0
```


## Model GRU - top states
```
           mae       mse      rmse      mape         r2        state      model    rank
948   0.192557  0.051423  0.203063  0.014716 -48.126039  Afghanistan  aging_GRU  2016.0
1180  0.171171  0.052169  0.203223  0.009255  -2.488553  Yemen, Rep.  aging_GRU  1214.0
1033  0.194231  0.083200  0.256824  0.009393   0.690486       Guinea  aging_GRU   645.0
```


## Model GRU - worst states
```
           mae        mse      rmse      mape          r2                 state      model    rank
1096  4.042627  22.725375  4.320467  0.243581  -41.705589              Mongolia  aging_GRU  7253.0
1125  4.920835  35.767494  5.267537  1.659808 -587.266017                 Qatar  aging_GRU  7547.0
1168  5.079042  37.884304  5.387745  2.324550 -331.857108  United Arab Emirates  aging_GRU  7526.0
```


## Model XGBoost - top states
```
           mae       mse      rmse      mape        r2               state          model    rank
1334  0.051121  0.007839  0.079051  0.002285 -7.272347          Montenegro  aging_XGBoost  1185.0
1393  0.066031  0.007102  0.079711  0.007135 -1.450954          Tajikistan  aging_XGBoost   746.0
1364  0.065874  0.009473  0.088145  0.003107  0.758235  Russian Federation  aging_XGBoost   110.0
```


## Model XGBoost - worst states
```
           mae        mse      rmse      mape         r2    state          model    rank
1200  1.777455   7.545408  2.179129  0.075996 -58.053500  Bahrain  aging_XGBoost  6821.0
1348  1.563013   6.173536  2.185801  0.095202 -10.047892     Oman  aging_XGBoost  6259.0
1293  2.583080  12.315211  3.062156  0.094847  -3.672046    Japan  aging_XGBoost  6507.0
```


## Model random_forest - top states
```
           mae       mse      rmse      mape        r2                       state                model   rank
1484  0.055716  0.005268  0.068356  0.003948  0.773282                     Eritrea  aging_random_forest   86.0
1424  0.073088  0.008341  0.087850  0.006495  0.182914  Africa Western and Central  aging_random_forest  333.0
1422  0.078978  0.011366  0.089891  0.003598 -0.651679                 Afghanistan  aging_random_forest  581.0
```


## Model random_forest - worst states
```
           mae        mse      rmse      mape         r2                 state                model    rank
1524  2.767163  19.781907  3.666463  0.088622 -13.864678    Iran, Islamic Rep.  aging_random_forest  6927.0
1530  3.718095  21.484240  4.038883  0.130584  -5.942076                 Japan  aging_random_forest  6760.0
1629  3.557975  22.464800  4.109318  0.124037  -4.101773  Syrian Arab Republic  aging_random_forest  6662.0
```


## Model LightGBM - top states
```
           mae       mse      rmse      mape        r2        state           model   rank
1659  0.037956  0.002790  0.044733  0.003383 -1.929763  Afghanistan  aging_LightGBM  809.0
1810  0.046013  0.003561  0.052323  0.002236  0.557443   Mozambique  aging_LightGBM  182.0
1721  0.067160  0.007975  0.079998  0.005849  0.413726      Eritrea  aging_LightGBM  242.0
```


## Model LightGBM - worst states
```
           mae        mse      rmse      mape        r2                 state           model    rank
1873  1.702838   7.508845  2.090165  0.053556 -7.298901   Trinidad and Tobago  aging_LightGBM  6272.0
1866  2.375887  11.283122  2.912794  0.086902 -1.573377  Syrian Arab Republic  aging_LightGBM  6199.0
1767  3.078361  15.983221  3.463545  0.111572 -4.558224                 Japan  aging_LightGBM  6634.0
```


## Per target metrics - model comparision
```
                          target                model       mae       mse      rmse      mape          r2  rank
0           population_ages_0-14       aging_LightGBM  0.449832  0.505705  0.533965  0.018043   -2.089682     3
1           population_ages_0-14        aging_XGBoost  0.478217  0.667427  0.606794  0.018971   -2.497383     5
2           population_ages_0-14  aging_random_forest  0.731907  1.371745  0.846763  0.028631  -14.960379    11
3           population_ages_0-14          aging_ARIMA  0.774603  1.770384  0.954558  0.030355  -17.061588    12
4           population_ages_0-14         aging_ARIMAX  1.066918  3.790711  1.280779  0.041287  -60.137786    17
5           population_ages_0-14            aging_RNN  1.626553  4.481320  1.765476  0.066998  -88.372686    21
6           population_ages_0-14           aging_LSTM  1.687159  5.650604  1.808610  0.067369  -65.256108    22
7           population_ages_0-14            aging_GRU  2.271102  8.196237  2.418588  0.089243 -108.087261    24
8          population_ages_15-64        aging_XGBoost  0.544665  1.033470  0.687576  0.008459   -3.305675     8
9          population_ages_15-64       aging_LightGBM  0.606528  1.126280  0.735565  0.009454   -4.340259     9
10         population_ages_15-64          aging_ARIMA  0.860648  1.964225  1.054725  0.013607   -5.423368    14
11         population_ages_15-64  aging_random_forest  1.026569  2.791326  1.195392  0.016013  -19.449460    16
12         population_ages_15-64            aging_RNN  1.234860  2.699417  1.319509  0.019850  -30.729315    18
13         population_ages_15-64         aging_ARIMAX  1.286266  4.769994  1.532711  0.020421  -97.014285    19
14         population_ages_15-64            aging_GRU  1.553919  4.055350  1.656459  0.024984  -39.669030    20
15         population_ages_15-64           aging_LSTM  1.961220  7.817834  2.141462  0.030094  -48.130883    23
16  population_ages_65_and_above       aging_LightGBM  0.297900  0.377071  0.359557  0.040714   -4.307343     1
17  population_ages_65_and_above        aging_XGBoost  0.295210  0.380312  0.373121  0.039963   -4.008964     2
18  population_ages_65_and_above  aging_random_forest  0.455044  0.720366  0.544857  0.056692   -2.737803     4
19  population_ages_65_and_above          aging_ARIMA  0.505753  0.859120  0.624398  0.063853   -3.834502     6
20  population_ages_65_and_above         aging_ARIMAX  0.542179  0.979602  0.669532  0.073035   -8.629867     7
21  population_ages_65_and_above           aging_LSTM  0.705735  0.905420  0.756784  0.119332 -118.821971    10
22  population_ages_65_and_above            aging_GRU  0.922059  1.682950  0.979121  0.195449  -67.098128    13
23  population_ages_65_and_above            aging_RNN  1.032048  1.861657  1.110412  0.196072 -113.214114    15
```


## Overall metrics - model comparision
```
        mae       mse      rmse      mape         r2                model  rank
5  0.469907  0.773085  0.594448  0.023742  -3.162772        aging_XGBoost   6.0
7  0.474313  0.730706  0.571284  0.023544  -3.477708       aging_LightGBM   6.0
1  0.756170  1.665824  0.931751  0.037795  -7.856792          aging_ARIMA  14.0
6  0.748777  1.724899  0.879608  0.034142 -11.892065  aging_random_forest  14.0
0  0.969567  3.015138  1.168805  0.044859 -49.691072         aging_ARIMAX  20.0
2  1.302317  3.052082  1.407487  0.095964 -68.266003            aging_RNN  26.0
3  1.421084  4.666500  1.546407  0.071161 -64.632170           aging_LSTM  27.0
4  1.606564  4.749886  1.715321  0.104869 -64.959586            aging_GRU  31.0
```


