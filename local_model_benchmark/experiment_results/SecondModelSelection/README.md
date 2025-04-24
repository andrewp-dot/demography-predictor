
# SecondModelSelection

**Description:** test

## Per target metrics - model comparision
```
                          target       mae        mse      rmse          r2          state       model   rank
37          population ages 0-14  0.215761   0.075336  0.274475   -0.782179        Czechia          rf   40.0
28          population ages 0-14  0.447607   0.245266  0.495243   -4.802080        Czechia     xgboost   63.0
19          population ages 0-14  2.927907   8.881996  2.980268 -209.114823        Czechia    base-gru  155.0
10          population ages 0-14  3.297837  11.335233  3.366784 -267.149244        Czechia   base-lstm  164.0
1           population ages 0-14  3.585490  13.347399  3.653409 -314.749567        Czechia  simple-rnn  174.0
27         population ages 15-64  0.039895   0.002393  0.048920    0.996719        Czechia     xgboost    4.0
36         population ages 15-64  0.541756   0.392453  0.626461    0.461887        Czechia          rf   53.0
18         population ages 15-64  2.996450   9.285842  3.047268  -11.732291        Czechia    base-gru  154.0
0          population ages 15-64  3.678013  13.977112  3.738598  -18.164729        Czechia  simple-rnn  170.0
9          population ages 15-64  4.209928  18.259349  4.273096  -24.036322        Czechia   base-lstm  174.0
38  population ages 65 and above  0.177719   0.037245  0.192991    0.911604        Czechia          rf   14.0
29  population ages 65 and above  0.231509   0.098557  0.313939    0.766090        Czechia     xgboost   31.0
20  population ages 65 and above  0.949145   0.917825  0.958032   -1.178309        Czechia    base-gru   78.0
2   population ages 65 and above  1.142251   1.322960  1.150200   -2.139828        Czechia  simple-rnn  103.0
11  population ages 65 and above  1.192643   1.447267  1.203024   -2.434851        Czechia   base-lstm  115.0
31          population ages 0-14  0.338237   0.202244  0.449715    0.967255       Honduras     xgboost   30.0
40          population ages 0-14  1.185011   2.077214  1.441254    0.663686       Honduras          rf   99.0
13          population ages 0-14  1.658080   3.125322  1.767858    0.493990       Honduras   base-lstm  114.0
22          population ages 0-14  1.658278   3.140563  1.772163    0.491523       Honduras    base-gru  118.0
4           population ages 0-14  1.916142   4.126576  2.031398    0.331881       Honduras  simple-rnn  125.0
39         population ages 15-64  0.232059   0.082563  0.287338    0.981547       Honduras          rf   21.0
30         population ages 15-64  0.524831   0.415107  0.644288    0.907221       Honduras     xgboost   43.0
21         population ages 15-64  0.738192   0.653551  0.808425    0.853928       Honduras    base-gru   51.0
12         population ages 15-64  0.873241   0.959223  0.979399    0.785608       Honduras   base-lstm   60.0
3          population ages 15-64  1.007127   1.204896  1.097678    0.730699       Honduras  simple-rnn   76.0
32  population ages 65 and above  0.128706   0.024659  0.157031    0.824952       Honduras     xgboost   13.0
41  population ages 65 and above  0.166220   0.048077  0.219266    0.658708       Honduras          rf   24.0
14  population ages 65 and above  0.951545   0.975400  0.987624   -5.924176       Honduras   base-lstm   90.0
5   population ages 65 and above  1.000556   1.101963  1.049744   -6.822617       Honduras  simple-rnn   95.0
23  population ages 65 and above  1.139392   1.408934  1.186985   -9.001746       Honduras    base-gru  115.0
43          population ages 0-14  0.264133   0.089496  0.299159    0.662477  United States          rf   34.0
34          population ages 0-14  0.344899   0.251259  0.501258    0.052408  United States     xgboost   52.0
25          population ages 0-14  2.882574   8.912434  2.985370  -32.612088  United States    base-gru  153.0
16          population ages 0-14  3.354876  12.163440  3.487612  -44.872833  United States   base-lstm  164.0
7           population ages 0-14  3.398751  12.460968  3.530010  -45.994923  United States  simple-rnn  168.0
24         population ages 15-64  0.669944   0.647192  0.804482   -0.428693  United States    base-gru   65.0
33         population ages 15-64  0.888678   1.212104  1.100956   -1.675752  United States     xgboost   90.0
42         population ages 15-64  0.961920   1.374083  1.172213   -2.033323  United States          rf  100.0
6          population ages 15-64  1.215192   2.146766  1.465185   -3.739042  United States  simple-rnn  124.0
15         population ages 15-64  1.645385   3.766226  1.940677   -7.314042  United States   base-lstm  138.0
35  population ages 65 and above  0.833575   1.103723  1.050582    0.215116  United States     xgboost   74.0
44  population ages 65 and above  0.871477   1.264523  1.124510    0.100768  United States          rf   82.0
17  population ages 65 and above  1.174831   1.392305  1.179960    0.009899  United States   base-lstm  100.0
8   population ages 65 and above  1.208945   1.474053  1.214106   -0.048234  United States  simple-rnn  110.0
26  population ages 65 and above  1.490503   2.241072  1.497021   -0.593679  United States    base-gru  120.0
```


## Overall metrics - model comparision
```
         mae        mse      rmse          r2          state       model  rank
9   0.239670   0.115406  0.286034   -1.013090        Czechia     xgboost   9.0
12  0.311745   0.168345  0.364642    0.197104        Czechia          rf   9.0
6   2.291168   6.361888  2.328523  -74.008474        Czechia    base-gru  52.0
0   2.801918   9.549157  2.847402 -111.684708        Czechia  simple-rnn  57.0
3   2.900136  10.347283  2.947635  -97.873472        Czechia   base-lstm  59.0
10  0.330591   0.214003  0.417012    0.899810       Honduras     xgboost  10.0
13  0.527763   0.735952  0.649286    0.767980       Honduras          rf  14.0
4   1.160955   1.686648  1.244960   -1.548193       Honduras   base-lstm  28.0
7   1.178620   1.734349  1.255858   -2.552099       Honduras    base-gru  33.0
1   1.307942   2.144478  1.392940   -1.920012       Honduras  simple-rnn  35.0
11  0.689051   0.855696  0.884265   -0.469409  United States     xgboost  21.0
14  0.699177   0.909367  0.865294   -0.423360  United States          rf  21.0
8   1.681007   3.933566  1.762291  -11.211487  United States    base-gru  40.0
2   1.940963   5.360596  2.069767  -16.594066  United States  simple-rnn  44.0
5   2.058364   5.773990  2.202750  -17.392325  United States   base-lstm  48.0
```


