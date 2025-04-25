
# SecondModelSelection

**Description:** test

## Per target metrics - model comparision
```
                          target       mae        mse      rmse          r2          state            model   rank
46          population ages 0-14  0.222103   0.079991  0.282827   -0.892294        Czechia               rf   56.0
37          population ages 0-14  0.422364   0.218236  0.467158   -4.162654        Czechia          xgboost   88.0
1           population ages 0-14  1.036545   2.157728  1.468921  -50.043785        Czechia  ensemble-arimas  143.0
28          population ages 0-14  1.672294   4.242592  2.059755  -99.363875        Czechia         base-gru  162.0
10          population ages 0-14  3.350705  14.933348  3.864369 -352.267208        Czechia       simple-rnn  191.0
19          population ages 0-14  4.133023  18.036921  4.246990 -425.686135        Czechia        base-lstm  199.0
36         population ages 15-64  0.087107   0.018903  0.137489    0.974081        Czechia          xgboost    8.0
45         population ages 15-64  0.401242   0.211916  0.460344    0.709431        Czechia               rf   58.0
0          population ages 15-64  0.545415   1.188419  1.090146   -0.629502        Czechia  ensemble-arimas  107.0
9          population ages 15-64  2.441216   7.727729  2.779879   -9.595883        Czechia       simple-rnn  164.0
27         population ages 15-64  4.625249  24.312374  4.930758  -32.335933        Czechia         base-gru  197.0
18         population ages 15-64  5.067881  26.544327  5.152119  -35.396278        Czechia        base-lstm  203.0
20  population ages 65 and above  0.108857   0.013753  0.117275    0.967359        Czechia        base-lstm    8.0
47  population ages 65 and above  0.176185   0.036538  0.191149    0.913283        Czechia               rf   23.0
38  population ages 65 and above  0.208534   0.067335  0.259490    0.840191        Czechia          xgboost   32.0
29  population ages 65 and above  0.453072   0.261706  0.511573    0.378882        Czechia         base-gru   75.0
2   population ages 65 and above  0.435238   0.264267  0.514069    0.372806        Czechia  ensemble-arimas   77.0
11  population ages 65 and above  0.822104   0.916184  0.957175   -1.174413        Czechia       simple-rnn  115.0
40          population ages 0-14  0.326397   0.138836  0.372607    0.977522       Honduras          xgboost   37.0
49          population ages 0-14  1.177396   2.021468  1.421784    0.672711       Honduras               rf  106.0
22          population ages 0-14  1.490621   2.465833  1.570297    0.600766       Honduras        base-lstm  116.0
4           population ages 0-14  2.829891  11.675513  3.416945   -0.890341       Honduras  ensemble-arimas  161.0
31          population ages 0-14  3.165970  12.202591  3.493221   -0.975678       Honduras         base-gru  166.0
13          population ages 0-14  6.537493  58.769114  7.666102   -8.515100       Honduras       simple-rnn  199.0
48         population ages 15-64  0.132097   0.022832  0.151103    0.994897       Honduras               rf   11.0
39         population ages 15-64  0.264999   0.183367  0.428214    0.959016       Honduras          xgboost   39.0
21         population ages 15-64  0.429894   0.320869  0.566453    0.928284       Honduras        base-lstm   67.0
3          population ages 15-64  1.904459   5.322997  2.307162   -0.189720       Honduras  ensemble-arimas  138.0
30         population ages 15-64  1.991555   5.527815  2.351131   -0.235498       Honduras         base-gru  142.0
12         population ages 15-64  6.744023  60.841847  7.800118  -12.598500       Honduras       simple-rnn  204.0
41  population ages 65 and above  0.121010   0.034833  0.186637    0.752725       Honduras          xgboost   22.0
50  population ages 65 and above  0.176074   0.056907  0.238552    0.596028       Honduras               rf   32.0
14  population ages 65 and above  0.298136   0.112005  0.334672    0.204895       Honduras       simple-rnn   52.0
32  population ages 65 and above  0.425479   0.190809  0.436816   -0.354513       Honduras         base-gru   70.0
5   population ages 65 and above  0.537353   0.462044  0.679738   -2.279962       Honduras  ensemble-arimas  105.0
23  population ages 65 and above  0.683585   0.499200  0.706541   -2.543726       Honduras        base-lstm  112.0
52          population ages 0-14  0.223292   0.064391  0.253755    0.757156  United States               rf   33.0
7           population ages 0-14  0.296269   0.250201  0.500201    0.056400  United States  ensemble-arimas   66.0
43          population ages 0-14  0.389309   0.302913  0.550376   -0.142398  United States          xgboost   78.0
34          population ages 0-14  2.500207   8.245379  2.871477  -30.096375  United States         base-gru  171.0
25          population ages 0-14  3.858735  16.546617  4.067753  -61.403417  United States        base-lstm  191.0
16          population ages 0-14  4.358081  26.150801  5.113785  -97.624352  United States       simple-rnn  203.0
42         population ages 15-64  0.493052   0.450210  0.670977    0.006151  United States          xgboost   88.0
51         population ages 15-64  0.894620   1.247060  1.116718   -1.752917  United States               rf  121.0
6          population ages 15-64  1.218958   1.920305  1.385751   -3.239124  United States  ensemble-arimas  131.0
24         population ages 15-64  2.415511   7.355535  2.712109  -15.237534  United States        base-lstm  163.0
15         population ages 15-64  2.611114   8.877644  2.979538  -18.597630  United States       simple-rnn  173.0
33         population ages 15-64  3.968563  19.320926  4.395558  -41.651447  United States         base-gru  194.0
35  population ages 65 and above  0.326801   0.207085  0.455066    0.852737  United States         base-gru   50.0
53  population ages 65 and above  0.652760   0.681461  0.825507    0.515397  United States               rf   91.0
44  population ages 65 and above  0.680549   0.763682  0.873889    0.456928  United States          xgboost   95.0
8   population ages 65 and above  1.202786   2.355074  1.534625   -0.674748  United States  ensemble-arimas  126.0
26  population ages 65 and above  1.719455   3.038472  1.743121   -1.160729  United States        base-lstm  139.0
17  population ages 65 and above  1.576231   3.912962  1.978121   -1.782599  United States       simple-rnn  142.0
```


## Overall metrics per state - model comparision
```
         mae        mse      rmse          r2          state            model  rank
6   3.103254  14.865001  3.172128 -153.371685        Czechia        base-lstm  68.0
15  0.266510   0.109482  0.311440    0.243473        Czechia               rf  10.0
12  0.239335   0.101492  0.288046   -0.782794        Czechia          xgboost  12.0
9   2.250205   9.605558  2.500695  -43.773642        Czechia         base-gru  56.0
3   2.204675   7.859087  2.533808 -121.012501        Czechia       simple-rnn  54.0
0   0.672400   1.203471  1.024379  -16.766827        Czechia  ensemble-arimas  35.0
1   1.757235   5.820185  2.134615   -1.120008       Honduras  ensemble-arimas  40.0
10  1.861001   5.973738  2.093723   -0.521896       Honduras         base-gru  39.0
13  0.237469   0.119012  0.329153    0.896421       Honduras          xgboost   8.0
7   0.868033   1.095301  0.947764   -0.338226       Honduras        base-lstm  28.0
16  0.495189   0.700402  0.603813    0.754545       Honduras               rf  16.0
4   4.526551  39.907655  5.266964   -6.969568       Honduras       simple-rnn  65.0
17  0.590224   0.664304  0.731993   -0.160121  United States               rf  22.0
14  0.520970   0.505601  0.698414    0.106894  United States          xgboost  18.0
11  2.265190   9.257797  2.574034  -23.631695  United States         base-gru  55.0
8   2.664567   8.980208  2.840995  -25.933893  United States        base-lstm  57.0
5   2.848475  12.980469  3.357148  -39.334861  United States       simple-rnn  64.0
2   0.906004   1.508527  1.140192   -1.285824  United States  ensemble-arimas  37.0
```


## Overall metrics - model comparision
```
        mae        mse      rmse         r2            model  rank
4  0.352183   0.271561  0.470153   0.253402          xgboost   5.0
5  0.489324   0.571630  0.599007   0.286826               rf   7.0
0  1.204207   3.188722  1.518920  -4.211067  ensemble-arimas  12.0
2  2.024703   6.937138  2.141339 -40.240424        base-lstm  17.0
3  2.099260   8.000349  2.366120 -18.203077         base-gru  19.0
1  3.400914  22.852008  3.968361 -42.066388       simple-rnn  24.0
```


