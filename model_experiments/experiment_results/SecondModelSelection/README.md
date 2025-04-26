
# SecondModelSelection

**Description:** test

## Per target metrics - model comparision
```
                          target       mae       mse      rmse         r2          state            model   rank
46          population_ages_0-14  0.222103  0.079991  0.282827  -0.892294        Czechia               rf   76.0
55          population_ages_0-14  0.376676  0.191011  0.437048  -3.518608        Czechia         lightgbm  108.0
37          population_ages_0-14  0.422364  0.218236  0.467158  -4.162654        Czechia          xgboost  119.0
1           population_ages_0-14  1.036545  2.157728  1.468921 -50.043785        Czechia  ensemble-arimas  179.0
28          population_ages_0-14  1.672294  4.242592  2.059755 -99.363875        Czechia         base-gru  198.0
..                           ...       ...       ...       ...        ...            ...              ...    ...
53  population_ages_65_and_above  0.652760  0.681461  0.825507   0.515397  United States               rf  125.0
44  population_ages_65_and_above  0.680549  0.763682  0.873889   0.456928  United States          xgboost  129.0
8   population_ages_65_and_above  1.202786  2.355074  1.534625  -0.674748  United States  ensemble-arimas  161.0
26  population_ages_65_and_above  1.719455  3.038472  1.743121  -1.160729  United States        base-lstm  174.0
17  population_ages_65_and_above  1.576231  3.912962  1.978121  -1.782599  United States       simple-rnn  177.0

[63 rows x 8 columns]
```


## Overall metrics per state - model comparision
```
         mae        mse      rmse          r2          state            model  rank
0   0.672400   1.203471  1.024379  -16.766827        Czechia  ensemble-arimas  47.0
9   2.250205   9.605558  2.500695  -43.773642        Czechia         base-gru  68.0
3   2.204675   7.859087  2.533808 -121.012501        Czechia       simple-rnn  66.0
18  0.273387   0.112532  0.322417   -0.587961        Czechia         lightgbm  23.0
6   3.103254  14.865001  3.172128 -153.371685        Czechia        base-lstm  80.0
12  0.239335   0.101492  0.288046   -0.782794        Czechia          xgboost  18.0
15  0.266510   0.109482  0.311440    0.243473        Czechia               rf  15.0
16  0.495189   0.700402  0.603813    0.754545       Honduras               rf  26.0
7   0.868033   1.095301  0.947764   -0.338226       Honduras        base-lstm  39.0
4   4.526551  39.907655  5.266964   -6.969568       Honduras       simple-rnn  77.0
10  1.861001   5.973738  2.093723   -0.521896       Honduras         base-gru  50.0
1   1.757235   5.820185  2.134615   -1.120008       Honduras  ensemble-arimas  52.0
13  0.237469   0.119012  0.329153    0.896421       Honduras          xgboost  13.0
19  0.199411   0.080492  0.250030    0.826400       Honduras         lightgbm   5.0
14  0.520970   0.505601  0.698414    0.106894  United States          xgboost  29.0
17  0.590224   0.664304  0.731993   -0.160121  United States               rf  33.0
2   0.906004   1.508527  1.140192   -1.285824  United States  ensemble-arimas  49.0
11  2.265190   9.257797  2.574034  -23.631695  United States         base-gru  67.0
8   2.664567   8.980208  2.840995  -25.933893  United States        base-lstm  69.0
5   2.848475  12.980469  3.357148  -39.334861  United States       simple-rnn  76.0
20  0.348845   0.198919  0.411081    0.582013  United States         lightgbm  22.0
```


## Overall metrics - model comparision
```
        mae        mse      rmse         r2            model  rank
6  0.273985   0.134453  0.328983   0.454460         lightgbm   4.0
4  0.352183   0.271561  0.470153   0.253402          xgboost   9.0
5  0.489324   0.571630  0.599007   0.286826               rf  11.0
0  1.204207   3.188722  1.518920  -4.211067  ensemble-arimas  16.0
2  2.024703   6.937138  2.141339 -40.240424        base-lstm  21.0
3  2.099260   8.000349  2.366120 -18.203077         base-gru  23.0
1  3.400914  22.852008  3.968361 -42.066388       simple-rnn  28.0
```


