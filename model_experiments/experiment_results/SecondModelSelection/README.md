
# SecondModelSelection

**Description:** test

## Model ensemble-arimas - top states
```
        mae       mse      rmse         r2          state            model  rank
0  1.188102  3.343381  1.705329 -47.299391        Czechia  ensemble-arimas  51.0
2  1.417881  3.552506  1.760174  -7.299024  United States  ensemble-arimas  52.0
1  1.843364  6.369864  2.197584  -0.784132       Honduras  ensemble-arimas  54.0
```


## Model ensemble-arimas - worst states
```
        mae       mse      rmse         r2          state            model  rank
0  1.188102  3.343381  1.705329 -47.299391        Czechia  ensemble-arimas  51.0
2  1.417881  3.552506  1.760174  -7.299024  United States  ensemble-arimas  52.0
1  1.843364  6.369864  2.197584  -0.784132       Honduras  ensemble-arimas  54.0
```


## Model simple-rnn - top states
```
         mae         mse       rmse           r2          state       model  rank
3   5.631112   53.914780   6.489965 -1033.493987        Czechia  simple-rnn  75.0
5   6.378090   65.581050   7.359443  -135.994860  United States  simple-rnn  74.0
4  10.041585  168.667937  12.559621  -177.838122       Honduras  simple-rnn  82.0
```


## Model simple-rnn - worst states
```
         mae         mse       rmse           r2          state       model  rank
3   5.631112   53.914780   6.489965 -1033.493987        Czechia  simple-rnn  75.0
5   6.378090   65.581050   7.359443  -135.994860  United States  simple-rnn  74.0
4  10.041585  168.667937  12.559621  -177.838122       Honduras  simple-rnn  82.0
```


## Model base-lstm - top states
```
        mae        mse      rmse         r2          state      model  rank
7  0.625284   0.775522  0.733402   0.783189       Honduras  base-lstm  30.0
8  2.228527   5.693800  2.366689 -12.937814  United States  base-lstm  58.0
6  2.601925  10.991409  2.670059 -53.457374        Czechia  base-lstm  64.0
```


## Model base-lstm - worst states
```
        mae        mse      rmse         r2          state      model  rank
7  0.625284   0.775522  0.733402   0.783189       Honduras  base-lstm  30.0
8  2.228527   5.693800  2.366689 -12.937814  United States  base-lstm  58.0
6  2.601925  10.991409  2.670059 -53.457374        Czechia  base-lstm  64.0
```


## Model base-gru - top states
```
         mae        mse      rmse          r2          state     model  rank
10  1.076709   1.711769  1.250938   -1.114989       Honduras  base-gru  45.0
9   5.151525  36.352269  5.674105 -151.904592        Czechia  base-gru  69.0
11  6.763624  71.114486  7.785149 -180.369297  United States  base-gru  80.0
```


## Model base-gru - worst states
```
         mae        mse      rmse          r2          state     model  rank
10  1.076709   1.711769  1.250938   -1.114989       Honduras  base-gru  45.0
9   5.151525  36.352269  5.674105 -151.904592        Czechia  base-gru  69.0
11  6.763624  71.114486  7.785149 -180.369297  United States  base-gru  80.0
```


## Model xgboost - top states
```
         mae       mse      rmse        r2          state    model  rank
12  0.187744  0.077287  0.235859  0.850871        Czechia  xgboost   6.0
13  0.221089  0.108515  0.294814  0.961653       Honduras  xgboost  10.0
14  0.389511  0.307270  0.481740  0.133279  United States  xgboost  30.0
```


## Model xgboost - worst states
```
         mae       mse      rmse        r2          state    model  rank
12  0.187744  0.077287  0.235859  0.850871        Czechia  xgboost   6.0
13  0.221089  0.108515  0.294814  0.961653       Honduras  xgboost  10.0
14  0.389511  0.307270  0.481740  0.133279  United States  xgboost  30.0
```


## Model rf - top states
```
         mae       mse      rmse        r2          state model  rank
15  0.337485  0.205502  0.396268  0.147344        Czechia    rf  21.0
17  0.576162  0.659533  0.724398 -0.141790  United States    rf  34.0
16  0.730526  1.160530  0.905682  0.653951       Honduras    rf  35.0
```


## Model rf - worst states
```
         mae       mse      rmse        r2          state model  rank
15  0.337485  0.205502  0.396268  0.147344        Czechia    rf  21.0
17  0.576162  0.659533  0.724398 -0.141790  United States    rf  34.0
16  0.730526  1.160530  0.905682  0.653951       Honduras    rf  35.0
```


## Model lightgbm - top states
```
         mae       mse      rmse        r2          state     model  rank
18  0.190203  0.062647  0.244938  0.337185        Czechia  lightgbm  12.0
20  0.324301  0.237680  0.436611  0.597263  United States  lightgbm  20.0
19  0.350500  0.269026  0.444059  0.733319       Honduras  lightgbm  22.0
```


## Model lightgbm - worst states
```
         mae       mse      rmse        r2          state     model  rank
18  0.190203  0.062647  0.244938  0.337185        Czechia  lightgbm  12.0
20  0.324301  0.237680  0.436611  0.597263  United States  lightgbm  20.0
19  0.350500  0.269026  0.444059  0.733319       Honduras  lightgbm  22.0
```


## Per target metrics - model comparision
```
                          target            model       mae         mse       rmse           r2  rank
0           population_ages_0-14         lightgbm  0.243953    0.111892   0.316440     0.142252     2
1           population_ages_0-14          xgboost  0.317223    0.263046   0.416095     0.175259     5
2           population_ages_0-14               rf  0.592161    0.969578   0.724302     0.191938     8
3           population_ages_0-14        base-lstm  1.922747    4.474404   2.057418   -48.688203    13
4           population_ages_0-14  ensemble-arimas  1.936215    6.671412   2.447552   -48.459671    14
5           population_ages_0-14         base-gru  3.690128   25.531930   4.379970  -175.925521    17
6           population_ages_0-14       simple-rnn  9.973406  137.887947  11.569628 -1131.290468    21
7          population_ages_15-64          xgboost  0.258230    0.117073   0.314758     0.896320     3
8          population_ages_15-64         lightgbm  0.291435    0.242146   0.383341     0.918029     4
9          population_ages_15-64               rf  0.710269    0.791745   0.872566    -0.186599    10
10         population_ages_15-64  ensemble-arimas  1.818117    5.848781   2.389844    -6.447711    12
11         population_ages_15-64        base-lstm  2.685495   11.638940   2.844603   -16.767302    15
12         population_ages_15-64         base-gru  6.244902   70.139751   7.090791  -133.227016    20
13         population_ages_15-64       simple-rnn  6.001323   89.266596   7.485225   -30.408110    18
14  population_ages_65_and_above          xgboost  0.222891    0.112953   0.281561     0.874224     1
15  population_ages_65_and_above         lightgbm  0.329615    0.215315   0.425827     0.607486     6
16  population_ages_65_and_above               rf  0.341745    0.264242   0.429480     0.654166     7
17  population_ages_65_and_above  ensemble-arimas  0.695016    0.745557   0.825690    -0.475164     9
18  population_ages_65_and_above        base-lstm  0.847494    1.347387   0.868130    -0.156493    11
19  population_ages_65_and_above         base-gru  3.056828   13.506844   3.239431   -24.236341    16
20  population_ages_65_and_above       simple-rnn  6.076058   61.009224   7.354176  -185.628390    19
```


## Overall metrics - model comparision
```
        mae         mse      rmse          r2            model  rank
4  0.282579    0.182649  0.358818    0.606107          xgboost   4.0
6  0.308950    0.216494  0.402569    0.601876         lightgbm   8.0
5  0.592296    0.773862  0.734101    0.235064               rf  12.0
0  1.545094    4.648500  1.926008  -12.402332  ensemble-arimas  17.0
2  1.654010    4.733864  1.766519  -15.234803        base-lstm  19.0
3  4.158160   36.401365  4.741484 -102.563457         base-gru  24.0
1  7.711428  104.907490  9.288943 -326.339032       simple-rnn  28.0
```


