
# DataUsedForTraining

**Description:** Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.


## ## Model comparision prediction plot
In the next feagure you can see each model predictions compared to each other and the reference data.

![## Model comparision prediction plot](./plots/state_prediction_comparions.png)

## Per target metrics - model comparision
```
                     target           mae           mse          rmse            r2    state               model   rank
18    fertility rate, total      0.125095  1.677263e-02      0.129509     -3.641157  Czechia    all_states_model   17.0
8         population growth      0.426526  6.638808e-01      0.814789     -0.078902  Czechia  single_state_model   19.0
5         agricultural land      0.310280  1.313032e-01      0.362358     -3.003984  Czechia  single_state_model   23.0
9     fertility rate, total      0.180050  3.496388e-02      0.186986     -8.674863  Czechia  group_states_model   24.0
26        population growth      0.792171  7.917190e-01      0.889786     -0.286658  Czechia    all_states_model   28.0
15  rural population growth      0.867053  8.575049e-01      0.926016     -0.218824  Czechia  group_states_model   28.0
0     fertility rate, total      0.313834  1.025096e-01      0.320171    -27.365459  Czechia  single_state_model   30.0
7          urban population      0.471166  2.715304e-01      0.521086     -4.588985  Czechia  single_state_model   33.0
17        population growth      0.916969  8.812153e-01      0.938731     -0.432102  Czechia  group_states_model   35.0
6   rural population growth      0.747903  1.258004e+00      1.121608     -0.788078  Czechia  single_state_model   40.0
22        death rate, crude      1.187074  1.559611e+00      1.248844     -0.237786  Czechia    all_states_model   44.0
4         death rate, crude      0.944787  2.100336e+00      1.449254     -0.666934  Czechia  single_state_model   51.0
25         urban population      1.084119  1.207377e+00      1.098807    -23.851767  Czechia    all_states_model   51.0
2               arable land      1.104534  1.226393e+00      1.107426   -188.459981  Czechia  single_state_model   57.0
3                gdp growth      2.624101  1.179854e+01      3.434901      0.011090  Czechia  single_state_model   58.0
24  rural population growth      1.533944  2.486146e+00      1.576752     -2.533711  Czechia    all_states_model   62.0
21               gdp growth      3.681825  1.492744e+01      3.863604     -0.251162  Czechia    all_states_model   65.0
20              arable land      1.193884  1.646148e+00      1.283023   -253.306158  Czechia    all_states_model   66.0
23        agricultural land      1.332259  1.929207e+00      1.388959    -57.829574  Czechia    all_states_model   67.0
13        death rate, crude      2.208486  6.062105e+00      2.462134     -3.811195  Czechia  group_states_model   70.0
12               gdp growth      5.023620  2.616884e+01      5.115549     -1.193375  Czechia  group_states_model   73.0
1             net migration   2797.172938  8.928029e+06   2987.980678     -3.244548  Czechia  single_state_model   88.0
16         urban population      7.236550  5.299043e+01      7.279453  -1089.716571  Czechia  group_states_model   91.0
10            net migration   2916.356492  9.945702e+06   3153.680662     -3.728368  Czechia  group_states_model   93.0
14        agricultural land      9.014088  8.198114e+01      9.054344  -2498.947394  Czechia  group_states_model   95.0
11              arable land     10.817089  1.170113e+02     10.817177 -18075.559301  Czechia  group_states_model   99.0
19            net migration  29951.977483  2.096671e+09  45789.421101   -995.795721  Czechia    all_states_model  105.0
```


## Overall metrics - model comparision
```
           mae           mse         rmse           r2    state               model  rank
0   311.568452  9.920051e+05   333.012475   -25.353976  Czechia  single_state_model   4.0
1   328.068933  1.105110e+06   354.495672 -2409.142444  Czechia  group_states_model   9.0
2  3329.211984  2.329635e+08  5088.988932  -148.637077  Czechia    all_states_model  11.0
```


