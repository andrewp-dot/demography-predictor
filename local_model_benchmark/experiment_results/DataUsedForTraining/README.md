
# DataUsedForTraining

**Description:** Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.


## ## Model comparision prediction plot
In the next feagure you can see each model predictions compared to each other and the reference data.

![## Model comparision prediction plot](./plots/state_prediction_comparions.png)

## Per target metrics - model comparision
```
                     target           mae           mse          rmse            r2    state               model   rank
18    fertility rate, total      0.125095  1.677263e-02      0.129509     -3.641157  Czechia    all_states_model   19.0
8         population growth      0.426526  6.638808e-01      0.814789     -0.078902  Czechia  single_state_model   20.0
5         agricultural land      0.310280  1.313032e-01      0.362358     -3.003984  Czechia  single_state_model   21.0
0     fertility rate, total      0.313834  1.025096e-01      0.320171    -27.365459  Czechia  single_state_model   26.0
26        population growth      0.792171  7.917190e-01      0.889786     -0.286658  Czechia    all_states_model   28.0
9     fertility rate, total      0.398740  1.628433e-01      0.403538    -44.060411  Czechia  group_states_model   32.0
7          urban population      0.471166  2.715304e-01      0.521086     -4.588985  Czechia  single_state_model   33.0
17        population growth      0.832301  1.186747e+00      1.089379     -0.928635  Czechia  group_states_model   34.0
6   rural population growth      0.747903  1.258004e+00      1.121608     -0.788078  Czechia  single_state_model   37.0
22        death rate, crude      1.187074  1.559611e+00      1.248844     -0.237786  Czechia    all_states_model   42.0
15  rural population growth      1.034247  1.640514e+00      1.280826     -1.331764  Czechia  group_states_model   47.0
25         urban population      1.084119  1.207377e+00      1.098807    -23.851767  Czechia    all_states_model   48.0
4         death rate, crude      0.944787  2.100336e+00      1.449254     -0.666934  Czechia  single_state_model   49.0
2               arable land      1.104534  1.226393e+00      1.107426   -188.459981  Czechia  single_state_model   55.0
12               gdp growth      2.771715  1.154899e+01      3.398380      0.032007  Czechia  group_states_model   59.0
3                gdp growth      2.624101  1.179854e+01      3.434901      0.011090  Czechia  single_state_model   61.0
24  rural population growth      1.533944  2.486146e+00      1.576752     -2.533711  Czechia    all_states_model   62.0
20              arable land      1.193884  1.646148e+00      1.283023   -253.306158  Czechia    all_states_model   66.0
13        death rate, crude      1.884223  4.877797e+00      2.208574     -2.871268  Czechia  group_states_model   66.0
23        agricultural land      1.332259  1.929207e+00      1.388959    -57.829574  Czechia    all_states_model   67.0
21               gdp growth      3.681825  1.492744e+01      3.863604     -0.251162  Czechia    all_states_model   70.0
16         urban population      3.700789  1.411101e+01      3.756463   -289.450845  Czechia  group_states_model   88.0
10            net migration   2740.544040  8.478274e+06   2911.747579     -3.030726  Czechia  group_states_model   89.0
1             net migration   2797.172938  8.928029e+06   2987.980678     -3.244548  Czechia  single_state_model   93.0
14        agricultural land      8.005597  6.425559e+01      8.015958  -1958.421230  Czechia  group_states_model   95.0
11              arable land     10.707030  1.146569e+02     10.707798 -17711.842294  Czechia  group_states_model   99.0
19            net migration  29951.977483  2.096671e+09  45789.421101   -995.795721  Czechia    all_states_model  106.0
```


## Overall metrics - model comparision
```
           mae           mse         rmse           r2    state               model  rank
1   307.764298  9.420540e+05   326.956499 -2223.545018  Czechia  group_states_model   6.0
0   311.568452  9.920051e+05   333.012475   -25.353976  Czechia  single_state_model   7.0
2  3329.211984  2.329635e+08  5088.988932  -148.637077  Czechia    all_states_model  11.0
```


