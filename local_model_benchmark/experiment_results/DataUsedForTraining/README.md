
# DataUsedForTraining

**Description:** Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.

## Per target metrics - model comparision
```
                             target            mae           mse           rmse            r2    state               model   rank
4                 birth rate, crude       0.159865  3.136621e-02       0.177105     -0.176233  Czechia  single_state_model    7.0
37                agricultural land       0.249985  1.107308e-01       0.332762     -2.376644  Czechia    all_states_model   20.0
12                population growth       0.396912  7.161546e-01       0.846259     -0.163854  Czechia  single_state_model   30.0
1             fertility rate, total       0.288813  8.752611e-02       0.295848    -23.219369  Czechia  single_state_model   32.0
41                 urban population       0.460133  2.432910e-01       0.493245     -4.007725  Czechia    all_states_model   34.0
42                population growth       0.432207  7.505279e-01       0.866330     -0.219716  Czechia    all_states_model   36.0
11                 urban population       0.459884  2.614152e-01       0.511288     -4.380780  Czechia  single_state_model   37.0
31            fertility rate, total       0.381260  1.459950e-01       0.382093    -39.398315  Czechia    all_states_model   42.0
8                  rural population       0.479921  2.698299e-01       0.519452     -4.553983  Czechia  single_state_model   42.0
39          rural population growth       0.938454  9.713706e-01       0.985581     -0.380669  Czechia    all_states_model   48.0
7                 agricultural land       0.484810  2.709208e-01       0.520501     -7.261506  Czechia  single_state_model   48.0
9           rural population growth       0.880509  1.472640e+00       1.213524     -1.093153  Czechia  single_state_model   51.0
34                birth rate, crude       0.488845  3.038181e-01       0.551197    -10.393179  Czechia    all_states_model   52.0
6                 death rate, crude       0.914653  2.021456e+00       1.421779     -0.604331  Czechia  single_state_model   54.0
13        adolescent fertility rate       1.307440  2.293227e+00       1.514340     -2.630507  Czechia  single_state_model   66.0
14  life expectancy at birth, total       1.315088  2.173695e+00       1.474346     -3.370906  Czechia  single_state_model   67.0
16            fertility rate, total       0.550635  3.075413e-01       0.554564    -84.099812  Czechia  group_states_model   67.0
36                death rate, crude       1.453148  2.569918e+00       1.603097     -1.039618  Czechia    all_states_model   68.0
27                population growth       1.617596  3.280300e+00       1.811160     -4.330961  Czechia  group_states_model   80.0
3                       arable land       1.254510  1.578840e+00       1.256519   -242.908047  Czechia  single_state_model   83.0
29  life expectancy at birth, total       1.739407  3.706667e+00       1.925271     -6.453433  Czechia  group_states_model   87.0
5                        gdp growth       2.121741  1.216120e+01       3.487291     -0.019306  Czechia  single_state_model   87.0
21                death rate, crude       2.020609  5.445014e+00       2.333455     -3.321440  Czechia  group_states_model   88.0
30                             year       2.538064  6.630154e+00       2.574909     -1.273196  Czechia    all_states_model   90.0
38                 rural population       1.532365  2.436469e+00       1.560919    -49.150504  Czechia    all_states_model   90.0
24          rural population growth       2.101584  5.158514e+00       2.271236     -6.332112  Czechia  group_states_model   93.0
20                       gdp growth       2.731216  1.228646e+01       3.505205     -0.029805  Czechia  group_states_model   93.0
35                       gdp growth       3.531945  1.440949e+01       3.795983     -0.207749  Czechia    all_states_model  103.0
23                 rural population       1.991381  3.967889e+00       1.991956    -80.672141  Czechia  group_states_model  103.0
44  life expectancy at birth, total       2.664035  7.524967e+00       2.743167    -14.131342  Czechia    all_states_model  106.0
19                birth rate, crude       2.765322  7.665270e+00       2.768622   -286.447629  Czechia  group_states_model  125.0
22                agricultural land       3.234474  1.157552e+01       3.402281   -351.985879  Czechia  group_states_model  129.0
0                              year       8.965970  8.429833e+01       9.181412    -27.902284  Czechia  single_state_model  134.0
26                 urban population       3.558805  1.270072e+01       3.563808   -260.422480  Czechia  group_states_model  135.0
28        adolescent fertility rate       6.641368  4.432659e+01       6.657821    -69.175348  Czechia  group_states_model  135.0
15                             year      10.730582  1.168211e+02      10.808379    -39.052932  Czechia  group_states_model  140.0
10             age dependency ratio      11.152354  1.287366e+02      11.346214    -30.037945  Czechia  single_state_model  141.0
40             age dependency ratio      11.592890  1.374831e+02      11.725319    -32.146705  Czechia    all_states_model  145.0
33                      arable land       7.940462  6.411010e+01       8.006878  -9903.084892  Czechia    all_states_model  147.0
25             age dependency ratio      13.737552  1.936026e+02      13.914115    -45.676918  Czechia  group_states_model  154.0
18                      arable land      12.866948  1.657418e+02      12.874074 -25603.713864  Czechia  group_states_model  165.0
2                     net migration   17087.098752  2.950351e+08   17176.586249   -139.265081  Czechia  single_state_model  165.0
43        adolescent fertility rate      27.533924  7.611872e+02      27.589622  -1204.068625  Czechia    all_states_model  167.0
17                    net migration  159053.979272  2.529866e+10  159055.526955 -12026.445257  Czechia  group_states_model  175.0
32                    net migration  211195.206883  4.464819e+10  211301.183648 -21225.564951  Czechia    all_states_model  179.0
```


## Overall metrics - model comparision
```
            mae           mse          rmse           r2    state               model  rank
0   1141.152081  1.966902e+07   1147.356808   -32.505819  Czechia  single_state_model   4.0
1  10608.017783  1.686577e+09  10608.260594 -2591.210668  Czechia  group_states_model   9.0
2  14083.796307  2.976546e+09  14090.959650 -2165.829589  Czechia    all_states_model  11.0
```


