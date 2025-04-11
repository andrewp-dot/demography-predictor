
# DataUsedForTraining

**Description:** Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.


## ## Model comparision prediction plot
In the next feagure you can see each model predictions compared to each other and the reference data.

![## Model comparision prediction plot](./plots/state_prediction_comparions.png)

## Per target metrics - model comparision
```
                             target          mae           mse         rmse            r2    state               model   rank
4                 birth rate, crude     0.122480  2.725370e-02     0.165087     -0.022014  Czechia  single_state_model    4.0
12                population growth     0.406053  7.414566e-01     0.861079     -0.204974  Czechia  single_state_model   28.0
7                 agricultural land     0.372203  1.722291e-01     0.415005     -4.251984  Czechia  single_state_model   29.0
42                population growth     0.584384  7.112388e-01     0.843350     -0.155866  Czechia    all_states_model   30.0
1             fertility rate, total     0.228885  5.620030e-02     0.237066    -14.551198  Czechia  single_state_model   33.0
11                 urban population     0.420660  2.261273e-01     0.475528     -3.654441  Czechia  single_state_model   34.0
8                  rural population     0.546173  3.435103e-01     0.586097     -6.070566  Czechia  single_state_model   45.0
39          rural population growth     0.958827  1.008848e+00     1.004414     -0.433937  Czechia    all_states_model   46.0
16            fertility rate, total     0.450387  2.072302e-01     0.455225    -56.342708  Czechia  group_states_model   47.0
14  life expectancy at birth, total     0.924923  9.777479e-01     0.988811     -0.966073  Czechia  single_state_model   49.0
38                 rural population     0.746991  5.935297e-01     0.770409    -11.216784  Czechia    all_states_model   50.0
31            fertility rate, total     0.539795  2.933201e-01     0.541590    -80.164678  Czechia    all_states_model   54.0
9           rural population growth     0.818025  1.371529e+00     1.171123     -0.949439  Czechia  single_state_model   57.0
27                population growth     0.880025  1.355361e+00     1.164200     -1.202657  Czechia  group_states_model   59.0
6                 death rate, crude     0.903323  1.991139e+00     1.411077     -0.580269  Czechia  single_state_model   61.0
41                 urban population     0.971457  1.024333e+00     1.012093    -20.084122  Czechia    all_states_model   71.0
24          rural population growth     1.124336  1.908754e+00     1.381576     -1.713029  Czechia  group_states_model   71.0
36                death rate, crude     1.367755  2.105013e+00     1.450866     -0.670645  Czechia    all_states_model   72.0
37                agricultural land     0.952483  1.216573e+00     1.102984    -36.098397  Czechia    all_states_model   76.0
29  life expectancy at birth, total     1.346060  2.309097e+00     1.519571     -3.643174  Czechia  group_states_model   80.0
13        adolescent fertility rate     1.408659  2.602405e+00     1.613197     -3.119980  Czechia  single_state_model   83.0
3                       arable land     1.035314  1.077781e+00     1.038162   -165.501657  Czechia  single_state_model   83.0
5                        gdp growth     1.997173  1.348604e+01     3.672334     -0.130349  Czechia  single_state_model   86.0
44  life expectancy at birth, total     1.566530  3.064782e+00     1.750652     -5.162720  Czechia    all_states_model   91.0
35                       gdp growth     3.901197  1.652089e+01     4.064590     -0.384720  Czechia    all_states_model   98.0
21                death rate, crude     2.431601  7.292975e+00     2.700551     -4.788076  Czechia  group_states_model  102.0
20                       gdp growth     4.218326  1.853766e+01     4.305538     -0.553758  Czechia  group_states_model  105.0
19                birth rate, crude     2.349253  5.536939e+00     2.353070   -206.635230  Czechia  group_states_model  112.0
22                agricultural land     2.396722  6.481946e+00     2.545967   -196.661591  Czechia  group_states_model  116.0
34                birth rate, crude     2.516308  6.434728e+00     2.536677   -240.302310  Czechia    all_states_model  118.0
30                             year     6.500000  4.283333e+01     6.544718    -13.685714  Czechia    all_states_model  124.5
26                 urban population     3.583743  1.286102e+01     3.586227   -263.721924  Czechia  group_states_model  126.0
0                              year     6.500000  4.516667e+01     6.720615    -14.485714  Czechia  single_state_model  127.5
23                 rural population     3.646314  1.329638e+01     3.646419   -272.682915  Czechia  group_states_model  130.0
17                    net migration  1725.549989  3.162010e+06  1778.204268     -0.503277  Czechia  group_states_model  137.0
10             age dependency ratio     9.427045  9.316283e+01     9.652089    -21.461237  Czechia  single_state_model  140.0
32                    net migration  1748.490714  3.229432e+06  1797.062020     -0.535331  Czechia    all_states_model  141.0
28        adolescent fertility rate     7.700211  5.964135e+01     7.722781    -93.420810  Czechia  group_states_model  141.0
15                             year     9.500000  9.316667e+01     9.652288    -30.942857  Czechia  group_states_model  144.0
2                     net migration  1334.133833  4.624110e+06  2150.374494     -1.198387  Czechia  single_state_model  147.0
25             age dependency ratio    13.599482  1.900106e+02    13.784432    -44.810896  Czechia  group_states_model  152.0
33                      arable land     7.791464  6.215294e+01     7.883713  -9600.731682  Czechia    all_states_model  152.0
40             age dependency ratio    16.085133  2.637013e+02    16.238883    -62.577481  Czechia    all_states_model  160.0
18                      arable land    11.896312  1.416331e+02    11.900972 -21879.274468  Czechia  group_states_model  162.0
43        adolescent fertility rate    15.370504  2.368862e+02    15.391107   -374.024828  Czechia    all_states_model  166.0
```


## Overall metrics - model comparision
```
          mae            mse        rmse           r2    state               model  rank
1  119.378184  210837.643936  122.994872 -1537.126491  Czechia  group_states_model   7.0
0   90.616317  308284.791036  145.292118   -15.809885  Czechia  single_state_model   8.0
2  120.556236  215338.030055  123.879871  -696.415281  Czechia    all_states_model   9.0
```


