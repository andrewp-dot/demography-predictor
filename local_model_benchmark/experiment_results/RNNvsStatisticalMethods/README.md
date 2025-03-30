
# RNNvsStatisticalMethods

**Description:** Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. ARIMA model is trained just to predict 1 feauture from all features and LSTM predict all features at once.

## Features
```
year
fertility rate, total
population, total
net migration
arable land
birth rate, crude
gdp growth
death rate, crude
agricultural land
rural population
rural population growth
age dependency ratio
urban population
population growth
adolescent fertility rate
life expectancy at birth, total
```

## Hyperparameters

```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             2

Learning rate:      0.0001
Epochs:             30

Bidirectional:      False
```


## Arima evaluation: fertility rate, total


![Arima evaluation: fertility rate, total](./plots/arima_evaluation_fertility_rate,_total.png)


## Arima evaluation: population, total


![Arima evaluation: population, total](./plots/arima_evaluation_population,_total.png)


## Arima evaluation: net migration


![Arima evaluation: net migration](./plots/arima_evaluation_net_migration.png)


## Arima evaluation: arable land


![Arima evaluation: arable land](./plots/arima_evaluation_arable_land.png)


## Arima evaluation: birth rate, crude


![Arima evaluation: birth rate, crude](./plots/arima_evaluation_birth_rate,_crude.png)


## Arima evaluation: gdp growth


![Arima evaluation: gdp growth](./plots/arima_evaluation_gdp_growth.png)


## Arima evaluation: death rate, crude


![Arima evaluation: death rate, crude](./plots/arima_evaluation_death_rate,_crude.png)


## Arima evaluation: agricultural land


![Arima evaluation: agricultural land](./plots/arima_evaluation_agricultural_land.png)


## Arima evaluation: rural population


![Arima evaluation: rural population](./plots/arima_evaluation_rural_population.png)


## Arima evaluation: rural population growth


![Arima evaluation: rural population growth](./plots/arima_evaluation_rural_population_growth.png)


## Arima evaluation: age dependency ratio


![Arima evaluation: age dependency ratio](./plots/arima_evaluation_age_dependency_ratio.png)


## Arima evaluation: urban population


![Arima evaluation: urban population](./plots/arima_evaluation_urban_population.png)


## Arima evaluation: population growth


![Arima evaluation: population growth](./plots/arima_evaluation_population_growth.png)


## Arima evaluation: adolescent fertility rate


![Arima evaluation: adolescent fertility rate](./plots/arima_evaluation_adolescent_fertility_rate.png)


## Arima evaluation: life expectancy at birth, total


![Arima evaluation: life expectancy at birth, total](./plots/arima_evaluation_life_expectancy_at_birth,_total.png)

## LSTM per feature dict metrics

```
[{'feature': 'year',
  'mae': 8.011534730593363,
  'mse': 67.4754916225334,
  'rmse': 8.214346694809842},
 {'feature': 'fertility rate, total',
  'mae': 0.2721751286586126,
  'mse': 0.07788073153943034,
  'rmse': 0.27907119439209477},
 {'feature': 'population, total',
  'mae': 199464.89359011254,
  'mse': 43972264840.36851,
  'rmse': 209695.64811976548},
 {'feature': 'net migration',
  'mae': 18081.86635327339,
  'mse': 329339818.6708778,
  'rmse': 18147.72213449605},
 {'feature': 'arable land',
  'mae': 1.3671420790838174,
  'mse': 1.8746714286730832,
  'rmse': 1.3691864112213075},
 {'feature': 'birth rate, crude',
  'mae': 0.12435846130053196,
  'mse': 0.024307039650216435,
  'rmse': 0.15590715073471273},
 {'feature': 'gdp growth',
  'mae': 1.9819849305958932,
  'mse': 13.073703739115585,
  'rmse': 3.615757699171169},
 {'feature': 'death rate, crude',
  'mae': 0.9060585811734203,
  'mse': 1.9984354735951744,
  'rmse': 1.4136603105396905},
 {'feature': 'agricultural land',
  'mae': 0.5412645406388042,
  'mse': 0.3280269735703572,
  'rmse': 0.5727363909953315},
 {'feature': 'rural population',
  'mae': 0.5616682877937951,
  'mse': 0.3602924278736092,
  'rmse': 0.6002436404274595},
 {'feature': 'rural population growth',
  'mae': 0.9252969038380066,
  'mse': 1.5560125334308428,
  'rmse': 1.247402314183697},
 {'feature': 'age dependency ratio',
  'mae': 10.885426371790233,
  'mse': 122.7602856102052,
  'rmse': 11.079724076447265},
 {'feature': 'urban population',
  'mae': 0.4488230725030083,
  'mse': 0.2506003237405389,
  'rmse': 0.5005999637839968},
 {'feature': 'population growth',
  'mae': 0.4070495940301933,
  'mse': 0.7450192100188971,
  'rmse': 0.8631449530750308},
 {'feature': 'adolescent fertility rate',
  'mae': 1.3286173085235065,
  'mse': 2.3776751472620945,
  'rmse': 1.5419711888560352},
 {'feature': 'life expectancy at birth, total',
  'mae': 1.1401956778425724,
  'mse': 1.5411717991936553,
  'rmse': 1.2414394061707785}]
```


## ARIMA per feature dict metrics

```
[{'feature': 'fertility rate, total',
  'mae': 0.5610371151272006,
  'mse': 0.31719009098543766,
  'rmse': 0.5631963165588334},
 {'feature': 'population, total',
  'mae': 387105.6294702664,
  'mse': 158017707444.21643,
  'rmse': 397514.41161826625},
 {'feature': 'net migration',
  'mae': 11858.603298119318,
  'mse': 220425926.98139465,
  'rmse': 14846.74802714031},
 {'feature': 'arable land',
  'mae': 7.543852417410421,
  'mse': 65.95635123463647,
  'rmse': 8.121351564526465},
 {'feature': 'birth rate, crude',
  'mae': 1.6370507711805156,
  'mse': 2.794585456871373,
  'rmse': 1.6717013659357263},
 {'feature': 'gdp growth',
  'mae': 2.0903225898000457,
  'mse': 11.808555973046964,
  'rmse': 3.436357951821516},
 {'feature': 'death rate, crude',
  'mae': 0.8010926557687856,
  'mse': 1.447441213115594,
  'rmse': 1.2030965103081275},
 {'feature': 'agricultural land',
  'mae': 7.186411114666996,
  'mse': 66.31489185784038,
  'rmse': 8.143395597528121},
 {'feature': 'rural population',
  'mae': 0.322564822168171,
  'mse': 0.13863078367323764,
  'rmse': 0.37233155073568186},
 {'feature': 'rural population growth',
  'mae': 0.7607792299874466,
  'mse': 1.403147266966067,
  'rmse': 1.1845451730373422},
 {'feature': 'age dependency ratio',
  'mae': 12.436515066674438,
  'mse': 165.0574678962462,
  'rmse': 12.84746931875092},
 {'feature': 'urban population',
  'mae': 0.32256481105406937,
  'mse': 0.13863073484542696,
  'rmse': 0.37233148516533887},
 {'feature': 'population growth',
  'mae': 0.7136306223958219,
  'mse': 0.7929854899479004,
  'rmse': 0.8904973273109248},
 {'feature': 'adolescent fertility rate',
  'mae': 2.1041557735861125,
  'mse': 6.1044579133946995,
  'rmse': 2.4707201204091693},
 {'feature': 'life expectancy at birth, total',
  'mae': 3.3860838841241665,
  'mse': 12.16965027482689,
  'rmse': 3.4885025834628376}]
```


# LSTM ARIMA comparision results

```
[{'best_model': 'ARIMA', 'feature': 'gdp growth'},
 {'best_model': 'FineTunableLSTM', 'feature': 'age dependency ratio'},
 {'best_model': 'FineTunableLSTM', 'feature': 'agricultural land'},
 {'best_model': 'FineTunableLSTM', 'feature': 'birth rate, crude'},
 {'best_model': 'ARIMA', 'feature': 'rural population growth'},
 {'best_model': 'ARIMA', 'feature': 'urban population'},
 {'best_model': 'FineTunableLSTM',
  'feature': 'life expectancy at birth, total'},
 {'best_model': 'FineTunableLSTM', 'feature': 'arable land'},
 {'best_model': 'ARIMA', 'feature': 'death rate, crude'},
 {'best_model': 'FineTunableLSTM', 'feature': 'fertility rate, total'},
 {'best_model': 'FineTunableLSTM', 'feature': 'adolescent fertility rate'},
 {'best_model': 'ARIMA', 'feature': 'net migration'},
 {'best_model': 'FineTunableLSTM', 'feature': 'population growth'},
 {'best_model': 'FineTunableLSTM', 'feature': 'population, total'},
 {'best_model': 'ARIMA', 'feature': 'rural population'}]
```


