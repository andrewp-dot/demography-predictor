
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
Layers:             3

Learning rate:      0.0001
Epochs:             10

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
  'mae': 4.081057627996036,
  'mse': 17.35674006864974,
  'rmse': 4.166142108551956},
 {'feature': 'fertility rate, total',
  'mae': 0.12826524238164227,
  'mse': 0.018642877490544368,
  'rmse': 0.13653892298734588},
 {'feature': 'population, total',
  'mae': 148905371.1004741,
  'mse': 2.271328490958719e+16,
  'rmse': 150709272.8055815},
 {'feature': 'net migration',
  'mae': 252858.61986476253,
  'mse': 67372665073.06037,
  'rmse': 259562.44927388933},
 {'feature': 'arable land',
  'mae': 0.619773585903376,
  'mse': 0.4104808951481756,
  'rmse': 0.6406878297175431},
 {'feature': 'birth rate, crude',
  'mae': 0.23637472595026074,
  'mse': 0.07131733101588096,
  'rmse': 0.26705304906681177},
 {'feature': 'gdp growth',
  'mae': 4.138712210359185,
  'mse': 18.021864335788294,
  'rmse': 4.245216641796776},
 {'feature': 'death rate, crude',
  'mae': 1.9768542954772714,
  'mse': 4.865948429289307,
  'rmse': 2.205889487097961},
 {'feature': 'agricultural land',
  'mae': 1.2808094498967375,
  'mse': 1.906739443466007,
  'rmse': 1.3808473642897707},
 {'feature': 'rural population',
  'mae': 4.332164721955851,
  'mse': 19.242552749508317,
  'rmse': 4.386633418637613},
 {'feature': 'rural population growth',
  'mae': 1.051271369662665,
  'mse': 1.1140311781505585,
  'rmse': 1.0554767539602938},
 {'feature': 'age dependency ratio',
  'mae': 6.845545116821579,
  'mse': 49.68180800350436,
  'rmse': 7.048532329748113},
 {'feature': 'urban population',
  'mae': 1.2660953697363648,
  'mse': 1.9276628717052766,
  'rmse': 1.3884029932643032},
 {'feature': 'population growth',
  'mae': 1.4129363374977066,
  'mse': 2.2708266510140613,
  'rmse': 1.5069262261351952},
 {'feature': 'adolescent fertility rate',
  'mae': 6.969341763705013,
  'mse': 51.320147652323136,
  'rmse': 7.163808180871619},
 {'feature': 'life expectancy at birth, total',
  'mae': 3.1523407565922787,
  'mse': 10.245348224287385,
  'rmse': 3.200835550959684}]
```


## ARIMA per feature dict metrics

```
[{'feature': 'fertility rate, total',
  'mae': 0.5602684465104834,
  'mse': 0.316258756158468,
  'rmse': 0.5623688790806867},
 {'feature': 'population, total',
  'mae': 370941.1752914631,
  'mse': 144419086723.67618,
  'rmse': 380025.1132802623},
 {'feature': 'net migration',
  'mae': 11799.035255443758,
  'mse': 220010366.23300537,
  'rmse': 14832.746415718344},
 {'feature': 'arable land',
  'mae': 7.599092942554923,
  'mse': 66.92873344143486,
  'rmse': 8.18099831569686},
 {'feature': 'birth rate, crude',
  'mae': 1.6137441872246951,
  'mse': 2.739890180501081,
  'rmse': 1.6552613631995043},
 {'feature': 'gdp growth',
  'mae': 2.3302906016233607,
  'mse': 11.76939406080592,
  'rmse': 3.430655048355331},
 {'feature': 'death rate, crude',
  'mae': 0.7876972392376341,
  'mse': 1.424140195664994,
  'rmse': 1.1933734518854497},
 {'feature': 'agricultural land',
  'mae': 7.097390973488636,
  'mse': 66.0157673204349,
  'rmse': 8.12500875817589},
 {'feature': 'rural population',
  'mae': 0.3218632438825306,
  'mse': 0.13816579017481842,
  'rmse': 0.3717065915138154},
 {'feature': 'rural population growth',
  'mae': 0.7605269024315477,
  'mse': 1.4010090929110846,
  'rmse': 1.1836422993924662},
 {'feature': 'age dependency ratio',
  'mae': 12.42299426967498,
  'mse': 164.7233864839216,
  'rmse': 12.834460895726068},
 {'feature': 'urban population',
  'mae': 0.32186101216711904,
  'mse': 0.13816223589998436,
  'rmse': 0.3717018104609989},
 {'feature': 'population growth',
  'mae': 0.7147228357015981,
  'mse': 0.7939705124383835,
  'rmse': 0.891050230031048},
 {'feature': 'adolescent fertility rate',
  'mae': 1.674478821521778,
  'mse': 4.3043129574584675,
  'rmse': 2.0746838210817735},
 {'feature': 'life expectancy at birth, total',
  'mae': 3.392656952446354,
  'mse': 12.2279364014081,
  'rmse': 3.4968466368155324}]
```


# LSTM ARIMA comparision results

```
[{'best_model': 'LSTM', 'feature': 'agricultural land'},
 {'best_model': 'LSTM', 'feature': 'birth rate, crude'},
 {'best_model': 'LSTM', 'feature': 'life expectancy at birth, total'},
 {'best_model': 'ARIMA', 'feature': 'urban population'},
 {'best_model': 'ARIMA', 'feature': 'population, total'},
 {'best_model': 'LSTM', 'feature': 'fertility rate, total'},
 {'best_model': 'LSTM', 'feature': 'arable land'},
 {'best_model': 'ARIMA', 'feature': 'adolescent fertility rate'},
 {'best_model': 'ARIMA', 'feature': 'death rate, crude'},
 {'best_model': 'ARIMA', 'feature': 'rural population'},
 {'best_model': 'LSTM', 'feature': 'age dependency ratio'},
 {'best_model': 'ARIMA', 'feature': 'gdp growth'},
 {'best_model': 'ARIMA', 'feature': 'net migration'},
 {'best_model': 'ARIMA', 'feature': 'population growth'},
 {'best_model': 'LSTM', 'feature': 'rural population growth'}]
```


