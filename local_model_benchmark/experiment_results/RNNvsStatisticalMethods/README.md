
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
Epochs:             1

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
  'mae': 5.837528069814273,
  'mse': 34.54422478061755,
  'rmse': 5.877433519880727},
 {'feature': 'fertility rate, total',
  'mae': 0.03877761103337,
  'mse': 0.00198141661548589,
  'rmse': 0.04451310610916621},
 {'feature': 'population, total',
  'mae': 652394683.4759794,
  'mse': 4.549243029561526e+17,
  'rmse': 674480765.445652},
 {'feature': 'net migration',
  'mae': 486914.9248923667,
  'mse': 239913242503.54044,
  'rmse': 489809.39405399363},
 {'feature': 'arable land',
  'mae': 24.938165330288925,
  'mse': 622.5359745072923,
  'rmse': 24.950670822791363},
 {'feature': 'birth rate, crude',
  'mae': 4.924754907526077,
  'mse': 24.563787864536355,
  'rmse': 4.956186827041163},
 {'feature': 'gdp growth',
  'mae': 3.309344092422618,
  'mse': 22.383145604042838,
  'rmse': 4.731082920858906},
 {'feature': 'death rate, crude',
  'mae': 2.3542043226721385,
  'mse': 6.729050594372914,
  'rmse': 2.594041363273322},
 {'feature': 'agricultural land',
  'mae': 12.221858602401293,
  'mse': 152.71422660279623,
  'rmse': 12.357759772822751},
 {'feature': 'rural population',
  'mae': 4.1508603218893265,
  'mse': 17.586966148584583,
  'rmse': 4.193681693760816},
 {'feature': 'rural population growth',
  'mae': 2.4490176054746127,
  'mse': 6.796868344952955,
  'rmse': 2.607080425486133},
 {'feature': 'age dependency ratio',
  'mae': 4.371474889824236,
  'mse': 21.5785505350291,
  'rmse': 4.6452718472689085},
 {'feature': 'urban population',
  'mae': 1.3979942072630014,
  'mse': 2.8118430278127047,
  'rmse': 1.6768551004224261},
 {'feature': 'population growth',
  'mae': 2.434466963533747,
  'mse': 6.647047763824273,
  'rmse': 2.5781869140588456},
 {'feature': 'adolescent fertility rate',
  'mae': 15.01700658112764,
  'mse': 227.56151551017993,
  'rmse': 15.085142210472526},
 {'feature': 'life expectancy at birth, total',
  'mae': 1.1867170825075348,
  'mse': 1.571636490329837,
  'rmse': 1.253649269265466}]
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
[{'best_model': 'ARIMA', 'feature': 'population, total'},
 {'best_model': 'LSTM', 'feature': 'life expectancy at birth, total'},
 {'best_model': 'ARIMA', 'feature': 'rural population growth'},
 {'best_model': 'ARIMA', 'feature': 'death rate, crude'},
 {'best_model': 'ARIMA', 'feature': 'gdp growth'},
 {'best_model': 'ARIMA', 'feature': 'agricultural land'},
 {'best_model': 'ARIMA', 'feature': 'arable land'},
 {'best_model': 'ARIMA', 'feature': 'net migration'},
 {'best_model': 'ARIMA', 'feature': 'rural population'},
 {'best_model': 'LSTM', 'feature': 'fertility rate, total'},
 {'best_model': 'LSTM', 'feature': 'age dependency ratio'},
 {'best_model': 'ARIMA', 'feature': 'urban population'},
 {'best_model': 'ARIMA', 'feature': 'birth rate, crude'},
 {'best_model': 'ARIMA', 'feature': 'population growth'},
 {'best_model': 'ARIMA', 'feature': 'adolescent fertility rate'}]
```


