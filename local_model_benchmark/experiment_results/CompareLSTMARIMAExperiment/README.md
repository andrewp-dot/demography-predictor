
# CompareLSTMARIMAExperiment

**Description:** The goal is to find the optimal parameters for the given BaseLSTM model.

# LSTM model parameters
Hyperparameters:
```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             20

Bidirectional:      False
```
# LSTM & ARIMA Comparision: Feature: fertility rate, total
Comparision of LSTM and ARIMA model of predicting feature fertility rate, total. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_fertility_rate,_total.png)


## RNN evaluation - fertility rate, total


![RNN evaluation - fertility rate, total](./plots/lstm_evaluation_fertility_rate,_total.png)

### Overall metrics (ARIMA)
```
{'mae': 0.5602717284986595,
 'mse': 0.3162614490249736,
 'r2': -86.51277605610322,
 'rmse': 0.5623712732928076}
```

### Overall metrics (RNN)
```
{'mae': 0.22035458644231162,
 'mse': 0.05188574602462187,
 'r2': -13.357316348089006,
 'rmse': 0.22778442884583194}
```

# LSTM & ARIMA Comparision: Feature: population, total
Comparision of LSTM and ARIMA model of predicting feature population, total. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_population,_total.png)


## RNN evaluation - population, total


![RNN evaluation - population, total](./plots/lstm_evaluation_population,_total.png)

### Overall metrics (ARIMA)
```
{'mae': 370941.1752923193,
 'mse': 144419086724.24503,
 'r2': -33.72977117822156,
 'rmse': 380025.1132810107}
```

### Overall metrics (RNN)
```
{'mae': 167174.5,
 'mse': 31948799135.166668,
 'r2': -6.683018280692578,
 'rmse': 178742.27014102362}
```

# LSTM & ARIMA Comparision: Feature: net migration
Comparision of LSTM and ARIMA model of predicting feature net migration. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_net_migration.png)


## RNN evaluation - net migration


![RNN evaluation - net migration](./plots/lstm_evaluation_net_migration.png)

### Overall metrics (ARIMA)
```
{'mae': 11799.035064214704,
 'mse': 220010359.513285,
 'r2': -103.59694254393696,
 'rmse': 14832.746189201951}
```

### Overall metrics (RNN)
```
{'mae': 17936.0693359375,
 'mse': 322660911.82640773,
 'r2': -152.3988896256641,
 'rmse': 17962.764593079977}
```

# LSTM & ARIMA Comparision: Feature: arable land
Comparision of LSTM and ARIMA model of predicting feature arable land. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_arable_land.png)


## RNN evaluation - arable land


![RNN evaluation - arable land](./plots/lstm_evaluation_arable_land.png)

### Overall metrics (ARIMA)
```
{'mae': 7.599400081005445,
 'mse': 66.93202401029895,
 'r2': -10339.03160682162,
 'rmse': 8.181199423696928}
```

### Overall metrics (RNN)
```
{'mae': 0.19794886142566526,
 'mse': 0.041229869317447966,
 'r2': -5.3694196939560435,
 'rmse': 0.20305139575350858}
```

# LSTM & ARIMA Comparision: Feature: birth rate, crude
Comparision of LSTM and ARIMA model of predicting feature birth rate, crude. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_birth_rate,_crude.png)


## RNN evaluation - birth rate, crude


![RNN evaluation - birth rate, crude](./plots/lstm_evaluation_birth_rate,_crude.png)

### Overall metrics (ARIMA)
```
{'mae': 1.6137452853810748,
 'mse': 2.7398928820067305,
 'r2': -101.74598307525268,
 'rmse': 1.6552621792352806}
```

### Overall metrics (RNN)
```
{'mae': 0.15959946314493786,
 'mse': 0.03302613784946507,
 'r2': -0.23848016935494365,
 'rmse': 0.18173094906885032}
```

# LSTM & ARIMA Comparision: Feature: gdp growth
Comparision of LSTM and ARIMA model of predicting feature gdp growth. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_gdp_growth.png)


## RNN evaluation - gdp growth


![RNN evaluation - gdp growth](./plots/lstm_evaluation_gdp_growth.png)

### Overall metrics (ARIMA)
```
{'mae': 2.3302902376052876,
 'mse': 11.769394255040899,
 'r2': 0.013533325648003247,
 'rmse': 3.4306550766640616}
```

### Overall metrics (RNN)
```
{'mae': 2.1244088768664735,
 'mse': 12.223515609416348,
 'r2': -0.024529430386454942,
 'rmse': 3.496214468452464}
```

# LSTM & ARIMA Comparision: Feature: death rate, crude
Comparision of LSTM and ARIMA model of predicting feature death rate, crude. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_death_rate,_crude.png)


## RNN evaluation - death rate, crude


![RNN evaluation - death rate, crude](./plots/lstm_evaluation_death_rate,_crude.png)

### Overall metrics (ARIMA)
```
{'mae': 0.787689799681974,
 'mse': 1.4241329879452902,
 'r2': -0.13026427614705516,
 'rmse': 1.193370431988865}
```

### Overall metrics (RNN)
```
{'mae': 0.9345354715983074,
 'mse': 2.0724726551958765,
 'r2': -0.6448195676157742,
 'rmse': 1.4396085076144405}
```

# LSTM & ARIMA Comparision: Feature: agricultural land
Comparision of LSTM and ARIMA model of predicting feature agricultural land. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_agricultural_land.png)


## RNN evaluation - agricultural land


![RNN evaluation - agricultural land](./plots/lstm_evaluation_agricultural_land.png)

### Overall metrics (ARIMA)
```
{'mae': 7.098129974948008,
 'mse': 66.01803320568136,
 'r2': -2012.16548797538,
 'rmse': 8.12514819592119}
```

### Overall metrics (RNN)
```
{'mae': 0.2991196599571649,
 'mse': 0.09432005640591878,
 'r2': -1.8762123492637839,
 'rmse': 0.307115705241394}
```

# LSTM & ARIMA Comparision: Feature: rural population
Comparision of LSTM and ARIMA model of predicting feature rural population. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_rural_population.png)


## RNN evaluation - rural population


![RNN evaluation - rural population](./plots/lstm_evaluation_rural_population.png)

### Overall metrics (ARIMA)
```
{'mae': 0.321854977446329,
 'mse': 0.13815560462336443,
 'r2': -1.8436944953131027,
 'rmse': 0.37169289019749147}
```

### Overall metrics (RNN)
```
{'mae': 0.6492548319498699,
 'mse': 0.4723236900166146,
 'r2': -8.721967349553845,
 'rmse': 0.6872580956355586}
```

# LSTM & ARIMA Comparision: Feature: rural population growth
Comparision of LSTM and ARIMA model of predicting feature rural population growth. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_rural_population_growth.png)


## RNN evaluation - rural population growth


![RNN evaluation - rural population growth](./plots/lstm_evaluation_rural_population_growth.png)

### Overall metrics (ARIMA)
```
{'mae': 0.7605267367985196,
 'mse': 1.4009997528344513,
 'r2': -0.9913269439332499,
 'rmse': 1.1836383539047943}
```

### Overall metrics (RNN)
```
{'mae': 0.90483099682326,
 'mse': 1.5378434227276483,
 'r2': -1.1858312515990561,
 'rmse': 1.2400981504411852}
```

# LSTM & ARIMA Comparision: Feature: age dependency ratio
Comparision of LSTM and ARIMA model of predicting feature age dependency ratio. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_age_dependency_ratio.png)


## RNN evaluation - age dependency ratio


![RNN evaluation - age dependency ratio](./plots/lstm_evaluation_age_dependency_ratio.png)

### Overall metrics (ARIMA)
```
{'mae': 12.422996581505052,
 'mse': 164.72343508529227,
 'r2': -38.7142527724771,
 'rmse': 12.834462789119467}
```

### Overall metrics (RNN)
```
{'mae': 11.67550981630597,
 'mse': 143.22266976473142,
 'r2': -33.53049231786827,
 'rmse': 11.967567412165742}
```

# LSTM & ARIMA Comparision: Feature: urban population
Comparision of LSTM and ARIMA model of predicting feature urban population. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_urban_population.png)


## RNN evaluation - urban population


![RNN evaluation - urban population](./plots/lstm_evaluation_urban_population.png)

### Overall metrics (ARIMA)
```
{'mae': 0.32187385509057265,
 'mse': 0.1381706003531911,
 'r2': -1.8440031565104915,
 'rmse': 0.3717130618544243}
```

### Overall metrics (RNN)
```
{'mae': 0.6569198303222663,
 'mse': 0.48256778939599826,
 'r2': -8.932824441410284,
 'rmse': 0.6946709936336757}
```

# LSTM & ARIMA Comparision: Feature: population growth
Comparision of LSTM and ARIMA model of predicting feature population growth. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_population_growth.png)


## RNN evaluation - population growth


![RNN evaluation - population growth](./plots/lstm_evaluation_population_growth.png)

### Overall metrics (ARIMA)
```
{'mae': 0.7147233755893437,
 'mse': 0.7939668740776032,
 'r2': -0.2903106914232829,
 'rmse': 0.8910481884149719}
```

### Overall metrics (RNN)
```
{'mae': 0.3968428013104792,
 'mse': 0.7223772093884944,
 'r2': -0.1739671602764956,
 'rmse': 0.8499277671593595}
```

# LSTM & ARIMA Comparision: Feature: adolescent fertility rate
Comparision of LSTM and ARIMA model of predicting feature adolescent fertility rate. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_adolescent_fertility_rate.png)


## RNN evaluation - adolescent fertility rate


![RNN evaluation - adolescent fertility rate](./plots/lstm_evaluation_adolescent_fertility_rate.png)

### Overall metrics (ARIMA)
```
{'mae': 1.6744787139958175,
 'mse': 4.304312821474244,
 'r2': -5.8143450030833,
 'rmse': 2.0746837883094966}
```

### Overall metrics (RNN)
```
{'mae': 1.313146427154541,
 'mse': 2.344925887943868,
 'r2': -2.7123542525512945,
 'rmse': 1.5313150844760421}
```

# LSTM & ARIMA Comparision: Feature: life expectancy at birth, total
Comparision of LSTM and ARIMA model of predicting feature life expectancy at birth, total. State: Czechia

## Arima evaluation


![Arima evaluation](./plots/arima_evaluation_life_expectancy_at_birth,_total.png)


## RNN evaluation - life expectancy at birth, total


![RNN evaluation - life expectancy at birth, total](./plots/lstm_evaluation_life_expectancy_at_birth,_total.png)

### Overall metrics (ARIMA)
```
{'mae': 3.3929559679476236,
 'mse': 12.230013890840752,
 'r2': -23.59233614313194,
 'rmse': 3.4971436760363095}
```

### Overall metrics (RNN)
```
{'mae': 0.7325761872578459,
 'mse': 1.3591219720869645,
 'r2': -1.7329473780982996,
 'rmse': 1.1658138668273612}
```

