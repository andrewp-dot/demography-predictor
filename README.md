# Welcome to the Demography-Predictor project!

This project is aimed to create an universal model for predicting future development of the states demography. Available parameters for predictions are aging groups (
    `population ages 0-14`, `population ages 15-64`, `population ages 65 and above`) and gender distribution (`population male`, `population female`). The full predictor is made by chained 2 models:

1. Model of feature future development prediction 
2. Model of target variables prediction 

You can create your own model by editing scripts in the `src` folder, or just use predefined CLI.

## Installation

### Requirements

- Python 3.11
- [Poetry](https://python-poetry.org/)

### Setup

Clone the repository and install dependencies:

If the python and poetry are installed, just run `poetry install` in the project root directory. 



## Available CLI

To succesful run CLI you need `.env` file (see `.env.example` file). Path and other configuration are listed in the `config.py` file.

### Running API

To select models modify `config.py` script. Then to run API simply run:

```
poetry run api
```

To try API you can use pre-implemented client. For more info run:

```
poetry run client --help
```

### Training the model

`poetry run cli train [age-predictor|gender-dist-predictor] [OPTIONS]`. Options for both available predictors are listed below.

```
Usage: cli train age-predictor [OPTIONS]

Options:
  --name TEXT                Name of the predictor. It is also a model key.
                             [required]
  --type [LSTM|ARIMA]        Type of the model. You can choose from options:
                             LSTM, ARIMA
  --wealth-groups TEXT       List of comma separated wealth groups. as a comma
                             separated strings string, e.g. --wealth-groups
                             "high_income, lower_middle_income"
  --geolocation-groups TEXT  List of comma separated geolocation groups. as a
                             comma separated strings string, e.g. --states
                             "europe, north america"
  --states TEXT              List of included in training states as a comma
                             separated strings string, e.g. --states "Czechia,
                             United States"
  --modify-for-target-model  If specified, modifed data will be used to train
                             the target prediction model. By deafult, data are
                             adjusted for feature target model
  --help                     Show this message and exit.
```

**Example:**

Training predictor with all data used for both models:

```
poetry run cli train age-predictor --name lstm-age-predictor --type lstm
```

### Runing predictor experiments
Predictor experiments include overall ealuation, by group evaluation and convergence experiment (if and when the model start to converge).

```
poetry run cli predictor-experiments --name <name_of_the_trained_pipline>
```


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
