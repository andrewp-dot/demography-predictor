# Implemenation of local model 

Local model predicts the independent/input parameters (X values) for the global model. Input is the sequence of the development time from range `X(t-n), X(t-n+1), ... , X(t)` and output is the vector of `X(t+1)`.

## Architectures
TODO: 
1. pure LSTM

2. pure BPNN

3. BPNN with LSTM
    - BPNN before
    - BPNN after
    - combined: X layers of `Linear-LSTM` layers, out layer: `Linear`

## Batching 
TODO: describe batching in training

## Multiple steps forecasting
Predcting values until the `X(t+r)` timestap.

## Inputing multiple sequences as input (not a single one)
TODO: look if it is possible 

## Used links:
1. [LSTM net implementation tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)