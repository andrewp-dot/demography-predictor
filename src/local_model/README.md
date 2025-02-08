# Implemenation of local model 

Local model predicts the independent/input parameters (X values) for the global model. Input is the sequence of the development time from range `X(t-n), X(t-n+1), ... , X(t)` and output is the vector of `X(t+1)`.

## Used links:
1. [LSTM net implementation tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)