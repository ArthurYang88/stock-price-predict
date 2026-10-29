## Neural Network

This is a simple neural network wrapped with OOP, in this project it's used to predict gains of different stocks.

#### Structure
The neural network is 4 layers, and can have any number of neurons in the first, second and the third layer, but can only have 1 neuron at layer4 (output layer). 

ReLu function was used as the activation function in the first two layers, no activation function was used at the output layer.

Use `train.py` to train the nn with a single stock, `train_multiple.py` to train on multiple different stocks. 

In `quant.py` is a simple quantitative finance strategy that uses the neural network to earn profits. See more at [30 days results](Last%30%days.pdf), [60 days results](Last%60%days.pdf), [Quant results](Quant%20results.pdf) The profit result was too high, so there must be some problem in there.

Use `Model.save()` or `Model.load()` to save or load trained weights and biases data. The trained data is stored at `train_data.txt`.

