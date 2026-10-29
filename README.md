## Neural Network

This is a simple neural network wrapped with OOP, in this project it's used to predict gains of different stocks.

#### Structure
The neural network is 4 layers, and can have any number of neurons in the first, second and the third layer, but can only have 1 neuron at layer4 (output layer). 

Use `train.py` to train the nn with a single stock, `train_multiple.py` to train on multiple different stocks. In `quant.py` is a simple quantitative finance strategy that
uses the neural network to earn profits. See more at [results]()

Use `Model.save()` or `Model.load()` to save or load trained weights and biases data. The trained data is stored at `train_data.txt`.
