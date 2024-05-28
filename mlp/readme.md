# Multi-Layer Perceptron (MLP)

This directory contains the implementation of a Multi-Layer Perceptron (MLP) neural network using PyTorch. The MLP is a foundational model in neural networks and serves as a good starting point for understanding more complex architectures.

In this implementation of the MLP, I generated synthetic data with a defined polynomial (to simulate a "pattern" in the data). Subsequently, the model is trained on this synthetic data.

## Files

- `main.py`: Runs the code from defining the model, to training it and showing the mlp predictions.
- `model.py`: Defines the MLP model architecture.
- `train.py`: Contains the training loop and functions to train the MLP model.
- `readme.md`: This README file.


## Overview

The MLP is a class of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Each node, except for the input nodes, is a neuron that uses a nonlinear activation function.

The MLP is one of the "entry-level" introduction to deep learning where one is introduced to the idea of a neural network.

### Model Architecture

A very simple MLP model is implemented in this directory which consists of the following layers:

1. **Input Layer**: Receives the input data.
2. **Hidden Layers**: One or more hidden layers with activation functions.
3. **Output Layer**: Produces the final output.

**See model.py for more details**

---

## How to Use

---

#### Running main.py

All of the code to train is already implemented in main.py which can be executed using the following command in the terminal:

```sh
python main.py
```

**Note**:
- Do make sure you have your own virtual environment set up and the dependencies are installed - see [here](https://github.com/SausageMuffins/Deeper-Learning-With-Pytorch).
- Do make sure that you are in the correct directory. If you type ls in the command, you should see main.py as one of the items.

---

#### Training

If you are interested in playing around the training of the model, feel free to clone this repository and play around with the optimizer (learning rate), losses or whatever you see fit.

The relevant files are in train.py where I defined the following functions:
- *generate_data*: to generate the synthetic data based on a pattern to train the model on.
- *select_device*: to select the appropriate hardware to train the model
- *train_model*: the training loop

**Note**:
- The parameters like "epochs" are defined in the main.py which goes through the entire dataset.
- The trainloader (which turns each epoch into batches) is also defined in the main.py

---

#### Model

If you are interested in playing around with the architecture of the model, the relevant file is model.py. Again, feel free to experiment around with different number of hidden layers etc.





