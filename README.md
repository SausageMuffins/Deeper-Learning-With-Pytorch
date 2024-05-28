# Deeper-Learning-With-Pytorch
This project implements various neural networks in past research using Pytorch. The goal of this project is to practice using Pytorch as well as gain a deeper understanding of the myriad of neural networks.

Hopefully this repository would be helpful as a reference to current and future students or just anybody that is curious on how the various neural networks can be implemented using Pytorch.

UPDATES:
- 2024 May 28: I have udpated the requirements.txt and added my implementation for the multi-layer-perceptron (MLP) neural network.
- 2024 May 23: I have updated the README.md for easy installation and the expected file structure in the future. The first model to be implemented will be MLP in the coming week.

---

## File Structure
The expected file structure of this repository is as follows:
```
Deeper-Learning-With-Pytorch/
├── data/
│   ├── MNIST
│   └── …
├── mlp/                # Directory for the neural network (Multilayer Perceptrons)
│   ├── readme.md       # Important details about the implemented neural network (directory's name)
│   ├── model.py
│   ├── train.py
│   └── …
├── README.md
└── requirements.txt
```

---

## Installation
To run this project, you need to install the necessary Python packages listed in the `requirements.txt` file. Follow the steps below to set up your environment:

Clone this repository to your local machine using the following command:

```sh
git clone https://github.com/yourusername/Deeper-Learning-With-Pytorch.git
cd Deeper-Learning-With-Pytorch
```

Create a virtual environment:
(Below is an example using [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) but you can always use your favorite virtual environment 😙)

```sh
conda create --name dlwp python=3.10
```

Switch to the newly created environment:
```sh
conda activate dlwp
```

Install the necessary dependencies/libraries:
```sh
pip install -r requirements.txt
```

You can always cross check the requirements.txt and your currently installed libraries with
```sh
pip list
```
