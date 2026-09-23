# Handwritten Digit Recognition (from scratch)

A pure Java implementation of a neural network designed to recognize handwritten digits, built entirely from the ground up without the use of external machine learning libraries.

## What I wanted to implement

- **Custom Neural Network**: Full implementation of a multi-layer perceptron (MLP) including forward propagation and backpropagation.
- **Mathematics**: Custom implementations of **Cross-Entropy Loss**, **ReLU**, and **Softmax** activation functions.
- **MNIST Data Management**: I create methods to load, the data from the official MNIST dataset
- **Data Augmentation**: Improves model robustness by applying random scaling and shifting using **bilinear interpolation** during training.
- **Preprocessing**:
  - Bounding box detection to isolate the digit.
  - Normalization and resizing to a 28x28 grid.
  - **Centering by "Mass"**: Aligns the digit's center of mass with the network's expected input center for maximum accuracy.

## Project Structure

```text
.
├── data/
│   ├── mnist/          # MNIST binary dataset files and loader
│   └── my_digits/      # Custom PNG images for testing recognition
├── models/
│   └── mnist-model.bin # After training I save the model in this file so I don't have to traing every time
├── neural/             # Core NN logic (Layers, Neurons, Activation, Loss)
├── recognition/        # Application entry point and image preprocessing
│   └── preprocessing/   # PngLoader, Augmentation, and Vectorizer
└── training/           # Training loop and weight optimization (Trainer)
```

## Technical Details

### Neural Network Architecture
The network follows a simple but effective feed-forward topology:
**784 (Input)** $\rightarrow$ **128 (Hidden, ReLU)** $\rightarrow$ **10 (Output, Softmax)**

- **Input Layer**: 784 neurons representing a flattened 28x28 pixel grayscale image.
- **Hidden Layer**: 128 neurons using the Rectified Linear Unit (ReLU) activation to introduce non-linearity.
- **Output Layer**: 10 neurons (digits 0-9) using Softmax to produce a probability distribution.
- **Optimization**: Stochastic Gradient Descent (SGD) with Cross-Entropy Loss.

### Preprocessing Pipeline
To ensure that custom-drawn digits are recognized as accurately as the MNIST dataset, the project implements a specific pipeline:
`PNG Image` $\rightarrow$ `Bounding Box Detection` $\rightarrow$ `Normalization` $\rightarrow$ `Center by Mass` $\rightarrow$ `Vectorization` $\rightarrow$ `Input Vector [784]`

## Getting Started

### Prerequisites
- **Java Runtime Environment (JRE)** 8 or higher.
- **Java Development Kit (JDK)** for compilation.

### Compilation
Compile the project from the root directory:
```bash
javac -d out neural/*.java data/mnist/*.java training/*.java recognition/preprocessing/*.java recognition/Main.java
```

### Running the Project
The project supports two primary modes:

#### 1. Training Mode
Loads the MNIST dataset, trains the network for several epochs, and saves the model with the highest test accuracy.
```bash
java -cp out recognition.Main 0
```

#### 2. Recognition Mode
Loads the trained model and predicts digits for all `.png` images found in the `data/my_digits` folder.
```bash
java -cp out recognition.Main 1
```

## Usage Guide

- **MNIST Data**: Place the MNIST binary files (`train-images.idx3-ubyte`, etc.) in the `data/mnist/` directory.
- **Custom Recognition**:
  1. Place your handwritten digit images (PNG format) in `data/my_digits/`.
  2. Run the project in recognition mode.
  3. Check the terminal for the predicted digit and the confidence percentage.
- **Model Storage**: The best-performing weights are automatically saved to `models/mnist-model.bin`.
