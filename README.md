# Simple Neural Network Library

## Introduction
This project is a simple neural network library implemented in Python using NumPy for numerical operations and Matplotlib for visualizations. The library includes essential components for building, training, and evaluating neural networks.

## Installation
Ensure you have Python 3.6 or higher and install the requirements:
```
pip install -r requirements.txt
```

## Usage

### Normalizing Data
To normalize data, use:
```
data = np.array([1, 2, 3, 4, 5])
normalized_data = normalize_data(data)
```

### Measuring Accuracy
To measure accuracy:
```
predicted = np.array([1, 0, 1, 1])
target = np.array([1, 0, 0, 1])
accuracy = measure_accuracy(predicted, target)
```

### Activation Functions and Gradients
For activation functions and gradients, you can use ReLU activation:
```
output = ReLU(input_array)
```
And the ReLU gradient:
```
gradient = ReLU_grad(input_array)
```

### Loss Functions and Gradients
For loss functions and gradients, use Mean Squared Error (MSE):
```
loss = MSE(predicted, target)
```
And the MSE gradient:
```
gradient = MSE_grad(predicted, target)
```

To use Softmax Cross-Entropy:
```
loss = softmax_crossentropy_with_logits(logits, labels)
```
And its gradient:
```
gradient = grad_softmax_crossentropy_with_logits(logits, labels)
```

## Classes

### DenseLayer
The `DenseLayer` class represents a dense (fully connected) layer in the neural network:
```
layer = DenseLayer(input_size=10, output_size=5, activation='relu')
forward_output = layer.forward(input_array)
grad_input = layer.backward(input_array, grad_output, iteration)
```

### NeuralNet
The `NeuralNet` class represents the neural network model, managing layers and training. You can choose between `linear` and `relu` activation functions as well as `mse` and `softmax_crossentropy_with_logits` loss functions:
```
nn = NeuralNet([
        DenseLayer(X_train.shape[1], 100, "relu"),
        DenseLayer(1000, 10, "relu"),
        DenseLayer(1000, 100, "relu"),
        DenseLayer(1000, 10)
    ], loss_func="softmax_crossentropy_with_logits")

```

## Training and Evaluation
Use the `train` method of `NeuralNet` to train your model. Verbose mode provides loss and accuracy plots:
```
nn.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=100, learning_rate=0.01, verbose=True, optimizer="adam")
predictions = nn.predict(X_test)
```

Currently supported optimizers are: `sgd` adn `adam`. \
Visualizations of training loss and validation accuracy are saved as "loss.png" and "accuracy.png".



## Conclusion
This framework provides a basic yet powerful tool for understanding and experimenting with neural networks. The exaple provided in `neural_net_test.py` above $98\%$ Test Accuracy on MNIST Datset with about a minute of training on a CPU.

---




