import numpy as np
from matplotlib import pyplot as plt



def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def measure_accuracy(predicted, target):
    return np.sum(predicted == target) / len(target)


def ReLU(arr):
    return np.maximum(arr, 0)


def ReLU_grad(arr):
    return np.where(arr > 0, 1, 0)


def MSE(predicted, target):
    return np.sum(np.square(predicted - target)) / (2 * target.size)


def MSE_grad(predicted, target):
    return (predicted - target) / target.size


def softmax_crossentropy_with_logits(X, y):
    exp_norm = X.max(axis=-1, keepdims=True)

    xentropy = np.sum(y * (-X + exp_norm + np.log(np.sum(np.exp(X - exp_norm), axis=-1, keepdims=True))), axis=-1, keepdims=True)

    return xentropy


def grad_softmax_crossentropy_with_logits(X, y):

    exp_norm = X.max(axis=-1, keepdims=True)
    softmax = np.exp(X - exp_norm) / np.exp(X - exp_norm).sum(axis=-1, keepdims=True)

    return (-y + softmax) / X.shape[0]


class DenseLayer:

    activations = {
        "relu": ReLU,
        "linear": lambda x: x
    }

    activations_grad = {
        "relu": ReLU_grad,
        "linear": lambda x: 1
    }

    def __init__(self, input_size, output_size, activation="linear", learning_rate=0.1,
                 optimizer="sgd", beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.weight_update = {
            "sgd": self.sgd_update,
            "adam": self.adam_update
        }

        self._weights = np.random.randn(input_size + 1, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self._learning_rate = learning_rate
        self._activation_func = activation
        self.optimizer = optimizer
        self.v = 0
        self.s = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def forward(self, layer_input):

        # add ones column to represent bias and be multiplied with additional weight
        layer_input_with_bias = np.hstack((layer_input, np.ones((layer_input.shape[0], 1))))

        return DenseLayer.activations[self._activation_func](np.dot(layer_input_with_bias, self._weights))

    def backward(self, layer_input, grad_output, iteration):

        # add ones column to represent bias
        layer_input_with_bias = np.hstack((layer_input, np.ones((layer_input.shape[0], 1))))

        # gradient of activation function with respect to neuron dot product
        activation_grad = grad_output * DenseLayer.activations_grad[self._activation_func](np.dot(layer_input_with_bias, self._weights))
        
        # gradient of dot product with respect to input
        grad_input = np.dot(activation_grad, self._weights[:-1].T)

        # gradient of dot product with respect to weights
        grad_weights = np.dot(layer_input_with_bias.T, activation_grad)

        # perform gradient descent step using chosen method
        self._weights -= self.weight_update[self.optimizer](grad_weights, iteration)

        return grad_input

    def sgd_update(self, grad_weights, *args):
        return self._learning_rate * grad_weights


    def adam_update(self, grad_weights, iteration, *args):
        # grad_weights from current minibatch
        # first moment
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad_weights

        # second moment
        self.s = self.beta2 * self.s + (1 - self.beta2) * np.square(grad_weights)

        # bias correction
        v_corr = self.v / (1 - self.beta1 ** iteration)
        s_corr = self.s / (1 - self.beta2 ** iteration)

        # compute update weights
        update = self.learning_rate * (v_corr / (np.sqrt(s_corr) + self.epsilon))
        return update


class NeuralNet:

    loss_funcs = {
        "mse": MSE,
        "softmax_crossentropy_with_logits": softmax_crossentropy_with_logits
    }

    loss_funcs_grad = {
        "mse": MSE_grad,
        "softmax_crossentropy_with_logits": grad_softmax_crossentropy_with_logits
    }

    def __init__(self, layers=None, loss_func=None):

        if layers is not None:
            self.layers = layers

        else:
            self.layers = []

        self.loss_func = loss_func

    def add_layer(self, layer):
        self.layers += layer

    def forward(self, X):

        layer_input = X
        layer_inputs = [X]

        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_inputs.append(layer_output)
            layer_input = layer_output

        return layer_inputs

    def predict(self, X):

        output = self.forward(X)[-1]

        return np.argmax(output, axis=-1)

    def _backpropagate(self, X, y, iteration):

        layer_inputs = self.forward(X)

        loss = NeuralNet.loss_funcs[self.loss_func](layer_inputs[-1], y)
        loss_grad = NeuralNet.loss_funcs_grad[self.loss_func](layer_inputs[-1], y)

        for i in range(len(self.layers) - 1, -1, -1):
            loss_grad = self.layers[i].backward(layer_inputs[i], loss_grad, iteration)

        return np.mean(loss)

    def _generate_minibatches(self, X, y, batch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start_ind in range(0, X.shape[0] - batch_size + 1, batch_size):

            batch_ind = indices[start_ind:(start_ind + batch_size)]
            yield X[batch_ind, :], y[batch_ind]

    def set_learning_rate(self, value):

        for layer in self.layers:
            layer.learning_rate = value

    def set_beta1(self, value):

        for layer in self.layers:
            layer.beta1 = value

    def set_beta2(self, value):

        for layer in self.layers:
            layer.beta2 = value

    def set_epsilon(self, value):

        for layer in self.layers:
            layer.epsilon = value

    def set_optimizer(self, optimizer):

        for layer in self.layers:
            layer.optimizer = optimizer

    def set_hyperparameters(self, learning_rate=None, beta1=None, optimizer=None,
                            beta2=None, epsilon=None, loss_func=None):

        if learning_rate is not None:
            self.set_learning_rate(learning_rate)

        if beta1 is not None:
            self.set_beta1(beta1)

        if beta2 is not None:
            self.set_beta2(beta2)

        if epsilon is not None:
            self.set_epsilon(epsilon)

        if optimizer is not None:
            self.set_optimizer(optimizer)

        if loss_func is not None:
            self.loss_func = loss_func

    def train(self, X, y, X_val ,y_val, batch_size, epochs, learning_rate=None, verbose=False,
              loss_func=None, optimizer="sgd", beta1=None, beta2=None, epsilon=None):

        self.set_hyperparameters(learning_rate, loss_func, optimizer,beta1, beta2, epsilon)

        if verbose:
            losses = []
            accuracies = []

        self.set_hyperparameters(learning_rate)
        iteration = 1
        for epoch in range(epochs):
            for batch_X, batch_y in self._generate_minibatches(X, y, batch_size):

                loss = self._backpropagate(batch_X, batch_y, iteration)
                iteration += 1

                if verbose:
                    losses.append(loss)

            if verbose:

                accuracy = measure_accuracy(self.predict(X_val), y_val)
                accuracies.append(accuracy)
                print("Epoch", epoch + 1)
                print("Mean loss:", losses[-1])
                print("Validation accuracy:", accuracies[-1])

        if verbose:
            plt.plot(losses, label='Mean loss')
            plt.legend(loc='best')
            plt.grid()
            plt.show()
            plt.savefig("loss")

            plt.plot(accuracies, label='Validation accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()
            plt.savefig("accuracy")

