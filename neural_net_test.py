import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from neural_net import NeuralNet, DenseLayer, measure_accuracy, normalize_data



if __name__ == '__main__':
    np.random.seed(0)

    # load data
    X, y = loadlocal_mnist(images_path='data/train-images-idx3-ubyte', labels_path='data/train-labels-idx1-ubyte')

    # normalize data
    X = normalize_data(X)

    # split to train and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

    # one hot encoding train data
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded = y_train.reshape(len(y_train), 1)
    y_train = one_hot_encoder.fit_transform(encoded)

    # load test data
    X_test, y_test = loadlocal_mnist(images_path='data/t10k-images-idx3-ubyte', labels_path='data/t10k-labels-idx1-ubyte')

    # normalize test data
    X_test = normalize_data(X_test)
    
    model = NeuralNet([
        DenseLayer(X_train.shape[1], 1000, "relu"),
        DenseLayer(1000, 10)
    ], loss_func="softmax_crossentropy_with_logits")

    model.train(
        X=X_train,
        y=y_train,
        X_val=X_validation,
        y_val=y_validation,
        batch_size=256,
        epochs=15,
        learning_rate=0.001,
        verbose=True,
        optimizer="adam"
    )

    print("Test accuracy:", measure_accuracy(model.predict(X_test), y_test))