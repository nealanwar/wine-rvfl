import tensorflow as tf
import numpy as np
import time
from HDRVFL import HDRVFL

def normalize(x): 
    return (x - x.min()) / (x.max() - x.min())

def test_wine(nn):
    data = [i.strip().split() for i in open("./wine.dat").readlines()]
    N = len(data) - 1
    K = 13
    X = np.zeros([N,K])
    Y = np.array([])
    for i, row in enumerate(data[1:]):
        X[i,:] = np.asarray(row[1:K+1], dtype=float)
        Y = np.append(Y, int(row[14]))
        
    wine_x = X
    wine_y = Y
    p = np.random.permutation(len(wine_x))
    wine_x = wine_x[p]
    wine_y = wine_y[p]
    N_train = 143
    train_wine_x = wine_x[:N_train]
    train_wine_y = wine_y[:N_train]
    test_wine_x = wine_x[N_train:]
    test_wine_y = wine_y[N_train:]
    for i in range(K):
        train_wine_x[:,i] = normalize(train_wine_x[:,i])
        test_wine_x[:,i] = normalize(test_wine_x[:,i])
    train_wine_y = tf.keras.utils.to_categorical(train_wine_y)
    test_wine_y = tf.keras.utils.to_categorical(test_wine_y)

    nn.train(train_wine_x, train_wine_y)
    acc = nn.test(test_wine_x, test_wine_y)
    print('Wine Accuracy:', acc)

def test_MNIST(nn):
    # get train and test samples/classes
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    N_train = train_x.shape[0]
    N_test = test_x.shape[0]

    # for less memory usage, restrict MNIST to 5 classes, 2000 samples (784 features)
    K = 784
    M = 5 # max 10

    N_train = 500
    N_test = 300

    train_idx = np.argwhere(train_y < M)
    test_idx = np.argwhere(test_y < M)
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    test_x = test_x[test_idx]
    test_y = test_y[test_idx]

    train_x = train_x[:N_train, :]
    train_y = train_y[:N_train]
    test_x = test_x[:N_test, :]
    test_y = test_y[:N_test]

    # normalize data
    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    train_x = normalize(np.reshape(train_x, (N_train, K)))
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=M)
    test_x = normalize(np.reshape(test_x, (N_test, K)))
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=M)

    nn.train(train_x, train_y)
    acc = nn.test(test_x, test_y)
    print('MNIST accuracy:', acc)

def main():
    nn = HDRVFL()
    nn.add_hidden_layers(6)

    test_wine(nn)
    test_MNIST(nn)

# time network
start_time = time.time()
main()
print("Seconds Elapsed: %s" % (time.time() - start_time))