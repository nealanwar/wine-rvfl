import tensorflow as tf
import time
import utils
from multilayerHD import MLHD
import numpy as np

def test_MNIST(N_train = 60000, N_test = 10000, classes = 10, layers = 2, L = 100, activation = 'linear', optimization = 'xor', binarize = False, pre_train_centroids = False):
    # get train and test samples/classes
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    K = 784

    # clip data and take subset of classes
    train_x, train_y = utils.clip_and_subset_classes(train_x, train_y, N_train, classes)
    test_x, test_y = utils.clip_and_subset_classes(test_x, test_y, N_test, classes)

    # convert classes to one-hot
    train_y = utils.onehot(train_y, classes)
    test_y = utils.onehot(test_y, classes)

    # normalize data
    train_x = utils.normalize_and_reshape(train_x, N_train, K)
    test_x = utils.normalize_and_reshape(test_x, N_test, K)

    # normalize, binarize data
    if binarize:
        train_x = utils.binarize(train_x)
        test_x = utils.binarize(test_x)

    # test images are binarized properly
    utils.print_MNIST(train_x[1,:], 'input')

    nn = MLHD(L if not binarize else 1, layers, activation, optimization, pre_train_centroids)
    nn.train(train_x, train_y)
    return nn.test(test_x, test_y)

def main():
    utils.clean_images()
    neuron_step = [40, 60, 80, 100, 120]
    layer_step = [1, 2, 3, 4, 5, 6]
    binary_step = [True, False]
    activation_step = ['linear', 'relu']
    optimization_step = ['pseudoinv']
    TRIALS = 1
    for opt in optimization_step:
        for act in activation_step:
            for layers in layer_step:
                trials = np.array([])
                for _ in range(TRIALS):
                    acc = test_MNIST(N_train=10000, 
                                N_test=1000, 
                                classes=5, 
                                layers=layers,
                                L=100,
                                activation=act,
                                optimization=opt,
                                binarize=False,
                                pre_train_centroids=False)
                    trials = np.append(trials, acc)
                    print('Mean %s-%s %d-layer Accuracy:' % (opt, act, layers), np.mean(trials))

# time network
start_time = time.time()
main()
print("Seconds Elapsed: %s" % (time.time() - start_time))