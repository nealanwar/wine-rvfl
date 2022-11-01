import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import tensorflow as tf

# HD ops
def hd_distance(A, B):
    return np.sum(np.abs(A - B))

def hd_bind(X, HDV):
    binding = np.zeros(X.shape)
    N = X.shape[0]
    for sample in range(N):
        enc = X[sample,:,:]
        binding[sample,:,:] = np.logical_xor(enc, HDV)
    return binding

def hd_bundle(vecs):
    K = vecs[0].shape[0]
    L = vecs[0].shape[1]

    # consensus sum
    sum = np.zeros((K, L))
    for vec in vecs: sum += vec
    return binarize(sum / len(vecs))

# making class protoypes, centroids
def convert_to_centroids(X, Y, centroids):
    N = X.shape[0]
    target = np.zeros(X.shape)
    for sample in range(N):
        target[sample,:,:] = centroids[ground_truth(sample, Y)]
    return target

def ground_truth(idx, Y):
    return int(np.where(Y[idx,:] == 1)[0][0])

def cluster(X, Y, classes):
    # get samples by class
    class_samples = dict.fromkeys(range(classes))
    for cls in range(classes): class_samples[cls] = []

    N = X.shape[0]
    for sample in range(N):
        enc = X[sample,:,:]
        class_samples[ground_truth(sample, Y)].append(enc)

    # create centroids
    centroids = dict.fromkeys(range(classes))
    for cls in range(classes): centroids[cls] = hd_bundle(class_samples[cls])

    return centroids

# DB encoding
def DB_decode(X, L):
    N = X.shape[0]
    K = X.shape[1]
    dec = np.zeros((N, K))
    for sample in range(N): dec[sample,:] = np.sum(X[sample,:], axis=1)
    return dec / L

def DB_encode(X, L):
    N = X.shape[0]
    K = X.shape[1]
    enc = np.zeros((N, K, L))
    for sample in range(N):
        for feature in range(K):
            val = X[sample,feature]
            quant_feature = round(val * L)
            DB_zeros = np.zeros((L))
            DB_ones = np.ones([quant_feature])
            DB_zeros[:len(DB_ones)] = DB_ones 
            enc[sample,feature,:] = DB_zeros
    return enc

# image handling
IMAGES_FOLDER = '/home/neal/multilayerHD-1/multilayerhd_HDC/images/'

def clean_images():
    files = glob.glob('./images/*')
    for f in files:
        os.remove(f)

def save_MNIST(image, name):
    plt.imshow(reshape_square(image, 28), interpolation='nearest')
    plt.savefig(IMAGES_FOLDER + name + '.png')

def print_MNIST_grid_9(images: list, titles: list, name, L):
    if images[0].shape[len(images[0].shape) - 1] == L: # if DB_encoded, DB_decode
        images = [DB_decode(np.expand_dims(img, axis=0), L) for img in images]
    # assert len(images) == 9, 'Must take 9 images'
    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    i = 0
    for img, title, ax in zip(images, titles, axs):
        ax.set_title(title)
        if len(img.shape) == 1 or img.shape[0] != img.shape[1]: img = reshape_square(img, 28)
        imgplot = ax.imshow(img, interpolation='nearest')
        i += 1
    # plt.colorbar(imgplot)
    plt.show()
    plt.savefig(IMAGES_FOLDER + name + '.png')

def print_MNIST(image, name, L=10):
    # if image is DB encoding, decode it
    if len(image.shape) > 1 and image.shape[1] == L:
        image = DB_decode(np.expand_dims(image, axis=0), L)
    save_MNIST(image, name)

def reshape_square(X, K):
    return np.reshape(X, (K, K))

# data cleaning
def clip_and_subset_classes(X, Y, N, classes):
    idx = np.argwhere(Y < classes)
    X = X[idx]
    Y = Y[idx]
    clip = np.random.choice(X.shape[0], N, replace=False)
    return X[clip,:], Y[clip]

def onehot(Y, classes):
    return tf.keras.utils.to_categorical(Y, num_classes=classes)

def binarize(X):
    X[X < 0.5] = 0
    X[X != 0] = 1
    return X

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

def normalize_and_reshape(X, N, classes):
    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    X = normalize(np.reshape(X, (N, classes)))
    return X