import numpy as np
import utils
import tqdm

# activations
class Linear:
    def forward(self, x):
        return x

    def backward(self, y):
        return y

class Sigmoid:
    def forward(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, y):
        y = utils.normalize(y)
        return np.log(y/(1-y))

class Relu:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, y):
        y[y < 0] = -np.random.rand(*y[y < 0].shape)
        return y

# activation layer
class Activation:
    def __init__(self, type):
        if type == 'linear':
            self.type = Linear() 
        elif type == 'sigmoid':
            self.type = Sigmoid()
        elif type == 'relu':
            self.type = Relu()
    
    def forward(self, x):
        return self.type.forward(x)

    def backward(self, y):
        return self.type.backward(y)

# layers
class Pseudoinverse():
    def __init__(self, K, L):
        self.L = L
        self.HD = np.random.randint(2, size=(K, L))

    def apply(self, X, W):
        return X @ W
        A = utils.DB_encode(utils.normalize(X), self.L) # [N x K] -> [N x K x L]
        H = utils.hd_bind(A, self.HD)
        B = utils.DB_decode(H, self.L) # [N x K x L] -> [N x K]
        return B @ W

    def train(self, X, target):
        return np.linalg.pinv(X) @ target # min ||XW - target||

    def feedback(self, target, W):
        B = target @ np.linalg.pinv(W) # min ||BW - Y||
        return B
        A = utils.DB_encode(B, self.L) # [N x K] -> [N x K x L]
        H = utils.hd_bind(A, self.HD)
        return utils.DB_decode(H, self.L) # [N x K x L] -> [N x K]

class XOR():
    def __init__(self, K, L):
        self.L = L
    
    def apply(self, X, W):
        return utils.hd_bind(X, W)

    def train(self, X, target):
        N = X.shape[0]

        # train HD needed to optimize
        HDV = []
        for sample in range(N):
            HDV.append(np.logical_xor(X[sample,:,:], target[sample,:,:]))

        return utils.hd_bundle(HDV)

    def feedback(self, X, W): # xor is an involution
        return self.apply(X, W)

# neural layer
class Neural:
    def __init__(self, K, L, type):
        if type == 'pseudoinv':
            self.type = Pseudoinverse(K, L) 
            self.weights = np.random.randint(2, size=(K, K))
        elif type == 'xor':
            self.type = XOR(K, L)
            self.weights = np.random.randint(2, size=(K, L))

    def apply(self, X, W):
        return self.type.apply(X, W)

    def train(self, X, target):
        return self.type.train(X, target)

    def feedback(self, X, W):
        return self.type.feedback(X, W)

class MLHD():
    def __init__(self, L: int, N_layers: int, activation: str, optimization: str, pre_train_centroids: bool):
        self.L = L # neurons in neural layer
        self.N_layers = N_layers # each layer is neural + activation
        self.layers = [] # layers array
        self.activation = activation
        self.optimization = optimization
        self.debug = False
        self.pre_train_centroids = pre_train_centroids
    
    def train(self, X, Y):
        # store X and Y
        stored_X = X.copy()
        stored_Y = Y.copy()

        # get dimensions
        self.N_train = X.shape[0]
        self.K = X.shape[1]
        self.M = Y.shape[1]

        # DB encode X
        DB = utils.DB_encode(X, self.L)
        # if self.debug: utils.print_MNIST(X[0], 'encoded', self.L)

        # initialize layers
        self.layers = []
        for _ in range(self.N_layers):
            self.layers.append(Neural(self.K, self.L, self.optimization))
            self.layers.append(Activation(self.activation))
        self.N_layers = len(self.layers)

        if self.debug: 
            layer_titles = ['Initial Random Weights: ' + str(i) for i, _ in enumerate(self.layers)]
            utils.print_MNIST_grid_9([layer.weights if isinstance(layer, Neural) else np.zeros((self.K, self.L if self.optimization == 'xor' else self.K)) for layer in self.layers], layer_titles, 'layer_weights', self.L)

        if self.optimization == 'xor': # let X = DB off the bat if we are using pure xor training
            X = DB
            # generate initial centroids
            if self.pre_train_centroids:
                self.centroids = utils.cluster(DB, Y, self.M)
            else:
                self.centroids = dict()
                for C in range(self.M): self.centroids[C] = np.random.randint(2, size=(self.K, self.L))
            if self.debug:
                c_titles = ['Initial Centroid ' + str(i) for i, _ in enumerate(self.centroids)]
                utils.print_MNIST_grid_9([C for _, C in self.centroids.items()], c_titles, 'initial_centroids', self.L)

        # generate feedback matrices
        feedback = []
        if self.optimization == 'xor': 
            target = utils.convert_to_centroids(DB, Y, self.centroids) # final targets are centroids
        elif self.optimization == 'pseudoinv': 
            target = Y # final target is one-hot encoding
            self.last_layer = Neural(self.K, self.L, self.activation)
            self.last_layer.weights = np.random.randint(2, size=(self.K, self.M))
            target = Y @ np.linalg.pinv(self.last_layer.weights) 
            feedback.append(target) # start the feedback off with minimization of ||W_N * X - Y||

        feedback.append(target)
        for i in range(self.N_layers - 1, 0, -1):
            layer = self.layers[i]
            print(i, target.shape)
            if isinstance(layer, Neural): 
                target = layer.feedback(target, layer.weights)
                feedback.append(target)
            elif isinstance(layer, Activation): 
                target = layer.backward(target)
                feedback.append(target)
        feedback.reverse() # order feedbacks in order of layers

        if self.debug: 
            fb_titles = ['Feedback From Layer: ' + str(i+1) for i, _ in enumerate(feedback)]
            utils.print_MNIST_grid_9([np.squeeze(fb[0]) for fb in feedback], fb_titles, 'feedback_tensors', self.L)

        results = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Neural): 
                print('processing neural', i, 'of', self.N_layers, 'X shape:', X.shape)
                target = feedback[i]
                layer.weights = layer.train(X, target) # train weights that turn X into target
                X = layer.apply(X, layer.weights) # now apply this
            elif isinstance(layer, Activation): 
                print('processing activation', i, 'of', self.N_layers, 'X shape:', X.shape)
                X = layer.forward(X)
            results.append(X)
        
        if self.debug: 
            x_titles = ['Output From Layer: ' + str(i) for i, _ in enumerate(feedback)]
            utils.print_MNIST_grid_9([result[0] for result in results], x_titles, 'layer_results', self.L)

        if self.optimization == 'xor':
            # generate final centroids
            self.centroids = utils.cluster(X, Y, self.M)
            if self.debug:
                c_titles = ['Final Centroids ' + str(i) for i, _ in enumerate(self.centroids)]
                utils.print_MNIST_grid_9([C for _, C in self.centroids.items()], c_titles, 'final_centroids', self.L)
        elif self.optimization == 'pseudoinv':
            # train final weights matrix
            self.last_layer.weights = np.linalg.pinv(X) @ Y

        print('Post-train eval:', self.test(stored_X, stored_Y))

    def test(self, X, Y):
        # get dimensions
        self.N_test = X.shape[0]

        # DB encode test X
        DB = utils.DB_encode(X, self.L)
        if self.optimization == 'xor': # let X = DB off the bat if we are using pure xor training
            X = DB
        if self.debug: utils.print_MNIST(X[0], 'test_encoded', self.L)

        # apply weights
        for layer in self.layers:
            if isinstance(layer, Neural): 
                X = layer.apply(X, layer.weights) # apply weight
            elif isinstance(layer, Activation): 
                X = layer.forward(X) # activate
        if self.debug: utils.print_MNIST(X[0], 'test_encoded_final', self.L)

        Y_hat = np.zeros((self.N_test, self.M))

        if self.optimization == 'xor':
            if self.debug: # print first 10 samples
                for i in range(10):
                    sample_0 = X[i,:,:]
                    dist_from_0 = list()
                    for cls, C in self.centroids.items():
                        dist_from_0.append(sample_0 - C)
                    dist_titles = ['Centroid Dist: ' + str(np.sum(np.abs(img))) for img in dist_from_0]
                    utils.print_MNIST_grid_9(dist_from_0, dist_titles, 'all_centroid_dists' + str(i), self.L)

            # now find nearest centroid
            for sample in range(self.N_test):
                dists = np.zeros((self.M))
                for cls, C in self.centroids.items(): dists[cls] = utils.hd_distance(X[sample,:,:], C)
                Y_hat[sample][np.argmin(dists)] = 1
        if self.optimization == 'pseudoinv':
            output = X @ self.last_layer.weights
            for sample in range(self.N_test):
                Y_hat[sample][np.argmax(output[sample,:])] = 1

        # accuracy
        missed = sum([int(np.any(row)) for row in Y_hat - Y])
        return float(self.N_test - missed) / self.N_test