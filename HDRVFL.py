import numpy as np
from matplotlib import pyplot as plt
import math

class DBEncoding():
    def __init__(self):
        self.unique = dict()

    def __eq__(self, other):
        if isinstance(other, DBEncoding):
            for k, v in self.unique.items():
                other_k = other.unique.get(k)
                if other_k is not None:
                    if not np.array_equal(v, other_k):
                       return False
            return True
        return False

    def to_sparse_matrix(self, K, L, common):
        if len(self.unique) == 0:
            return None
            
        mat = np.zeros((K, L))
        for row in range(mat.shape[0]):
            if self.unique.get(row) is not None:
                mat[row,:] = self.unique[row]
            else:
                mat[row,:] = common[row,:]
        return mat


    def to_matrix(self):
        if len(self.unique) == 0:
            return None
            
        mat = np.zeros(len(self.unique[next(iter(self.unique))]))
        for _, row in self.unique.items():
            mat = np.vstack((mat, row))
        return mat[1:]

class DBTensor():
    def __init__(self):
        self.common = None
        self.encs = dict()

    def imprint(self):
        imprint = 0
        for _, enc in self.encs.items():
            for _, row in enc.unique.items():
                imprint += np.sum(row)
        return imprint

class NeuralLayer():
    def __init__(self):
        # weights
        self.weights = None

class ActivationLayer():
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def antisigmoid(self, x):
        low = np.min(x)
        high = np.max(x)
        frac = x / (1 - x)
        return np.log(frac)

    def relu(self, x):
        x[x < 0] = 0
        return x

    def antirelu(self, y):
        y[y < 0] = -1*np.random.random()
        return y

    def activate_DB_tensor(self, T: DBTensor, forward):
        FT = DBTensor()

        normalize = lambda x: (x - x.min()) / (x.max() - x.min())

        # apply function to common
        dequantized_common = np.squeeze(np.sum(T.common, axis=1))
        if forward:
            transformed_common = self.relu(dequantized_common)
        else:
            transformed_common = np.nan_to_num(self.antirelu(normalize(dequantized_common)), nan=0, posinf=0, neginf=0)
        K = T.common.shape[0]
        L = T.common.shape[1]

        transformed_common = normalize(transformed_common)

        FT.common = np.zeros((K, L))
        for i in range(K):
            feature = transformed_common[i]
            requantized_feature = round(feature * L)
            DB_encoding = np.zeros((L))
            DB_encoded_feature = np.ones([requantized_feature])
            DB_encoding[:len(DB_encoded_feature)] = DB_encoded_feature 
            FT.common[i,:] = DB_encoding
        
        # apply the function to uniques
        for idx, enc in T.encs.items():
            new_enc = DBEncoding()
            old_enc = enc.to_sparse_matrix(K, L, FT.common)
            dequantized_old_enc = np.squeeze(np.sum(old_enc, axis=1))

            if forward:
                transformed_enc = self.relu(dequantized_old_enc)
            else:
                transformed_enc = np.nan_to_num(self.antirelu(normalize(dequantized_old_enc)), nan=0, posinf=0, neginf=0)
            K = T.common.shape[0]
            L = T.common.shape[1]
            normalize = lambda x: (x - x.min()) / (x.max() - x.min())
            transformed_enc = normalize(transformed_enc)

            for enc_idx, _ in enc.unique.items():
                feature = transformed_enc[enc_idx]
                requantized_feature = round(feature * L)
                DB_encoding = np.zeros((L))
                DB_encoded_feature = np.ones([requantized_feature])
                DB_encoding[:len(DB_encoded_feature)] = DB_encoded_feature 
                new_enc.unique[enc_idx] = DB_encoding
            
            FT.encs[idx] = new_enc
        return FT

    def activate(self, output: DBTensor):
        return self.activate_DB_tensor(output, True)

    def deactivate(self, output: DBTensor):
        return self.activate_DB_tensor(output, False)

class HDRVFL():
    def __init__(self):
        # encoding length
        self.L = 100
        # number of layers
        self.N_layers = 0
        # layer weights
        self.layers = dict() # np.array([], dtype=int)
        # proportion of class centroid feedback tensor that is common (i.e. similarity % of each centroid to all others)
        self.centroid_feedback_common_ratio = 0.7
    
    def quantize_and_DB_encode_row(self, B):
        row_encoding = DBEncoding()

        for i in range(self.K):
            feature = B[i]
            # empty feature
            if feature == 0:
                #row_encoding.empty_idx.add(i)
                continue
            # full feature
            quantized_feature = round(feature * self.L)
            DB_encoding = np.zeros((self.L))
            DB_encoded_feature = np.ones([quantized_feature])
            DB_encoding[:len(DB_encoded_feature)] = DB_encoded_feature 
            row_encoding.unique[i] = DB_encoding
        return row_encoding

    def quantize_and_DB_encode(self, A):
        encoding = DBTensor()
        encoding.common = np.zeros([self.K, self.L])
        for samples in range(A.shape[0]):
            encoded_sample = self.quantize_and_DB_encode_row(A[samples,:])
            encoding.encs[samples] = encoded_sample
        return encoding

    def unquantize_and_reshape(self, A, N):
        image = np.zeros((A.shape[0]))
        for i in range(A.shape[0]):
            image[i] = np.sum(A[i,:])
        
        return np.reshape(image, (N,N))

    # custom binding function for two matrices (weights and common -> common output needed)
    def HD_bind_matrices(self, W, C):
        binding = np.logical_xor(W, C).astype(int)
        return binding

    # custom binding function for a matrix and a DB encoding (weights and sample unique -> unique output needed)
    def HD_bind_matrix_DB_encoding(self, W, DB: DBEncoding):
        binding = DBEncoding()
        # bind every unique row in DB with the corresponding row of W
        for idx, enc in DB.unique.items():
            binding.unique[idx] = np.logical_xor(W[idx,:], enc).astype(int)
        return binding

    # custom binding function for two DB encodings (feedback DB and input DB -> HD DB needed)
    def HD_bind_DB_encodings(self, FB: DBEncoding, IN: DBEncoding):
        HD = DBEncoding()
        for idx in FB.unique.keys():
            if IN.unique.get(idx) is not None:
                HD.unique[idx] = np.logical_xor(FB.unique[idx], IN.unique[idx]).astype(int)
            else:
                HD.unique[idx] = FB.unique[idx]
        return HD

    # binding function for a matrix and a DB tensor
    def HD_bind_matrix_DB_tensor(self, A, B: DBTensor, N):
        binding = DBTensor()
        binding.common = self.HD_bind_matrices(A, B.common)
        for sample in range(N):
            binding.encs[sample] = self.HD_bind_matrix_DB_encoding(A, B.encs[sample])
        return binding

    # binding function for two DB Tensors
    def HD_bind_DB_tensors(self, A: DBTensor, B: DBTensor, N):
        binding = DBTensor()
        binding.common = self.HD_bind_matrices(A.common, B.common)
        for sample in range(N):
            binding.encs[sample] = self.HD_bind_DB_encodings(A.encs[sample], B.encs[sample])
        return binding

    def matrices_to_DB_tensor(self, M):
        T = DBTensor()
        T.common = np.zeros((self.K, self.L))
        for i, mat in M.items():
            DB = DBEncoding()
            for row in range(mat.shape[0]):
                DB.unique[row] = mat[row,:]
            T.encs[i] = DB
        return T

    # custom bundling function to condense a DBTensor into a single matrix
    def HD_bundle(self, T: DBTensor):
        N = len(T.encs)
        if N == 0:
            return None

        bundle = np.zeros((self.K, self.L))

        # consensus sum encs, if a particular row is not in an enc the enc's contribution
        # to the consensus sum is the corresponding common row
        for sample in range(N):
            sample_enc = T.encs[sample]
            for feature_idx in range(self.K):
                if feature_idx in sample_enc.unique.keys():
                    bundle[feature_idx,:] += sample_enc.unique[feature_idx]
                else:
                    bundle[feature_idx,:] += T.common[feature_idx,:]

        bundle /= N
        bundle[bundle >= 0.5] = -1
        bundle[bundle != -1] = 0
        bundle[bundle == -1] = 1
        return bundle

    def HD_distance_matrices(self, A, B):
        return np.sum(np.abs(A - B), axis=1)

    def HD_distance_DB_encodings(self, DB: DBEncoding, C: DBEncoding, K, common_dist):
        dist = 0
        # if the feature is not present in this encoding, the contribution comes from the 
        # corresponding row of the common encoding
        for feature_idx in range(K):
            if feature_idx in DB.unique.keys():
                dist += np.sum(np.abs(DB.unique[feature_idx] - C.unique[feature_idx]))
            else:
                dist += common_dist[feature_idx]
        return dist

    def get_random_DB(self, L):
        zeros_vec = np.zeros((L))
        ones_vec = np.ones((np.random.randint(L)))
        zeros_vec[:len(ones_vec)] = ones_vec
        return zeros_vec

    def add_hidden_layer(self):
        self.layers[self.N_layers] = NeuralLayer()
        self.N_layers += 1

    def add_activation(self):
        self.layers[self.N_layers] = ActivationLayer()
        self.N_layers += 1

    def train(self, X, Y):
        self.N_train = X.shape[0]
        self.K = X.shape[1]
        self.M = Y.shape[1]
      
        # quantize and DB-encode data
        DB = self.quantize_and_DB_encode(X)

        # create random weights for hidden layers
        for layer in range(self.N_layers):
            if isinstance(self.layers[layer], NeuralLayer):
                self.layers[layer].weights = np.random.randint(2, size=(self.K, self.L))

        # get samples by class
        samples_by_class = dict.fromkeys(range(self.M))
        for cls in range(self.M):
            samples_by_class[cls] = DBTensor()
            samples_by_class[cls].common = DB.common

        for sample in range(self.N_train):
            ground_truth = int(np.where(Y[sample,:] == 1)[0][0])
            cls_samples_idx = len(samples_by_class[ground_truth].encs)
            samples_by_class[ground_truth].encs[cls_samples_idx] = DB.encs[sample]

        # create initial
        class_prototypes = dict()
        for cls in range(self.M):
            prototype = self.HD_bundle(samples_by_class[cls])
            if prototype is not None:
                class_prototypes[cls] = prototype
            
        # create centroid matrix as weight matrix for final layer
        self.centroids = self.matrices_to_DB_tensor(class_prototypes)
        

        self.centroids = DBTensor()
        # divide common and unique feature proportions according to centroid commonality hyperparameter
        self.centroids.common = np.zeros((self.K, self.L))
        for feature in range(self.K):
            self.centroids.common[feature,:] = self.get_random_DB(self.L)
        N_random = round((1 - self.centroid_feedback_common_ratio) * self.K)
        for cls in range(self.M):
            cls_centroid = DBEncoding()
            centroid_unique_idx = np.random.choice(self.K, N_random, replace=False)
            for i in centroid_unique_idx:
                cls_centroid.unique[i] = self.get_random_DB(self.L) # np.random.randint(2, size=(self.L))
            self.centroids.encs[cls] = cls_centroid
        
        # print init centroids
        for idx, enc in self.centroids.encs.items():
            sparse = enc.to_sparse_matrix(self.K, self.L, self.centroids.common)
            sparse = self.unquantize_and_reshape(sparse, 28)

        # generate ZORB feedback tensors up front to avoid recalculating them every iteration
        FB = dict()
        # reverse final centroid classifier process, store as last output needed
        output_needed = DBTensor()
        output_needed.common = self.centroids.common
        for sample in range(self.N_train):
            ground_truth = int(np.where(Y[sample,:] == 1)[0][0])
            enc = DBEncoding()
            enc.unique = self.centroids.encs[ground_truth].unique
            output_needed.encs[sample] = enc
        FB[self.N_layers] = output_needed

        # bind output with weights from each of the preceding layers
        for layer in range(self.N_layers - 1, 0, -1):
            print('Feedback', layer)
            # feedback neural
            if isinstance(self.layers[layer], NeuralLayer):
                weights = self.layers[layer].weights
                output_needed = self.HD_bind_matrix_DB_tensor(weights, output_needed, self.N_train)
                FB[layer] = output_needed # output_needed
                #print('Out imprint:', np.sum(output_needed.imprint()), flush=True)
            # feedback deactivation
            else:
                output_needed = self.layers[layer].deactivate(output_needed)
                FB[layer] = output_needed

        # train one layer at a time, up to penultimate layer
        for layer in range(self.N_layers):
            print('Iteration', layer)
            if isinstance(self.layers[layer], NeuralLayer):
                print('Training neural layer', layer)
                # fetch needed output for layer i, minimizing ||(Centroids ⊕ W_N-1 ⊕ W_N-2 ⊕ ... ⊕ W_i+1 ⊕ X) - Y||
                output_needed = FB[layer+1]

                # train HD vector encodings that produce the needed output row corresponding to each sample
                HDV = DBTensor()
                HDV = self.HD_bind_DB_tensors(output_needed, DB, self.N_train)

                # bundle each individual encoding into one general encoding
                trained_weights = self.HD_bundle(HDV)
                self.layers[layer].weights = trained_weights

                # produce input to the next layer by binding this layer's newly trained weights to X
                DB = self.HD_bind_matrix_DB_tensor(trained_weights, DB, self.N_train)

            elif isinstance(self.layers[layer], ActivationLayer):
                print('Applying activation layer', layer)
                output_needed = self.layers[layer].activate(DB)

        class_prototypes = dict()
        for cls in range(self.M):
            prototype = self.HD_bundle(samples_by_class[cls])
            if prototype is not None:
                class_prototypes[cls] = prototype
                #print('Class', cls , 'Proto size:', prototype.shape,'Prototype imprint:', np.sum(prototype))
            
        self.centroids = self.matrices_to_DB_tensor(class_prototypes)

        # print final centroids
        for idx, enc in self.centroids.encs.items():
            sparse = enc.to_sparse_matrix(self.K, self.L, self.centroids.common)
            sparse = self.unquantize_and_reshape(sparse, 28)
            plt.imshow(sparse, interpolation='nearest')
            plt.savefig('./images2/FINAL_CENTROID_' + str(idx) + '.png')
                
    def test(self, X, Y):
        self.N_test = X.shape[0]

        # quantize and DB-encode data
        DB = self.quantize_and_DB_encode(X)
        
        # run the test data through the hidden layers
        for layer in range(self.N_layers):
            if isinstance(self.layers[layer], NeuralLayer):
                print('Applying layer', layer, 'weights')
                weights = self.layers[layer].weights
                # produce input to the next layer by binding this layer's trained weights to X

                DB = self.HD_bind_matrix_DB_tensor(weights, DB, self.N_test)      
                #print('Layer', layer, 'common imprint:', np.sum(DB.common))
            else:
                print('Applying layer', layer, 'activation')
                DB = self.layers[layer].activate(DB)

        # find distances of common vectors among samples to each centroid
        common_distance = dict()
        for cls in range(self.M):
            centroid = self.centroids.encs[cls].to_matrix()
            common_distance[cls] = self.HD_distance_matrices(centroid, DB.common)

        # now find the nearest centroid for each sample
        Y_hat = np.zeros((self.N_test, self.M))
        for sample in range(self.N_test):
            centroid_dist = np.zeros(self.M)
            for cls in range(self.M):
                centroid_dist[cls] = self.HD_distance_DB_encodings(DB.encs[sample], self.centroids.encs[cls], self.K, common_distance[cls])
            Y_hat[sample][np.argmin(centroid_dist)] = 1

        diff = Y_hat - Y
        missed = 0
        for row in diff:
            if np.any(row):
                missed += 1
        acc = float(self.N_test - missed)/self.N_test
        return acc
