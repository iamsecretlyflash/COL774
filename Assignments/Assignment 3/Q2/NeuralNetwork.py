import numpy as np 

class NeuralNetwork:
    def __init__(self, layers_config, activation_func = 'sigmoid', leakyRelu = None):
        '''
        Initializes the neural network with given parameters
        
        Attributes : layers_config : List containing the dimensions of each layer
                   activation_func : Activation function to be used in the neural network
                   leakyRelu : Slope of leakyReLU activation function (if used)
                   '''
        #layers_config exclude input layer dimensions
        self.layers_config = layers_config
        self.W = {}
        self.b = {}
        self.activation_function = activation_func
        self.leakyRelu_slope = leakyRelu
        self.stochastic_losses = []

    def pre_trained_model(self, model_dict):
        '''
        Loads a pre-trained model from model dictionary
        Model dictionary contains the following attributes:
        W : Dictionary containing the weights of the neural network
        b : Dictionary containing the biases of the neural network
        layers_config : List containing the dimensions of each layer
        activation_function : Activation function used in the neural network
        leakyRelu_slope : Slope of leakyReLU activation function (if used)

        '''
        self.W = model_dict["W"]
        self.b = model_dict["b"]
        self.layers_config = model_dict["layers_config"]
        self.activation_function = model_dict["activation_function"]
        self.leakyRelu_slope = model_dict["leakyRelu_slope"]
        self.best_model = model_dict
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def ReLU(self,x):
        return np.maximum(x, 0)

    def ReLU_derivative(self,x):
        return np.where(x>0, 1, 0)

    def LeakyReLU(self, x):
        return np.maximum(x, x*self.leakyRelu_slope)

    def LeakyReLU_derivative(self, x):
        return np.where(x>0, 1, self.leakyRelu_slope)
 
    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum(axis=0, keepdims=True)
    
    def softmax_derivative(self,x):
        return self.softmax(x)*(1-self.softmax(x))

    def cross_entropy_loss(self,d1,d2):
        return -1*np.sum( d1 * np.log(d2 + 1e-9))

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        if self.activation_function == 'ReLU':
            return self.ReLU(x)
        if self.activation_function == 'LeakyReLU':
            return self.LeakyReLU(x)

    def derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        if self.activation_function == 'ReLU':
            return self.ReLU_derivative(x)
        if self.activation_function == 'LeakyReLU':
            return self.LeakyReLU_derivative(x)
 
    def initialize_parameters(self):
        '''
        Initializes the weights and biases for the neural network using random uniform sampling
        '''
 
        for l in range(1, len(self.layers_config)):
            self.W["W" + str(l)] = np.random.randn(self.layers_config[l], self.layers_config[l - 1]) / np.sqrt(self.layers_config[l-1])
            self.b["b" + str(l)] = np.zeros((self.layers_config[l], 1))
 
    def forward_propagation(self, X):
        '''
        Performs forward propagation and returns the final output and cache
        
        Attributes : X : Input of shape (num_examples, n_features)

        Returns : output : Final output of shape (num_examples, num_output_classes)
        '''

        cache = {}
        cache["act_out_0"] = X
        num_layers = len(self.W)
        for l in range(1,num_layers + 1):

            cache['net_' + str(l)] = np.dot(self.W["W" + str(l)], cache['act_out_' + str(l-1)]) + self.b["b" + str(l)] # W.O_(l-1) + b
            if l != num_layers :
                cache['act_out_' + str(l)] = self.activation(cache['net_' + str(l)])
            else :
                cache["act_out_" + str(l)] = self.softmax(cache['net_' + str(l)])
        return cache["act_out_" + str(num_layers)], cache
 
    def backward_propagation(self, X, Y, cache):
        '''
        Performs backpropagation and returns the derivatives of weights and biases

        Attributes : X : Input of shape (num_examples, n_features)
                     Y : Labels of shape (num_examples, num_output_classes)
                 cache : Dictionary containing the activation unit outputs and final outputs from each layer

        Returns : derivatives : Dictionary containing the derivatives of weights and biases
        '''
        
        # O stores the activation unit outputs/ final outputs from a layer
        num_layers = len(self.W)
        num_examples = X.shape[1]
        derivatives = {}
    
        act_out = cache["act_out_" + str(num_layers)] # Softmax output for last layer
        dnet = (act_out - Y)  # derivative for cross-entropy loss   

        dW = dnet.dot(cache["act_out_" + str(num_layers-1)].T) / num_examples
        db = np.sum(dnet, axis=1, keepdims=True) / num_examples
        d_act_out_next = self.W["W" + str(num_layers)].T.dot(dnet)
 
        derivatives["dW" + str(num_layers)] = dW
        derivatives["db" + str(num_layers)] = db
 
        for l in range(num_layers-1, 0, -1):
            dnet = d_act_out_next * self.derivative(cache["net_" + str(l)])
            dW =  dnet.dot(cache["act_out_" + str(l - 1)].T) / num_examples
            db =  np.sum(dnet, axis=1, keepdims=True) / num_examples
            if l > 1:
                d_act_out_next = self.W["W" + str(l)].T.dot(dnet)
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
            
        return derivatives

    def data_batch(self, trainX, trainY, batch_size = 32):
        '''
        Splits the data into batches for stochastic gradient descent

        Attributes : trainX : Training data of shape (num_examples, n_features)
                     trainY : Training labels of shape (num_example, num_output_classes)
                 batch_size : Batch size for stochastic gradient descent

        Returns : batchX : List of batches of training data of shape (batch_size, n_features) except last batch.
                           Last batch is of shape (num_examples % batch_size, n_features)
                  batchY : List of batches of training labels of shape (batch_size, num_output_classes) except last batch.
        '''
        temp = np.concatenate([trainX, trainY], axis = 1)
        np.random.shuffle(temp)
        trainX, trainY = temp[:,:trainX.shape[1]] , temp[:,trainX.shape[1]:]
        num_batches = trainX.shape[0] // batch_size
        batchX = list(trainX[:num_batches * batch_size].reshape(num_batches, batch_size, trainX.shape[1]))
        batchX.extend([trainX[num_batches * batch_size:]])

        batchY = list(trainY[:num_batches * batch_size].reshape(num_batches, batch_size, trainY.shape[1]))
        batchY.extend([trainY[num_batches * batch_size:]])

        return batchX, batchY

    def fit(self,trainX, trainY, learning_rate, EPOCHS_MAX = 1000, stop_thresh = 1e-6, init_param = True, batch_size = 32, print_every = 10):
        '''
        Trains the neural network on given training data

        Attributes : trainX : Training data of shape (num_examples, n_features)
                     trainY : Training labels of shape (num_examples, num_output_classes)
                     learning_rate : Learning rate for gradient descent (lambda function. use constant for constant learning rate otherwise provide function)
                        EPOCHS_MAX : Maximum number of epochs to train
                        stop_thresh : Threshold for stopping training if error change is less than threshold
                        init_param : Whether to initialize parameters or not ( if not then please use pre_trained_model() function to load pre-trained model)
                        batch_size : Batch size for stochastic gradient descent
                        print_every : Print loss after every print_every epochs

        Returns : None
                     '''

        if init_param : self.initialize_parameters(); self.stochastic_losses = []
        self.epoch_losses = []
        TOTAL_EXAMPLES = trainX.shape[0]
        
        consec_const = 0
        prev_const = 0
        for epoch in range(EPOCHS_MAX):
            net_epoch_loss = 0
            trainX_shuffle , trainY_shuffle = self.data_batch(trainX, trainY,batch_size)
            best_loss = float('inf')
            self.best_model = None
            for batchX, batchY in zip(trainX_shuffle, trainY_shuffle):
                outputL, cache = self.forward_propagation(batchX.T)
                local_loss = self.cross_entropy_loss(batchY.T,  outputL) 
                net_epoch_loss += local_loss
                #print(batchY.shape, batchX.shape)
                derivatives = self.backward_propagation(batchX.T, batchY.T, cache)
                self.deri = derivatives
                #print(derivatives.keys())
                for l in range(1,len(self.W) + 1):
                    self.W["W" + str(l)] -= learning_rate(epoch+1) * derivatives["dW" + str(l)]
                    self.b["b" + str(l)] -= learning_rate(epoch+1) * derivatives["db" + str(l)]
                self.stochastic_losses.append(local_loss / batchX.shape[0])
            self.epoch_losses.append(net_epoch_loss/TOTAL_EXAMPLES)

            if self.epoch_losses[-1] < best_loss:
                best_loss = self.epoch_losses[-1]
                self.best_model = {"W" : self.W , "b" : self.b , "loss" : best_loss, "epoch" : epoch}

            if epoch % print_every == 0 :
                print(f'Loss for epoch : {epoch} == {self.epoch_losses[-1]}')
            
            if epoch >= 1 :
                # check convergence
                if abs(self.epoch_losses[-1] - self.epoch_losses[-2]) < stop_thresh:
                    if prev_const : consec_const += 1
                    else : consec_const = 1; prev_const = 1
                else:
                    prev_const = 0
                if consec_const == 5: 
                    print("Error change less than threshold. Terminating")
                    break
                if epoch == EPOCHS_MAX - 1:
                    print(" MAXIMUM EPOCHS REACHED WITHOUT DESIRED CONVERGENCE. TERMINATING TRAINING!!")

        self.W = self.best_model["W"]
        self.b = self.best_model["b"]

    def predict(self, X):
        '''
        Predicts the output for given input X
        Args : X : Input of shape (num_examples, n_features)
        Returns : pred : Predicted output of shape (num_examples, 1)
        '''
        pred, _ = self.forward_propagation(X.T)
        return pred.argmax(axis = 0)
        
 