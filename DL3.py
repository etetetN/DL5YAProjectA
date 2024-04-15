from sqlite3 import adapt
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import h5py

class DLLayer():

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float)
        self.db = np.zeros((self._num_units,1), dtype=float)
        self.Nlm1 = np.sum(self._input_shape[0]) #Put it in a variable so you don't need to calculate it again, it represents n(l-1)
        if W_initialization == "zeros" :
            self.W = np.zeros((self._num_units, *(self._input_shape)), dtype=float)
        elif W_initialization == "random" :
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        elif W_initialization == "He" :
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(1.0 / self.Nlm1)
        elif W_initialization == "Xaviar" :
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(2.0 / self.Nlm1)
        else: #Initialization isn't anything we expect initially
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

        self.dW = np.zeros((self._num_units, *(self._input_shape)), dtype=float)

    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2, optimization=None, random_scale=0.01):
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self.activation_forward = self._relu
        self.activation_backward = self._relu_backward
        self.activation_trim = 1e-10
        if self._activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward
        elif self._activation == "leaky_relu":
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward
        elif self._activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward
        elif self._activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._sigmoid_backward
        elif self._activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._tanh_backward
        elif self._activation == "softmax":
            self.activation_forward = self._softmax
            self.activation_backward = self._softmax_backward
        elif self._activation == "trim_softmax":
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backward
           
        
        self.alpha = learning_rate
        self._optimization = optimization
        if optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)

            self._adaptive_alpha_W = np.full((self._num_units, *(self._input_shape)), self.alpha)

        self.random_scale = random_scale #0.01 by default

        self.leaky_relu_d = 0.1
        self.adaptive_cont = 0.0
        self.adaptive_switch = 0.0

        self.init_weights(W_initialization)

    #def forward_propagation():

    #def backward_propagation():

    #def update_parameters():

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backward(self,dA):

        A = self._sigmoid(self._Z)

        dZ = dA * A * (1-A)

        return dZ

    
    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_backward(self,dA):
        dZ = np.array(dA, copy=True)

        dZ = np.where(self._Z <= 0, 0, dA)

        return dZ


    def _leaky_relu(self, Z):
        return np.where(Z <= 0, self.leaky_relu_d * Z, Z)

    def _leaky_relu_backward(self, dA):

        dZ = np.where(self._Z < 0, self.leaky_relu_d * dA, dA)

        return dZ


    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        dZ = A**2
        dZ = 1 - dZ
        return dA * dZ

    def _trim_sigmoid(self,Z):

        with np.errstate(over='raise', divide='raise'):

            try:

                A = 1/(1+np.exp(-Z))

            except FloatingPointError:

                Z = np.where(Z < -100, -100,Z)

                A = A = 1/(1+np.exp(-Z))

        TRIM = self.activation_trim

        if TRIM > 0:

            A = np.where(A < TRIM,TRIM,A)

            A = np.where(A > 1-TRIM,1-TRIM, A)

        return A

    def _trim_tanh(self,Z):

        A = np.tanh(Z)

        TRIM = self.activation_trim

        if TRIM > 0:

            A = np.where(A < -1+TRIM,TRIM,A)

            A = np.where(A > 1-TRIM,1-TRIM, A)

        return A

    def _softmax(self, Z):
        exp_by_z = np.exp(Z)
        A = exp_by_z/np.sum(exp_by_z, axis=0)
        return A

    def _softmax_backward(self, dA):
        return dA

    def _trim_softmax(self, Z):

        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100,Z)
                eZ = np.exp(Z)
        A = eZ/np.sum(eZ, axis=0)

        return A

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, self._A_prev) + self.b
        return self.activation_forward(self._Z)


    def backward_propagation(self, dA):
        m = self._A_prev.shape[1]
        dZ = self.activation_backward(dA)

        self.dW = np.dot(dZ, (self._A_prev).T) / m
        self.db = (np.sum(dZ , axis=1, keepdims=True) ) / m

        dA_Prev = np.dot((self.W).T, dZ)

        return dA_Prev

    def update_parameters(self):
        #match self._optimization: #Found this when searching for switch in python
        if self._optimization == None:
            self.W -= self.dW * self.alpha
            self.b -= self.db * self.alpha
        elif self._optimization == 'adaptive':
            self._adaptive_alpha_W = np.where(self.dW * self._adaptive_alpha_W > 0, 
                                                  self._adaptive_alpha_W * self.adaptive_cont, 
                                                  self._adaptive_alpha_W * -self.adaptive_switch)

            self._adaptive_alpha_b = np.where(self.db * self._adaptive_alpha_b > 0, 
                                                  self._adaptive_alpha_b * self.adaptive_cont, 
                                                  self._adaptive_alpha_b * -self.adaptive_switch)

            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b

    def __str__(self):

        s = self.name + " Layer:\n"

        s += "\tnum_units: " + str(self._num_units) + "\n"

        s += "\tactivation: " + self._activation + "\n"

        s += "\tinput_shape: " + str(self._input_shape) + "\n"

        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"

        if self._activation == "leaky_relu":

            s += "\t\tleaky relu parameters:\n"

            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"

        #optimization

        if self._optimization == "adaptive":

            s += "\t\tadaptive parameters:\n"

            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"

            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"

        # parameters

        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"

        s += "\t\tshape weights: " + str(self.W.shape)+"\n"

        plt.hist(self.W.reshape(-1))

        plt.title("W histogram")

        plt.show()

        return s;




class DLModel():

    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def save_weights(self, path):
        for i in range(1, len(self.layers)): #Check for every layer

            file_name = self.layers[i].name
            self.layers[i].save_weights(path, file_name)

    def compile(self, loss, threshold=0.5):
        self.loss = loss
        if loss == "squared_means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        if loss == "cross_entropy":
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward
        if loss == "categorical_cross_entropy":
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward

        self.threshold = threshold
        self._is_compiled = True

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        losses = self.loss_forward(AL, Y)

        return (1/m) * np.sum(losses)

    def train(self, X, Y, num_iterations):

        print_ind = max(num_iterations // 100, 1)

        L = len(self.layers)

        costs = []

        for i in range(num_iterations):

            # forward propagation

            Al = X

            for l in range(1,L):

                Al = self.layers[l].forward_propagation(Al,False)

            #backward propagation

            dAl = self.loss_backward(Al, Y)

            for l in reversed(range(1,L)):

                dAl = self.layers[l].backward_propagation(dAl)

                # update parameters

                self.layers[l].update_parameters()

            #record progress

            if i % print_ind == 0:

                J = self.compute_cost(Al, Y)

                costs.append(J)

                print("cost after ",str(i//print_ind),"%:",str(J))

        return costs

    def predict(self, X):
        A_Prev = X

        for i in range(1, len(self.layers)):
            A_Prev = self.layers[i].forward_propagation(A_Prev, True)

        if self.loss != "categorical_cross_entropy": 
            result = A_Prev > self.threshold
        else:
            result = A_Prev == A_Prev.max(axis=0)
        return result


    def _squared_means(self, AL, Y):
        return (AL - Y)**2

    def _squared_means_backward(self, AL, Y):
        return 2 * (AL - Y)


    def _cross_entropy(self, AL, Y):
        return np.where(Y == 0, -np.log(1 - AL), -np.log(AL))

    def _cross_entropy_backward(self, AL, Y):
        return np.where(Y == 0, 1/(1-AL), -1/AL)

    def _categorical_cross_entropy(self, AL, Y):
        return np.sum(np.where(Y == 1, -np.log(AL), 0))

    def _categorical_cross_entropy_backward(self, AL, Y):
        return (AL - Y)

    #לא עובד כי אין לי מושג איזה ספרייה זה בשביל הפונקציה confusion matrix ולכן זה ייתן שגיאה
    #def confusion_matrix(self, X, Y):
    #    prediction = self.predict(X)
    #    prediction_index = np.argmax(prediction, axis=0)
    #    Y_index = np.argmax(Y, axis=0)
    #    right = np.sum(prediction_index == Y_index)
    #    print("accuracy: ",str(right/len(Y[0])))
    #    cf = confusion_matrix(prediction_index, Y_index)
    #    print(cf)
    #    return cf

    def __str__(self):

        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"

        if self._is_compiled:

            s += "\tCompilation parameters:\n"

            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"

            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):

            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"

        return s