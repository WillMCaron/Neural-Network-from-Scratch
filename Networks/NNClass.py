# Look for new one
import numpy as np
from enum import IntEnum


class ActivType(IntEnum):
    ELU = 0
    RELU = 1  # broken
    TANH = 2
    SIGMOID = 3


class NN():
    def __init__(self,
                 shape,
                 active=ActivType.RELU,
                 end=ActivType.SIGMOID,
                 lr=.01):
        # sets the size (number of layers) for the network
        self.size = len(shape)
        # learning rate
        self.lr = lr
        # batching index
        self.Xindex = 0
        self.Yindex = 0
        # contains all parameters of the network
        self.params = {}
        # amount for testing
        self.testSize = 1000
        # saves the neuron amounts for the layers of the network
        for size in range(self.size):
            self.params[f"L{size+1}"] = shape[size]

        # the activation functions
        active_funcs = [self.ELU, self.ReLU, self.tanh, self.sigmoid]

        # sets the activation function
        self.active = active_funcs[active]
        #self.active = self.ReLU
        # sets the end activation function
        self.end = active_funcs[end]
        #self.end = self.ReLU
        # sets the error calculator
        self.error = self.mse
        # sets the network optimizer
        self.optimizer = self.adam

        # sets some of the parameters for adam optimizer
        if self.optimizer == self.adam:
            for i in range(1, self.size):
                self.params[f"l{i}_m"] = 0
                self.params[f"l{i}_v"] = 0
            self.t = 0

    # initializes the weights and biases before saving them for reference
        self.reset()
        self.reset(True)

    # this function initializes the wieghts and biases randomly but can also save them

    def reset(self, save=False):
        # saves the weights and biases in a separate parameters dictionary
        if save == True:
            self.params2 = self.params
        else:
            # initializes the weights and biases randomly
            self.params["W1"] = np.random.randn(self.params["L1"],
                                                self.params["L2"]) * .01
            # The for loop wont work if there are 2 layers
            if self.size > 2:
                for size in range(self.size - 1):
                    self.params[f"W{size+1}"] = np.random.randn(
                        self.params[f"L{size+1}"],
                        self.params[f"L{size+2}"]) * .01
                    self.params[f"b{size+1}"] = np.random.randn(
                        1, self.params[f"L{size+2}"]) * .01
            else:
                self.params["b1"] = np.random.randn(1, self.params["L2"]) * .01

    def tanh(self, x, der=False):
        if der == False:
            t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return t
        dt = 1 - x**2
        return dt

    def sigmoid(self, inputs, der=False):
        if der == False:
            return 1.0 / (1.0 + np.exp(-inputs))
        elif der == True:
            return inputs * (1.0 - inputs)

    def ReLU(self, inputs, der=False):
        if der == False:
            return np.maximum(0, inputs)
        elif der == True:
            inputs[inputs <= 0] = 0
            inputs[inputs > 0] = 1
            return inputs

    def ELU(self, inputs, der=False):
        if der == False:
            return np.where(inputs > 0, inputs, np.exp(inputs) - 1)
        elif der == True:
            return np.where(inputs > 0, 1, np.exp(inputs))

    def softmax(self, x):
        mx = np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(x - mx)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return numerator / denominator

    def mse(self, X, y):
        error = np.sum((self.forward(X) - y)**2) / len(y)
        return error

    # ADAM is a way to change the weights
    def adam(self, decay_rate_1=.9, decay_rate_2=.99, epsilon=10e-8):
        for i in reversed(range(1, self.size)):
            self.params[f"Wd{i}"] = (np.dot(self.params[f"a{i-1}"].T,
                                            self.params[f"d{i}"]))

        self.t += 1  # Increment Time Step

        # Computing 1st and 2nd moment for each layer
        for i in reversed(range(1, self.size)):
            self.params[f"l{i}_m"] = self.params[f"l{i}_m"] * decay_rate_1 + (
                1 - decay_rate_1) * self.params[f"Wd{i}"]
            self.params[f"l{i}_v"] = self.params[f"l{i}_v"] * decay_rate_2 + (
                1 - decay_rate_2) * (self.params[f"Wd{i}"]**2)

        for i in reversed(range(1, self.size)):
            self.params[f"l{i}_m_corrected"] = self.params[f"l{i}_m"] / (
                1 - (decay_rate_1**self.t))
            self.params[f"l{i}_v_corrected"] = self.params[f"l{i}_v"] / (
                1 - (decay_rate_2**self.t))

        for i in reversed(range(1, self.size)):
            self.params[f"w{i}_update"] = self.params[f"l{i}_m_corrected"] / (
                np.sqrt(self.params[f"l{i}_v_corrected"]) + epsilon)

        for i in reversed(range(1, self.size)):
            self.params[f"W{i}"] += self.lr * self.params[f"w{i}_update"]

        for i in reversed(range(1, self.size)):
            self.params[f"b{i}"] = self.params[f"b{i}"] + (
                self.lr * (np.sum(self.params[f"d{i}"], axis=0)))

    # another way to update the weights and biases
    def sgd(self):
        for i in reversed(range(1, self.size)):
            self.params[f"W{i}"] += self.lr * (np.dot(self.params[f"a{i-1}"].T,
                                                      self.params[f"d{i}"]))

        for i in reversed(range(1, self.size)):
            #self.params[f"b{i}"] = self.params[f"b{i}"] - (np.sum(self.params[f"d{i}"], axis = 0))
            self.params[f"b{i}"] = self.params[f"b{i}"] + (
                self.lr * (np.sum(self.params[f"d{i}"], axis=0)))

    # Forward propagation through the network
    def forward(self, inputs):
        # sets the first layer as the inputs
        self.params["a0"] = inputs
        # multiplies the previous layer outputs by the weights and biases of the current layer
        for i in range(self.size - 2):
            # z is the output of W*X+B
            self.params[f"z{i+1}"] = np.dot(
                self.params[f"a{i}"],
                self.params[f"W{i+1}"]) + self.params[f"b{i+1}"]
            # a is the activation of z
            self.params[f"a{i+1}"] = self.active(self.params[f"z{i+1}"], False)

        # This does the last layer with the separate activation function
        self.params[f"z{self.size-1}"] = np.dot(
            self.params[f"a{self.size-2}"],
            self.params[f"W{self.size-1}"]) + self.params[f"b{self.size-1}"]

        self.params[f"a{self.size-1}"] = self.end(
            self.params[f"z{self.size-1}"], False)
        # Returns the last activation, or the end result
        return self.params[f"a{self.size-1}"]

    # Backward propagation through the network
    def backward(self, target):
        #eval(")(tixe"[::-1]) #lol
        # starts out with calculating the error of the output layer
        # the d = (target-output)*the derivative of the output
        self.params[f"d{self.size-1}"] = (
            target - self.params[f"a{self.size-1}"])# * self.end(self.params[f"a{self.size-1}"], True)
        # backpropagates through the rest of the network
        for i in reversed(range(1, self.size - 1)):
            # d = previous d dot matrix multiplies by the weights of the current layer transposed times the derivative of the previous layers deriviative of the activation
            self.params[f"d{i}"] = self.params[f"d{i+1}"].dot(
                self.params[f"W{i+1}"].T) * self.active(
                    self.params[f"a{i}"], True)
        # the optimizer that adjusts the weights
        self.optimizer()

    # batches the data
    def batch(self, X, y, batchSize):
        X_train = []
        y_train = []
        for i in range(batchSize):
            X_train.append(X[self.Xindex])
            y_train.append(y[self.Yindex])
            self.Xindex += 1
            self.Yindex += 1
            if self.Xindex > len(X) - 1:
                self.Xindex = 0
            if self.Yindex > len(y) - 1:
                self.Yindex = 0
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train

        # checks all the seeds for an optimal output of initial weights
    def seed_Check(self, X, y, seedPlus=1, numTest=100, log=True):
        seed = 0
        best = 10000
        self.costs = []
        seeds = []
        if log == True:
            print("Testing for optimal random seed...")
        for i in range(100):
            np.random.seed(i)
            self.reset()
            for loop in range(self.trainSize):
                self.forward(X)
                self.backward(y)
            cost = self.error(X, y)
            if cost < best:
                seed = i
                best = cost
                self.reset(True)
            if log == True:
                print(f"Seed: {i}/99 ")
            self.costs.append(cost)
            seeds.append(i)
        np.random.seed(seed)
        if log == True:
            print(f"Optimal Seed: {seed}")
            print()
        return seed

    # Checks the batch size for an optimal size
    def batch_test(self, X,y, batchStart = 10, batchPlus = 100, numTest = 10, log=True):
        batch = batchStart
        best = 10000
        batch_Best = batchStart
        batches = []
        self.batchCost = []
        if log == True:
            print("Testing batch sizes...")
        for i in range(numTest):
            self.params = self.params2
            self.train(X,y,batch, self.trainSize, False)
            self.batchCost.append(self.error(X,y))
            if (self.error(X,y) < best):
                best = self.error(X,y)
                batch_Best = batch
                if log == True:
                    print("Current Best Cost:",best) 
            batches.append(batch)
            batch+=batchPlus
        if log == True:
            print("Best Cost:",best)
            print("Best Batch Size:",batch_Best)
            print()
        self.params = self.params2
        return batch_Best

    # Checks for an optimal learning rate
    def lr_test(self, X, y, bestie, increment = 10, numTest = 5, log=True):
        lr = 1./(increment**numTest)
        lrs = []
        bestLR = lr
        best = 10000
        self.lrCost = []
        self.lr = lr
        print("Testing learning rates...")
        for i in range(numTest):
            self.params = self.params2
            self.train(X,y,bestie, self.trainSize,False)
            self.lrCost.append(self.error(X,y))
            if self.error(X,y) < best:
                best = self.error(X,y)
                bestLR = lr
                print(bestLR)
                if log == True:
                    print("Current Best Cost:",best)
            lrs.append(lr)
            lr *= increment
            self.lr = lr
        if log == True:
            print("Best Cost:",best)
            print("Best lr:",bestLR)
            print()
            #self.plot_cost(lrs, "lr")
        self.lr = bestLR
        return bestLR

# Trains the neural network

    def train(self, X, y, BatchSize, Iterations, log = True):
        for i in range(Iterations):
            xTrain, yTrain = self.batch(X, y, BatchSize)
            #print(yTrain.size)
            self.forward(xTrain)
            self.backward(yTrain)
            if i%1000 == 0 and log==True:
              print(f"Cost: {self.error(X,y)}")
'''


X = np.random.uniform(0,2*np.pi,[100,1])
#print(X.shape)
y = np.sin(X)

X = np.random.uniform(0,100, [100,1])
y = X


X = np.array([[0,1],[1,0],[0,0],[1,1]])
y = np.array([[1],[1],[0],[0]])

nn = NN([1,10,1], ActivType.TANH, ActivType.TANH, lr = .001)
nn.trainSize = 1000
nn.seed_Check(X,y)
batch = nn.batch_test(X,y)
#lr = nn.lr_test(X,y,batch)
#nn.lr=lr
nn.train(X, y, batch, 50000) 
X = np.random.uniform(0,2*np.pi,[10,1])
y = np.sin(X)
outs = nn.forward(X)
print(y)
print(outs)
print(nn.error(X,y))
'''
