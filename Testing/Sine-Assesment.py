import numpy as np
import matplotlib.pyplot as plt
import copy


class NN():
    def __init__(self, shape):
        self.shape = shape
        self.size = len(self.shape)
        self.lr = 0.01
        self.active = self.ReLU
        self.end = self.sigmoid
        self.optimizer = self.sgd
        self.params = {}
        self.costs = []
        #np.random.seed(99)

        for i in range(1, self.size + 1):
            self.params[f"L{i}"] = self.shape[i - 1]

        for i in range(1, self.size):
            self.params[f"l{i}_m"] = 0
            self.params[f"l{i}_v"] = 0
        self.t = 0

        self.initialize()
        self.initialize(True)

    def initialize(self, save=False):
        if save == False:
            for i in range(1, self.size):
                self.params[f"W{i}"] = np.random.randn(self.params[f"L{i}"],
                                                       self.params[f"L{i+1}"])
                self.params[f"b{i}"] = np.random.randn(1,
                                                       self.params[f"L{i+1}"])
        else:
            #self.archive = self.params.copy()
            self.archive = copy.deepcopy(self.params)

    def reset(self):
        #self.params = self.archive.copy()
        self.params = copy.deepcopy(self.archive)
        self.costs = []

    def sigmoid(self, x, derivative=False):
        if derivative == False:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return x * (1.0 - x)

    def ReLU(self, inputs, derivative=False):
        if derivative == False:
            return np.maximum(0, inputs)
        elif derivative == True:
            inputs[inputs <= 0] = 0
            inputs[inputs > 0] = 1
            return inputs

    def tanh(self, x, derivative=False):
        if derivative == False:
            t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return t
        dt = 1 - x**2
        return dt

    def ELU(self, inputs, derivative=False):
        if derivative == False:
            return np.where(inputs > 0, inputs, np.exp(inputs) - 1)
        elif derivative == True:
            return np.where(inputs > 0, 1, np.exp(inputs))

    def sgd(self):
        for i in reversed(range(1, self.size)):
            self.params[f"W{i}"] += (np.dot(self.params[f"a{i-1}"].T,
                                            self.params[f"d{i}"]))
            self.params[f"b{i}"] += (np.sum(self.params[f"d{i}"], axis=0))

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

    def forward(self, x):
        self.params["a0"] = x
        for i in range(1, self.size - 1):
            self.params[f"z{i}"] = np.dot(
                self.params[f"a{i-1}"],
                self.params[f"W{i}"]) + self.params[f"b{i}"]
            self.params[f"a{i}"] = self.active(self.params[f"z{i}"])

        self.params[f"z{self.size-1}"] = np.dot(
            self.params[f"a{self.size-2}"],
            self.params[f"W{self.size-1}"]) + self.params[f"b{self.size-1}"]
        self.params[f"a{self.size-1}"] = self.end(
            self.params[f"z{self.size-1}"])

        return self.params[f"a{self.size-1}"]

    def backward(self, x, y):
        cost = np.sum((y - self.params[f"a{self.size-1}"])**2) / len(y)
        self.costs.append(cost)
        self.params[f"d{self.size-1}"] = (
            y - self.params[f"a{self.size-1}"]) * self.end(
                self.params[f"a{self.size-1}"], True)

        for i in reversed(range(1, self.size - 1)):
            self.params[f"d{i}"] = self.params[f"d{i+1}"].dot(
                self.params[f"W{i+1}"].T) * self.active(
                    self.params[f"a{i}"], True)

        self.optimizer()

x = np.random.uniform(0,2*np.pi,[100,1])
y = np.sin(x)

nn = NN([1,2,10,5,1])
nn.active = nn.tanh
nn.end = nn.tanh
nn.optimizer = nn.adam
for i in range(1500):
    nn.forward(x)
    nn.backward(x, y)
plt.plot(nn.costs, label="ELU")
x = np.random.uniform(0,2*np.pi,[10,1])
y = np.sin(x)
print(nn.forward(x))
print(y)
print(np.sum((y - nn.params[f"a{nn.size-1}"])**2) / len(y))
plt.legend(loc="upper left")
plt.show()
