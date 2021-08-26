import numpy as np
import matplotlib.pylab as plt

class NN():
    def __init__ (self, shape, lr = .1, activ = 'ELU', seed = None):
        self.size = len(shape)
        self.params = {}
        for size in range(len(shape)):
            self.params["L"+str(size+1)] = shape[size]
        self.lr = lr
        if seed != None:
            np.random.seed = seed
        # Relu sometimes converges at .5, XOR also at 2 layers
        if activ == "ELU":
            self.activ = self.ELU
        elif activ == "ReLU":
            self.activ = self.ReLU
        else:
            self.activ = self.sigmoid

        self.end = self.sigmoid
        # Weights
        self.params["W1"] = np.random.randn(self.params["L1"], self.params["L2"])
        if len(shape) > 2:
            for size in range(len(shape)-1):
                self.params["W" + str(size+1)] = np.random.randn(self.params["L"+str(size+1)], self.params["L"+str(size+2)])
                self.params["b" + str(size+1)] = np.random.randn(1, self.params["L"+str(size+2)])
        else:
            self.params["b1"] = np.random.randn(1,self.params["L2"])

    def reset(self):
        self.params["W1"] = np.random.randn(self.params["L1"], self.params["L2"])
        if self.size > 2:
            for size in range(self.size-1):
                self.params["W" + str(size+1)] = np.random.randn(self.params["L"+str(size+1)], self.params["L"+str(size+2)])
                self.params["b" + str(size+1)] = np.random.randn(1, self.params["L"+str(size+2)])
        else:
            self.params["b1"] = np.random.randn(1,self.params["L2"])
    
    def sigmoid(self, inputs, der):
        if der == False:
            return 1/(1+np.exp(-inputs))
        elif der == True:
            return inputs * (1 - inputs)
    
    def ReLU(self, inputs, der):
        if der == False:
            return np.maximum(0,inputs)
        elif der == True:
            inputs[inputs<=0] = 0
            inputs[inputs>0] = 1
            return inputs

    def ELU(self, inputs, der):
        if der == False:
            return np.where(inputs>0, inputs, np.exp(inputs)-1)
        elif der == True:
            return np.where(inputs>0, 1, np.exp(inputs))
    def softmax(self, inputs, der):
        if der == False:
            #e_x = inputs
            e_x = np.exp(inputs - inputs.max())
            return e_x / e_x.sum(axis = 0)
        else:
            pass
    def forward(self, inputs): #[#!#]
        self.params["a0"] = inputs
        for i in range(self.size-2):
            self.params["z"+str(i+1)] = np.dot(self.params["a" + str(i)], self.params["W"+str(i+1)]) + self.params["b"+str(i+1)] 
            self.params["a"+str(i+1)] = self.activ(self.params["z"+str(i+1)], False)

        self.params["z"+str(self.size-1)] = np.dot(self.params["a"+str(self.size-2)], self.params["W"+str(self.size-1)]) + self.params["b"+str(self.size - 1)]
        self.params["a"+str(self.size-1)] = self.end(self.params["z"+str(self.size-1)], False)
        # Return final calculated value of network
        return self.params["a"+str(self.size-1)]

    def backward(self, target): 
        # This backpropagates our network and adjusts our weights/biases
        self.params["d"+str(self.size-1)] = (target - self.params["a" + str(self.size-1)])*self.end(self.params["a" + str(self.size-1)], True)
        for i in reversed(range(1,self.size-1)):
            self.params["d"+str(i)] = self.params["d"+str(i+1)].dot(self.params["W"+str(i+1)].T)*self.activ(self.params["a"+str(i)], True)
        
        # Update weights
        for i in reversed(range(1,self.size)):
            self.params["W"+str(i)] += (np.dot(self.params["a"+str(i-1)].T, self.params["d"+str(i)]))
        
        # Update biases
        for i in reversed(range(1,self.size)):
            self.params["b"+str(i)] = self.params["b"+str(i)] + (np.sum(self.params["d"+str(i)], axis = 0))
        
    def train(self,X, y, iterations, log = False):
        best = 100
        seed = 0
        self.costs = []
        for i in range(100):
            np.random.seed(i)
            self.reset()
            for loop in range(1000):
                outs = self.forward(X)
                self.backward(y)
            cost = np.sum((self.params["a"+str(self.size-1)]-y)**2)/len(y)
            if cost < best:
                seed = i
            if log == True:
                print(i+1, "/100", sep='')
        np.random.seed(seed)
        self.seed = seed
        self.reset()
        for loop in range(iterations):
            outs = self.forward(X)
            self.backward(y)
            cost = np.sum((self.params["a"+str(self.size-1)]-y)**2)/len(y)
            self.costs.append(cost)
        print(cost)

    def train_batch(self,X, y, batchSize, iterations, log = False):
        best = 100
        seed = 0
        self.costs = []
        for i in range(100):
            np.random.seed(i)
            self.reset()
            dat = X
            tru = y
            sav = 0
            for loop in range(int(1000/batchSize)):
                for i in range(batchSize):
                    data = dat[0:batchSize]
                    true = tru[0:batchSize]
                    sav = true
                    outs = self.forward(data)
                    self.backward(true)
                    dat = dat[batchSize:len(dat)]
                    tru = tru[batchSize:len(tru)]
                    if len(data) == 0:
                        data = X
                        true = y
            cost = np.sum((self.params["a"+str(self.size-1)]-true)**2)/batchSize
            if cost < best:
                seed = i
            if log == True:
                print(i+1, "/100", sep='')
        np.random.seed(seed)
        self.seed = seed
        self.reset()
        dat = X
        tru = y
        sav = 0
        for loop in range(int(1000/batchSize)):
            for i in range(batchSize):
                data = dat[0:batchSize]
                true = tru[0:batchSize]
                sav = true
                #print(data, ":", true)
                outs = self.forward(data)
                self.backward(true)
                dat = dat[batchSize:len(dat)]
                tru = tru[batchSize:len(tru)]
                if len(data) == 0:
                    dat = X
                    tru = y
            cost = np.sum((self.params["a"+str(self.size-1)]-true)**2)/batchSize
            self.costs.append(cost)
            '''
        for loop in range(iterations):
            outs = self.forward(X)
            self.backward(y)
            cost = np.sum((self.params["a"+str(self.size-1)]-y)**2)/len(y)
            self.costs.append(cost)
            '''
        print(cost)

    def save(self, specs):
        open("KeyVals"+specs+".txt","w").close()
        file = open("KeyVals"+specs+".txt", "a")
        for i in range(1,self.size):
            np.savetxt(file, self.params["W" + str(i)], fmt = "%s")
            file.write(",\n")
        for i in range(1,self.size-1):
            np.savetxt(file, self.params["b" + str(i)], fmt = "%s")
            file.write(",\n")
        np.savetxt(file, self.params["b" + str(self.size-1)], fmt = "%s")
        #Don't Need Seed For Loading The Required Stuff
        file.close()

    def load(self, file):
        file = open(file,'r')
        line = file.readline()
        data = []
        while line != '':
            temp = []
            while not(line == '') and (line[0] != ","):
                if line[-1] == "C":
                    print(line)
                line = line.split()
                #print(line)
                for item in range(len(line)):
                    line[item] = float(line[item])
                temp.append(line)
                line = file.readline()
                #print(line=='')
            if line != '':
                line = file.readline()
            data.append(temp)
        #print(data)
        count = -1
        for i in range(1,int(len(data)/2)+1):
            self.params["W"+str(i)] = np.array(data[i-1])
            count+= 1
        count+=1
        for i in range(1,int(len(data)/2)+1):
            self.params["b"+str(i)] = np.array(data[count])
            count+=1
        
    def query(self, X, specs):
        file = "KeyVals"+specs+".txt"
        self.load(file)
        #self.end = self.softmax
        return self.forward(X)

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()
        



