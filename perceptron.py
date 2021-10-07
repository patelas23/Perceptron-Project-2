import math
import numpy as np
import random as rand

class Perceptron:

    MAX_ITE = 5000
    
    def __init__(self, learning_rate=0.3, ite=200):
        self.alpha = learning_rate
        self.ite = ite # number of training cycles
        self.np = 400 # number of patterns
        self.activator = self.sign

        # Initialize weights with values between -0.5 and 0.5
        self.weights = [rand.random()/-2, rand.random()/-2]

        # Initialize counters for confusion stats
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    # Iterative function for testing current weights against a set of inputs
    # IN -  ni: number of inputs per sample
    #       pat: 2D array of inputs
    #       dout: 1D array of desired outputs
    #       train_size: size of training samples
    #       data_size: total size of training samples
    def test_data(self, ni, pat, dout, train_size, data_size, soft):
        # For each test sample, 
        #   beginning from end of test data to end of dataset
        ou = [0] * data_size - train_size # Array to store output
        for pattern in range(train_size, data_size):
            net = 0
            # For each data point in the sample
            for i in range(0, ni):
                # Multiply weights and points
                net = net + self.weights[i] * pat[pattern][i] 

            if(soft):
                ou[pattern] = self.softActivator(net)
            else:
                ou[pattern] = self.activator(net)

            # Determine whether the sample was classified correctly
            if((ou[pattern] == 1) and (dout[pattern] == 1)):
                self.true_positives += 1

            elif(ou[pattern] == -1 and dout[pattern] == -1):
                self.true_negatives += 1

            elif(ou[pattern] == 1 and dout[pattern] == -1):
                self.false_positives += 1
            
            elif(ou[pattern] == -1 and dout[pattern] == 1):
                self.false_negatives += 1

        print("True Positives: ", self.true_positives)
        print("True Negatives: ", self.true_negatives)
        print("False Positives: ", self.false_positives)
        print("False Negatives: ", self.false_negatives)



    # Iterative function to train perceptron over a specified portion of the data
    # IN -  ni: number of inputs
    #       pat: 2D array of inputs
    #       dout: 1D array of desired outputs 
    #       stopCrit: maximum error before stopping
    #       train_size: number of samples to use for testing
    #       data_size: number of samples in entire dataset    
    def train_data(self, ni, pat, dout, stopCrit, train_size, data_size, soft=False):
        # For each training cycle
        for iteration in range(0, self.ite):
            ou = [0] * train_size # Temporary empty array to store output
            # For each row in training set
            for pattern in range(0, train_size):
                te = 0 # Total error
                net = 0 
                # For each input
                for i in range(0, ni):
                    net = net + self.weights[i]*pat[pattern][i]

                # Use activation function
                if(soft):
                    ou[pattern] = self.softActivator(net)
                else:
                    ou[pattern] = self.sign(net)

                err = dout[pattern] - ou[pattern]
                te = te + (err ** 2)

                learn = self.alpha * err
                 # Update weights
                for i in range(0, ni):
                    self.weights[i] = self.weights[i] + learn*pat[i][pattern]

            print("Total error: ", te)

            # Stopping criterion
            if(te < stopCrit or iteration >= self.MAX_ITE):
                return self.weights

    # Function to update weights (refactoring from above)
    def learning_cycle(self, ni):
        net = 0

    def printData(self, iteration, pattern, net, err, learn, ww):
        ww_formatted = [ '%.2f' % elem for elem in ww]
        print("ite= ", iteration, ' pat= ', pattern, ' net= ', round(net, 5), 
            ' error= ', err, ' lrn= ', learn, ' weights= ', ww_formatted)

    
    def softActivator(self, net):
        k = 0.2
        return 2/(1 + math.exp(-2*k*net) ) -1
             

    def sign(self, net):
        return np.where(net>=0, 1, -1)

    