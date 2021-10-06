from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn import preprocessing
import numpy as np
import pandas as pd


DATA_PATH = 'groupC.txt'

# In class Perceptron training Example
def sign(net):
    if net >= 0:
        return 1
    else:
        return -1

def printData(iteration, pattern, net, err, learn, ww):
    ww_formatted = [ '%.2f' % elem for elem in ww]
    print("ite= ", iteration, ' pat= ', pattern, ' net= ', round(net, 5), 
        ' error= ', err, ' lrn= ', learn, ' weights= ', ww_formatted)

ite = 8 # number of training cycles


data_path = input("Enter filename for dataset: ")

# Stats for confusion matrix
true_positives = 0
true_negatives = 0
false_negatives = 0
false_positives = 0

# Data Input
data = pd.read_csv(DATA_PATH, sep=',', header=None)

# min-max normalization
scaler = preprocessing.MinMaxScaler()
df = pd.DataFrame(data)
df[[0,1]] = scaler.fit_transform(df[[0,1]])

# Separating data into columns
x = np.array(df[0])
y = np.array(df[1])
z = np.array(data[2])

i = 0
for zi in z:
    # If car is small
    if (zi == 0):
        # print("here in if!")
        # if car was identified as large
        if is_above([df[0][i], df[1][i]], lob_b, lob_a):
            false_negatives += 1
        # Else- car was correctly identified as small
        else: 
            true_positives += 1
    # Else if car is large
    elif (zi == 1):
        if is_above([df[0][i], df[1][i]], lob_b, lob_a):
            true_negatives += 1
        else:
            false_positives += 1
    i += 1

print("True Positives: " + str(true_positives))
print("True Negatives: " + str(true_negatives))
print("False Positives: " + str(false_positives))
print("False Negatives: " + str(false_negatives))

# Create scatter plot and display it to screen
plt.scatter(x, y, c=z)

plt.show()