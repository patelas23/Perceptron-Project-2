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

ite = 500 # number of training cycles


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

# Create scatter plot and display it to screen
plt.scatter(x, y, c=z)

plt.show()