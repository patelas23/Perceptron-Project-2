from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn import preprocessing
from perceptron import Perceptron
import numpy as np
import pandas as pd


DATA_PATH = 'groupA.txt'

DATA_SET_A = "groupA.txt"
DATA_SET_B = "groupB.txt"
DATA_SET_C = "groupC.txt"

ite = 500 # number of training cycles


data_path = input("Enter filename for dataset: ")

# TODO: instantiate Perceptron(s)
soft_perc_a = Perceptron()
soft_perc_b = Perceptron()
soft_perc_c = Perceptron()

hard_perc_a = Perceptron()
hard_perc_b = Perceptron()
hard_perc_c = Perceptron()

# Data Input
data = pd.read_csv(DATA_PATH, sep=',', header=None)

# min-max normalization
scaler = preprocessing.MinMaxScaler()
df = pd.DataFrame(data)
df[[0,1]] = scaler.fit_transform(df[[0,1]])

# Separating data into columns
x = np.array(df[0])
y = np.array(df[1])
data_pattern = np.array([x, y])
print(data_pattern.shape)
z = np.array(data[2])

# Combine x and y arrays

# TODO: Run perceptron on each dataset, using both hard and soft activation
# Training first soft perceptron on 75% of the dataset
trained_weights = soft_perc_a.train_data(2, data_pattern, z, 0.00005, 3000, 4000, True)
# soft_perc_a.test_data()
# TODO: Plot corresponding line for each perceptron 
#   y = mx + b //Matlab desired form

# Create scatter plot and display it to screen
plt.scatter(x, y, c=z)

plt.show()