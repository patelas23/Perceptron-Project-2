from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn import preprocessing
import numpy as np
import pandas as pd


DATA_PATH = 'groupC.txt'

DATA_SET_A = "groupA.txt"
DATA_SET_B = "groupB.txt"
DATA_SET_C = "groupC.txt"

ite = 500 # number of training cycles


data_path = input("Enter filename for dataset: ")

# TODO: instantiate Perceptron(s)
soft_perc_a = None
soft_perc_b = None
soft_perc_c = None

hard_perc_a = None
hard_perc_b = None
hard_perc_c = None

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

# TODO: Run perceptron on each dataset, using both hard and soft activation
# TODO: Plot corresponding line for each perceptron 
#   y = mx + b //Matlab desired form

# Create scatter plot and display it to screen
plt.scatter(x, y, c=z)

plt.show()