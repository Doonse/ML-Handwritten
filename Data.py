from Imports import *

# Read data 
data = np.genfromtxt("optdigits.csv", delimiter=" ")# shape of data (3823, 65)

# Data
train_size = int(0.8 * data.shape[0]) # 3058
train_data = data[:train_size, :] # (3058, 65)
(train_images, train_labels) = data[:train_size, 1:], data[:train_size, 0]
(test_images, test_labels) = data[train_size:, 1:], data[train_size:, 0]

# Label array
labels_array = np.zeros((train_size, 10)) 
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 10 labels, 0-9
for i in range(train_size):
    idx = labels.index(train_labels[i]) # Find index of the matching label
    if labels[idx] == labels.index(train_labels[i]):
        labels_array[i,idx] =  1       # Add 1 to the matching label, leaving the other labels as 0. Example:  Label = 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# Constants
learning_rate = 0.01 # how fast the network learns
max_epochs = 20 # number of times to go through the data

# Dimensions of layers
input_dim = len(data[0]) - 1
hidden_dim = 100 # number of nodes in hidden layer
output_dim = 10 # 0-9

# Weights and Bias
w_ih = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim)) # weights input -> hidden
w_ho = np.random.uniform(-0.5, 0.5, (output_dim, hidden_dim)) # weights hidden -> output
b_ih = np.zeros((hidden_dim, 1))
b_ho = np.zeros((output_dim, 1))