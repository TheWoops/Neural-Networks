# Neuronales Netz für MNIST dataset Klassifkation
from tensorflow.examples.tutorials.mnist import input_data # helper function to load data
import tensorflow as tf
import numpy as np
import os

# Storage Path dataset
dir_path = "C:/Users/The Woops/Documents/Wissensdatenbank/udemy/Udemy Kurs/Daten"
# load data and save it under dir_paths and perform OHE
mnist = input_data.read_data_sets("dir_path", one_hot=True) #  deprecated (to fix)

# Train / Test set
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels
print("Train_size: " , X_train.shape) # 55k, 784
print("Test_size: " , Y_train.shape) # 55k
# print(Y_train[1]) # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] This would be a 3
# print(X_train[1]) # Normalisiert

target_classes = 10 # 0-9 digits (10 target_classes)
features = 784 # total pixels (28x28)

# Definition Model-Nodes (input one image) 
layer_nodes = [features, 250, target_classes] # input, hidden (250 neurons - adaptable), output

# Hyperparameters (fixed during training)
epochs = 10
batchs_size = 128  # how many images I want to be processed in parallel, # enter 2 potency as batch-size (better for graphic board)            
learning_rate = 1e-3
stddev = 0.100 # für Initialisierung Gewichte
bias_init = 0.0 # Initialisieung Bias

# TF Placeholders (input / output):
    # constants provide a value at the time of defining the tensor
    # placeholders allow to create tensors whose values can be provided at runtime.
x = tf.placeholder(dtype = tf.float32, shape=[None, features], name = "x")
y = tf.placeholder(dtype = tf.float32, shape=[None, target_classes], name = "y") # das was eigentlich rauskommen soll

# Weights
W1 = tf.Variable(tf.truncated_normal(shape =[layer_nodes[0], layer_nodes[1]], stddev=stddev, name = "W1")) # Input zu Hidden: Normalverteilung truncated [-2*stdev, + 2*stddev]
W2 = tf.Variable(tf.truncated_normal(shape =[layer_nodes[1], layer_nodes[2]], stddev=stddev, name = "W2")) # Hidden zu Output
# Biases 
    # bestimmt , wie stark der kumulierte Reiz sein muss, um das Neuron überhaupt anzuregen.
    #  Bias verschiebt Grundniveau der Aktivierung.  Neuron aktiveren/deaktivieren?   
b1 = tf.Variable(tf.constant(bias_init, shape = [layer_nodes[1]], name = "b1" )) # Hidden
b2= tf.Variable(tf.constant(bias_init, shape = [layer_nodes[2]], name = "b2" )) # Output

#def nn_model(x):












