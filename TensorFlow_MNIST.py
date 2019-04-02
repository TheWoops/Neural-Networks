# Neuronales Netz für MNIST dataset Klassifkation
from tensorflow.examples.tutorials.mnist import input_data # helper function to load data
import tensorflow as tf
import numpy as np
import os

from plotting import *

# Storage Path dataset
dir_path = "C:/Users/The Woops/Documents/Wissensdatenbank/udemy/Udemy Kurs/Daten"
# load data and save it under dir_paths and perform OHE
mnist = input_data.read_data_sets("dir_path", one_hot=True) #  deprecated (to fix)

# Train / Test set
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels
train_size = mnist.train.num_examples 
test_size = mnist.test.num_examples 

print("Train_size: " , X_train.shape) # 55k, 784
print("Test_size: " , Y_train.shape) # 55k
# print(Y_train[1]) # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] This would be a 3
# print(X_train[1]) # Normalisiert

target_classes = 10 # 0-9 digits (10 target_classes)
features = 784 # total pixels (28x28)

# Definition Model-Nodes (input one image) 
layer_nodes = [features, 250, target_classes] # input, hidden (250 neurons - adaptable), output
stddev = 0.100 # für Initialisierung Gewichte
bias_init = 0.0 # Initialisieung Bias

# Hyperparameters (fixed during training)
epochs = 30
train_batch_size = 128  # how many images I want to be processed in parallel, # enter 2 potency as batch-size (better for graphic board)            
test_batch_size = 128
learning_rate = 1e-3

# Helper variables
train_mini_batches = int(train_size/ train_batch_size) + 1
test_mini_batches = int(test_size/ test_batch_size) + 1
    # Für die Visualisierung des Trainings
train_errors, test_errors = [],[]
train_accs, test_accs = [], []

# TF Placeholders (input / output):
    # constants provide a value at the time of defining the tensor
    # placeholders allow to create tensors whose values can be provided at runtime.
x = tf.placeholder(dtype = tf.float32, shape=[None, features], name = "x") # was wir reingeben
y = tf.placeholder(dtype = tf.float32, shape=[None, target_classes], name = "y") # das was eigentlich rauskommen soll

# Weights
W1 = tf.Variable(tf.truncated_normal(shape =[layer_nodes[0], layer_nodes[1]], stddev=stddev, name = "W1")) # Input zu Hidden: Normalverteilung truncated [-2*stdev, + 2*stddev]
W2 = tf.Variable(tf.truncated_normal(shape =[layer_nodes[1], layer_nodes[2]], stddev=stddev, name = "W2")) # Hidden zu Output
# Biases 
    # bestimmt , wie stark der kumulierte Reiz sein muss, um das Neuron überhaupt anzuregen.
    # Bias verschiebt Grundniveau der Aktivierung.  Neuron aktiveren/deaktivieren?   
b1 = tf.Variable(tf.constant(bias_init, shape = [layer_nodes[1]], name = "b1" )) # Hidden
b2= tf.Variable(tf.constant(bias_init, shape = [layer_nodes[2]], name = "b2" )) # Output

def nn_model(x):
    input_layer_dict = {"weights": W1, "biases": b1} # Input to Hidden
    hidden_layer_dict = {"weights": W2, "biases": b2} # Hidden to Output
    # Input Layer
    input_layer = x # shape(?,784) hängt übergebendem Parameter ab
    # From Input to Hidden Layer
    hidden_layer_in = tf.add(  # vgl. Keras output = activation(dot(input, kernel) + bias)
        tf.matmul(input_layer, input_layer_dict["weights"]),
        input_layer_dict["biases"])
    hidden_layer_out = tf.nn.relu(hidden_layer_in) # Relu Aktivierungsfunktion
    # From Hidden Layer to Output Layer
    output_layer = tf.add(
        tf.matmul(hidden_layer_out, hidden_layer_dict["weights"]), 
        hidden_layer_dict["biases"]) 
        # Keine Aktivierungsfunktion, weil wir Softmax + Cross entropy am Ende nutzen (In Kostenfunktion)
    return output_layer

# Train and Test the Neural Network
def nn_run():
    # TensorFlow Ops
    pred = nn_model(x) # = Output von Netzwerk
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
        # 1. tf.nn.softmax: wandelt output-tensor in Wahrscheinlichkeiten um (Normalisierung)
        # 2. cross_entropy: versucht Error zwischen y und y_pred zu minimieren (passt Gewichte an)
        # = Fehlerfunktion
        # 3. tf.reduce_mean: Mittelwert von Fehlerfunktion bildens
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
       # AdamOptimizer = Erweiterung zum stochastic gradient descent (weiter verbreitet)
       # Kostenfunktion soll mit AdamOptmizer minimiert (* optimiert) werden
    correct_result = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y, axis = 1))
        # Scoring: Wie oft liegen wir richtig?
        # z.B. pred = [0.1, 0.9]
        #     y_true =[ 0 ,  1 ]
        # tf.argmax(pred) = 1 - An Index 1 ist größter Wert
    accuracy = tf.reduce_mean(tf.cast(correct_result, tf.float32))

    # Start TensorFlow Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
      
        # Training
        for epoch in range(epochs):
            train_acc, train_loss = 0.0, 0.0
            test_acc, test_loss = 0.0, 0.0

            # Train the weights
                # wir wollen Dataset in Mini-Batches ausführen, ganzes Dataset zu groß
                # Wie viele Bilder haben wir: 1000 / 16 = 6.25 → + 1 = 7 damit keine Bilder verloren gehn
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist.train.next_batch(train_batch_size)
                feed_dict = {x: epoch_x, y:epoch_y } 
                # Optimizer geht zurük bis dahin, wo er was minimieren kann
                    # Optimizer -> Cost -> Pred -> NN
                sess.run(optimizer, feed_dict=feed_dict)

            # Check the performance of the train set 
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist.test.next_batch(test_batch_size)
                feed_dict = {x: epoch_x, y:epoch_y } 
                a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
                train_acc += a
                train_loss +=  c 
            # avg. performance in diesem batch
            train_acc = train_acc / train_mini_batches
            train_loss = train_loss / train_mini_batches
            train_accs.append(train_acc)
            train_errors.append(train_loss)

            # Check the performance of the test set 
            for i in range(test_mini_batches):
                epoch_x, epoch_y = mnist.test.next_batch(test_batch_size)
                feed_dict = {x: epoch_x, y:epoch_y } 
                a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
                test_acc += a
                test_loss +=  c 
            # avg. performance in diesem batch
            test_acc = test_acc / test_mini_batches
            test_loss = test_loss / test_mini_batches
            test_accs.append(test_acc)
            test_errors.append(test_loss)

            print("Epoch: ", epoch+1, " of ", epochs, 
                   " - Train Loss: ", round(train_loss , 3), 
                   " Train Acc: ", round(train_acc, 3), 
                   " - Test Loss: ", round(test_loss , 3), 
                   " Test Acc: ", round(test_acc, 3))

        # Testing every Mini-batch
        test_acc, test_loss = 0.0, 0.0
        for i in range(test_mini_batches):
            epoch_x, epoch_y = mnist.test.next_batch(test_batch_size)
            feed_dict = {x: epoch_x, y:epoch_y } 
            a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
            test_acc += a
            test_loss +=  c 
        # avg. performance in diesem batch
        test_acc = test_acc / test_mini_batches
        test_loss = test_loss / test_mini_batches

        print("Test Loss: ", round(test_loss, 3), " Test Acc; ", round(test_acc , 3))


    # Visualisierung Performance/Training
    display_convergence_error(train_errors, test_errors)
    display_convergence_acc(train_accs, test_accs)

# Sicherstellen, dass Run-Funktion nur ausgeführt wird, 
# wenn wir Datei als Main ausführen (wenn console: "python TensorFlow_MNist.py")
# und nicht, wenn wir sie z.B. inkludieren
if __name__ == "__main__":
    nn_run()










