from __future__ import division
import os
import gzip
import tarfile
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as ss
import matplotlib.image as img
import tensorflow as tf
from sklearn.utils import shuffle


"""loading emotion labels from file"""
lables_y = np.load('../../data/emotionRecognizerData/labels.npy')
"""loading images - pixel matrices from file"""
images = np.load('../../data/emotionRecognizerData/image_matrix.npy')

# images, lables_y = shuffle(images, lables_y, random_state=0)


"""splitting to train and test"""
# train_images = images[0:300]
# test_images = images[300:327]
# train_y = lables_y[0:300]
# test_y = lables_y[300: 327]

train_images = images
train_y = lables_y

"""normalization"""
train_x_norm = (train_images / 255)
# test_x_norm = (test_images / 255)


"""reshaping X to contain channels = 1"""
#print(train_x_norm.shape)
train_x_norm1 = train_x_norm.reshape(train_x_norm.shape[0], train_x_norm.shape[1], train_x_norm.shape[2], 1)
# test_x_norm1 = test_x_norm.reshape(test_x_norm.shape[0], test_x_norm.shape[1], test_x_norm.shape[2], 1)



"""one hot encoding function"""
def One_Hot_Encoding(arr, samples_num):
    encode_matrix = np.zeros((samples_num, 8))
    for i in range(samples_num):
        encode_matrix[i][int(float(arr[i][0]))] = 1
    return encode_matrix

train_y_encode = One_Hot_Encoding(train_y, len(train_y))
# test_y_encode = One_Hot_Encoding(test_y, len(test_y))



"""function to initialize weights and bias for convolution 2 layers"""
def initialize_weights():
    tf.set_random_seed(1)                              
        
    W1 = tf.get_variable('W1',[5, 5, 1, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2',[5, 5, 6, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable('W3',[5, 5, 8, 10], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    B1 = tf.get_variable('B1', [6], initializer=tf.zeros_initializer())
    B2 = tf.get_variable('B2', [8], initializer=tf.zeros_initializer())
    B3 = tf.get_variable('B3', [10], initializer=tf.zeros_initializer())
    weights = ( W1, W2, W3)
    bias = ( B1,B2, B3)

    return weights, bias



"""forward propagation. convolution and fully connected. dropout is applied as regularization for fully conencted layers"""

def frwd_propagation(X_input, weights, bias):
    W1, W2, W3 = weights
    B1, B2, B3 = bias
    # first convolution with relu as activation
    conv1 = tf.nn.conv2d(X_input, W1, strides = [1,1,1,1], padding = 'SAME', name='conv1')
    print("conv1 layer shape")
    print(conv1.shape)
    Z1 = tf.nn.bias_add(conv1, B1)
    A1 = tf.nn.relu(Z1)
    #max pooling
    pool1 = tf.nn.max_pool(A1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
    print("pool1 shape")
    print(pool1.shape)
    # 2nd convolution layer with relu as activation
    conv2 = tf.nn.conv2d(pool1, W2, strides = [1,1,1,1], padding = 'VALID', name='conv2')
    Z2 = tf.nn.bias_add(conv2, B2)
    A2 = tf.nn.relu(Z2)
    print("conv2 shape")
    print(conv2.shape)
    #avg pooling
    pool2 = tf.nn.max_pool(A2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
    print("pool2 shape before unfaltten")
    print(pool2.shape)
    conv3 = tf.nn.conv2d(pool2, W3, strides = [1,1,1,1], padding = 'SAME', name='conv3')
    Z3 = tf.nn.bias_add(conv3, B3)
    A3 = tf.nn.relu(Z3)
    print("conv3 shape")
    print(conv3.shape)
    pool3 = tf.nn.max_pool(A3, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
    print("pool3 shape before unfaltten")
    print(pool3.shape)
    pool3 = tf.contrib.layers.flatten(pool3)
    print("pool3 shape after flatten")
    print(pool3.shape)
    #fully connected layer relu as activation
    Z4 = tf.contrib.layers.fully_connected(pool3, 3000, activation_fn=None)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.contrib.layers.fully_connected(A4, 400, activation_fn=None)
    A5 = tf.nn.relu(Z5)
    #dropout regularization
    Z6 = tf.contrib.layers.fully_connected(A5, 8, activation_fn=None)
    A6 = tf.nn.sigmoid(Z6)
    return A6



# def model(X_train, Y_train, X_test, Y_test, iterations = 2, learning_rate = 0.0001, minibatch_size = 15):
def model(X_train, Y_train, iterations = 2, learning_rate = 0.0001, minibatch_size = 15):

    tf.set_random_seed(1)                                                                       
    (m, n_h, n_w, n_c) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = [] 
    seed = 3
    weights, bias = initialize_weights()
    W1, W2, W3 = weights
    B1, B2, B3 = bias
    X = tf.placeholder(tf.float32, shape=[None, n_h, n_w, n_c])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])
    #forward propagation
    A5 = frwd_propagation(X, weights, bias)
    #calcualting cost
    cost_total = tf.nn.softmax_cross_entropy_with_logits(logits = A5, labels = Y)
    cost = tf.reduce_mean(cost_total )
    #back propagation
    adamoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    soft = tf.nn.softmax(A5)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    
    with tf.Session() as sess:
        
        sess.run(init_op)
        for iteration in range(iterations):
            
            print("iteration is %s" %iteration)
            cost_min_batch = 0
            seed = seed + 1
            
            minbatches = generate_min_batch(X_train, Y_train, seed, minibatch_size)
            minbatch_num = int(m / minibatch_size)
            for batch in minbatches:

                (min_batch_X, min_batch_Y) = batch
                _ , cost_single_batch = sess.run([adamoptimizer, cost], feed_dict={X: min_batch_X, Y: min_batch_Y})
                cost_min_batch += cost_single_batch / minbatch_num
            print("iteration %s done" %iteration)  
            if iteration % 1 == 0:
                print("cost is %s " %cost_min_batch)
        print("iterations are done")
        

        # Calculate correct predictions
        predict_y = tf.argmax(A5, 1)
        print("prediction y done")
        correct_prediction = tf.equal(predict_y, tf.argmax(Y, 1))
        print("corect pred computed")
        
        # Calculate accuracy on the TRAIN SET AND test set
        accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        # test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # print("pred : %s" %predict_y.eval({X: X_test, Y: Y_test}))
        print("Train Accuracy:", train_accuracy)
        # print("Test Accuracy:", test_accuracy)
        save_path = saver.save(sess, '../../data/emotionRecognizerData/my_model')
        print("Model saved in path: %s" % save_path)
        #print(Weights1)       
        return train_accuracy, weights, bias, costs


def generate_min_batch(X_train, Y_train, seed, minbatch_size):
    m = X_train.shape[0]
    #print("this is m %s" %m)
    np.random.seed(seed)
    random_p = np.random.permutation(m)
    #print("permutaion %s " %random_p)
    X_shuffled = X_train[random_p, :]
    Y_shuffled = Y_train[random_p, :]
    
    num_min_batches = m/minbatch_size
    min_batches = []
    for i in range(0, int(num_min_batches)):
        batch_X = X_shuffled[i * minbatch_size:(i + 1) * minbatch_size, :]
        batch_Y = Y_shuffled[i * minbatch_size:(i + 1) * minbatch_size, : ] 
        min_full_batch = (batch_X, batch_Y)
        min_batches.append(min_full_batch)
    
    return min_batches


from datetime import datetime
start=datetime.now()
# train_accuracy, test_accuracy, weights, bias, costs = model(train_x_norm1, train_y_encode, test_x_norm1, test_y_encode, 20)
train_accuracy, test_accuracy, weights, bias, costs = model(train_x_norm1, train_y_encode, 43)
end = datetime.now() - start

