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
import cv2


def initialise_variables():
    tf.reset_default_graph()
    W1 = tf.get_variable('W1',[5, 5, 1, 6])
    W2 = tf.get_variable('W2',[5, 5, 6, 8])
    W3 = tf.get_variable('W3',[5, 5, 8, 10])
    B1 = tf.get_variable('B1', [6])
    B2 = tf.get_variable('B2', [8])
    B3 = tf.get_variable('B3', [10])
    fully_connectedweights_1 = tf.get_variable('fully_connected/weights', [9000, 3000])
    fully_connectedweights_2 = tf.get_variable('fully_connected_1/weights', [3000, 400])
    fully_connectedweights_3 = tf.get_variable('fully_connected_2/weights', [400, 8])
    fully_connectedbiases_1 = tf.get_variable('fully_connected/biases', [3000])
    fully_connectedbiases_2 = tf.get_variable('fully_connected_1/biases', [400])
    fully_connectedbiases_3 = tf.get_variable('fully_connected_2/biases', [8])
    Ws = (W1, W2, W3, fully_connectedweights_1, fully_connectedweights_2, fully_connectedweights_3)
    Bs = (B1, B2, B3, fully_connectedbiases_1, fully_connectedbiases_2, fully_connectedbiases_3)
    return Ws, Bs


def restoremodel(W, B):
    saver = tf.train.Saver()
    W1, W2, W3, fully_connectedweights_1, fully_connectedweights_2, fully_connectedweights_3 = W
    B1, B2, B3, fully_connectedbiases_1, fully_connectedbiases_2, fully_connectedbiases_3 = B
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "../../data/emotionRecognizerData/my_model")
        print("Model restored.")
        W1_restore = W1.eval()
        W2_restore = W2.eval()
        W3_restore = W3.eval()
        B1_restore = B1.eval()
        B2_restore = B2.eval()
        B3_restore = B3.eval()
        W_fully_conn_1 = fully_connectedweights_1.eval()
        W_fully_conn_2 = fully_connectedweights_2.eval()
        W_fully_conn_3 = fully_connectedweights_3.eval()
        Bias_fully_conn_1 = fully_connectedbiases_1.eval()
        Bias_fully_conn_2 = fully_connectedbiases_2.eval()
        Bias_fully_conn_3 = fully_connectedbiases_3.eval()
    Ws = (W1_restore, W2_restore, W3_restore, W_fully_conn_1, W_fully_conn_2, W_fully_conn_3)
    Bs = (B1_restore, B2_restore, B3_restore, Bias_fully_conn_1, Bias_fully_conn_2, Bias_fully_conn_3)
    return Ws, Bs


def predict_model(restored_ws, restored_bs, test_x):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder("float32", [None, 256, 256, 1])

        """ normalizing the input test matrix"""
        def norm_calc(test_x):
            test_x_norm = (test_x / 255)
            return test_x_norm

        def frwd_propagation(X_input, W_Rs, B_Rs):
            W1, W2, W3, W_fully_conn_1, W_fully_conn_2, W_fully_conn_3 = W_Rs
            B1, B2, B3, Bias_fully_conn_1, Bias_fully_conn_2, Bias_fully_conn_3 = B_Rs
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
            Z4 = tf.contrib.layers.fully_connected(pool3, 3000, activation_fn=None, weights_initializer=tf.constant_initializer(W_fully_conn_1, dtype=tf.float32), biases_initializer=tf.constant_initializer(Bias_fully_conn_1, dtype=tf.float32))
            A4 = tf.nn.relu(Z4)
            Z5 = tf.contrib.layers.fully_connected(A4, 400, activation_fn=None, weights_initializer=tf.constant_initializer(W_fully_conn_2, dtype=tf.float32), biases_initializer=tf.constant_initializer(Bias_fully_conn_2, dtype=tf.float32))
            A5 = tf.nn.relu(Z5)
            #dropout regularization
            Z6 = tf.contrib.layers.fully_connected(A5, 8, activation_fn=None, weights_initializer=tf.constant_initializer(W_fully_conn_3, dtype=tf.float32), biases_initializer=tf.constant_initializer(Bias_fully_conn_3, dtype=tf.float32))
            A6 = tf.nn.sigmoid(Z6)
            return A6

        test_x_norm = norm_calc(x)
        A6 = frwd_propagation(test_x_norm, restored_ws, restored_bs)
        soft = tf.nn.softmax(A6)


    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        pred = tf.argmax(soft, 1)
        prediction = pred.eval({x: test_x})
        emotions_levels = soft.eval({x: test_x})
        print("Emotion", emotions_levels)

    return prediction