#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:36:34 2019

@author: aniketpramanick
"""
import os, math, random, PIL, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

"""class NeuralNet:
    def __init__(self, filename):
        self.filename = filename
        info = list(os.walk(filename))
        root = info[0][0]
        dirs = info[0][1]
        self.classes = dirs
        data = list()
        for d in dirs:
            dir_path = os.path.join(root, d)
            files = os.listdir(dir_path)
            tmp1 = float(dirs.index(d)+1)/len(dirs)
            sys.stdout.write('{}% Directory Reading Completed .....\n'.format(tmp1*100))
            for f in files:
                file_path = os.path.join(dir_path, f)
                img = PIL.Image.open(file_path).convert("L")
                matrix = np.array(img)
                shape = matrix.shape
                reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ])
                data.append((reshaped_matrix, d))
            #labels.append(d)
                tmp2 = float(files.index(f)+1)/len(files)
                sys.stdout.write('{}% File Reading Completed .....\r'.format(int(tmp2*100)))
            
            sys.stdout.write('{}% File Reading Completed .....\n'.format(int(100)))
        
        sys.stdout.write('{}% Directory Reading Completed .....\n'.format(int(100)))
        self.data = np.array(data)"""
    
    


def one_hot(y, num_classes):
    onehot = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        onehot[i, int(y[i])] = 1
    return onehot


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))



def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
    

def softmax(x):
    exps = np.exp(x-np.max(x))
    return exps/np.sum(exps)
    #return np.exp(x) / np.sum(np.exp(x), axis=0)


def read_data(filename):
    #i = 0
    info = list(os.walk(filename))
    root = info[0][0]
    dirs = info[0][1]
    num_classes = len(dirs)
    data = list()
    labels = list()
    for d in dirs:
        dir_path = os.path.join(root, d)
        files = os.listdir(dir_path)
        c = float(dirs.index(d)+1)/len(dirs)
        sys.stdout.write('{}% Directory Reading Completed .....\n'.format(int(c*100)))
        for f in files:
            #print "Reading File", f
            file_path = os.path.join(dir_path, f)
            img = PIL.Image.open(file_path).convert("L")
            matrix = np.array(img)
            shape = matrix.shape
            reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ]) 
            data.append(reshaped_matrix)
            labels.append(d)
            e = float(files.index(f)+1)/len(files)
            sys.stdout.write('{}% File Reading Completed .....\r'.format(int(e*100)))
            
        sys.stdout.write('{}% File Reading Completed .....\n'.format(int(100)))
        """print np.array(img).shape
            plt.imshow(np.array(img))
            plt.show()
            return"""
    sys.stdout.write('{}% Directory Reading Completed .....\n'.format(int(100)))    
    return np.array(data), np.array(labels), num_classes

def initialize_parameters(configuration):
    w = dict()
    b = dict()
    for layer in range(1, len(configuration)):
        w[layer] = np.random.random_sample((configuration[layer], configuration[layer-1]))
        b[layer] = np.random.random_sample((configuration[layer], ))
    print "Weights Initialized ....."
    return w, b

def initialize_del_parameters(configuration):
    del_w = dict()
    del_b = dict()
    for layer in range(1, len(configuration)):
        del_w[layer] = np.zeros((configuration[layer], configuration[layer-1]))
        del_b[layer] = np.zeros((configuration[layer], ))
    print "Derivative accumulators initialized ....."
    return del_w, del_b

def delta_out(output, output_h, output_eta):
    output_h_softmax = softmax(output_h)
    return -(output-output_h_softmax)*dsigmoid(output_eta)

def delta_inside(delta_next, w_l, eta_l):
    return np.dot(np.transpose(w_l), delta_next)*dsigmoid(eta_l)
    

def forward_prop(X, w, b):
    eta = dict()
    h = dict()
    h[1] = X
    for layer in range(1, len(w)+1):
        if layer == 1:
            inp = X
        else:
            inp = h[layer]
        eta[layer+1] = w[layer].dot(inp) + b[layer]
        h[layer+1] = sigmoid(eta[layer+1])
    return eta, h
"""def back_prop(configuration, delta, del_w, del_b): 
    delta = dict()
    for layer in range(len(configuration), 0, -1):
        if layer == len(configuration):
            delta[layer] = delta_out()
    """
    
def batch_generator(X, Y, batch_size):
    num_batches = len(Y)/batch_size
    for i in range(num_batches):
        start = i*batch_size
        yield X[start:start+batch_size], Y[start:start+batch_size]

"""def train2(configuration, x_train, y_train, epoch = 5000, learning_rate = 0.01, regularizer = 0.001):
    print "Training Started with "+str(epoch)+" iterations ....."
    w, b = initialize_parameters(configuration)
    num_data = len(y_train)
    mean_cost = list()
    for i in range(epoch):
        if (i+1)%5 == 0:
            print "Epoch Completed:", i+1
        del_w, del_b = initialize_del_parameters(configuration)
        cost = 0
        delta = dict()
        eta, h = forward_prop(x_train, w, b)
        #Back Prop
        for layer in range(len(configuration), 0, -1):
            if layer == len(configuration):
                delta[layer] = delta_out(y_train, h[layer], eta[layer])
                cost += np.linalg.norm((y_train-h[layer]))
            else:
                if layer > 1:
                    delta[layer] = delta_inside(delta[layer+1], w[layer], eta[layer])
                del_w[layer] += np.dot(delta[layer+1][:, np.newaxis], np.transpose(h[layer][:, np.newaxis]))
                del_b[layer] += delta[layer+1]
        #Gradient Descent
        for layer in range(len(configuration)-1, 0, -1):
            w[layer] += -learning_rate*(del_w[layer]/float(num_data) + regularizer*w[layer])
            b[layer] += -learning_rate*(del_b[layer]/float(num_data))
        mean_cost.append(float(cost)/num_data)
    return w, b, mean_cost"""
                

def train(configuration, x_train, y_train, epoch = 5000, learning_rate = 0.01, regularizer = 0.00001):
    print "Training Started with "+str(epoch)+" iterations ....."
    w, b = initialize_parameters(configuration)
    num_data = len(y_train)
    mean_cost = list()
    for i in range(epoch):
        if (i+1)%5 == 0:
           print "Epoch:", i+1
        del_w, del_b = initialize_del_parameters(configuration)
        cost = 0
        for j in range(num_data):
            delta = dict()
            eta, h = forward_prop(x_train[j, :], w, b)
            #Back Propagation
            for layer in range(len(configuration), 0, -1):
                if layer == len(configuration):
                    delta[layer] = delta_out(y_train[j, :], h[layer], eta[layer])
                    cost += np.linalg.norm((y_train[j, :]-h[layer]))
                else:
                    if layer > 1:
                        delta[layer] = delta_inside(delta[layer+1], w[layer], eta[layer])
                    del_w[layer] += np.dot(delta[layer+1][:, np.newaxis], np.transpose(h[layer][:, np.newaxis]))
                    del_b[layer] += delta[layer+1]
        #Gradient Descent
        for layer in range(len(configuration)-1, 0, -1):
            w[layer] += -learning_rate*(del_w[layer]/float(num_data) + regularizer*w[layer])
            b[layer] += -learning_rate*(del_b[layer]/float(num_data))
        if (i+1)%5==0:
            print "Mean Cost:", float(cost)/num_data
        mean_cost.append(float(cost)/num_data)
    return w, b, mean_cost








def predict(x_test, w, b, num_layers):
    num_data = len(x_test)
    prediction = np.zeros((num_data, ))
    for i in range(num_data):
        eta, h = forward_prop(x_test[i, :], w, b)
        prediction[i] = np.argmax(h[num_layers])
    return prediction
                        
            
            
        
        
        
        

def main():
    raw_data, raw_labels, num_classes = read_data("../data/MNIST")
    #print raw_data.shape
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)
    print "Scaling Completed ....."
    n_components=64
    pca = PCA(n_components)
    data = pca.fit_transform(scaled_data)
    print "Dimensionaly Reduced ....."
    if num_classes == 2:
        labels = np.array([0 if x=='cat' else 1 for x in raw_labels])
    elif num_classes == 10:
        labels = np.array(list(map(int, raw_labels)))
    d1, d2, l1, l2 = train_test_split(data, labels, test_size = 0.1, shuffle = True)
    onehot_l1 = one_hot(l1, num_classes)
    
    #Defining Neural Network Structure
    neural_structure = [n_components,30, 10]
    w, b, cost_list = train(neural_structure, d1, onehot_l1)
    
    predicted_y = predict(d2, w, b, len(neural_structure))
    print "Accuracy: ", accuracy_score(l2, predicted_y)*100
    print "F1 Score(Micro):", f1_score(l2, predicted_y, average = "micro")
    print "F1 Score(Macro):", f1_score(l2, predicted_y, average = "macro")
    

if __name__ == "__main__":
    main()