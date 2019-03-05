#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:36:18 2019

@author: aniketpramanick
"""
import os, math, random, PIL, sys, argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import nn
import pickle as pkl



def main():
    parser = argparse.ArgumentParser(description = "Command Line Parser")
    parser.add_argument("--train-data", dest = "train_data", default = '$', help = "Give path to the training data set.")
    parser.add_argument("--test-data", dest = "test_data", required = True, help = "Give path to the testing data set.")
    parser.add_argument("--dataset", dest = "dataset", required = True, help = "Give Name of data set.")
    parser.add_argument("--configuration", dest = "configuration", default = '$', help = "Give structure of Neural Net.")
    args = parser.parse_args()
    train_data = args.train_data
    test_data = args.test_data
    dataset = args.dataset
    configuration = args.configuration
    if configuration != '$':
        configuration = list(map(int, configuration.strip('[]').split(',')))
    #Test mode
    if train_data == '$':
        
        test_filename = test_data
        testdata = list()
        if dataset == 'MNIST':
            num_classes = 10
            
            with open("../mem/weight_mnist.pkl", 'r') as saved:
                w = pkl.load(saved)
            with open("../mem/bias_mnist.pkl", 'r') as saved:
                b = pkl.load(saved)
            with open("../mem/configuration_mnist.pkl", 'r') as saved:
                nn_structure = pkl.load(saved)
            
            
            files = os.listdir(test_filename)
            for f in files:
            #print "Reading File", f
                file_path = os.path.join(test_filename, f)
                img = PIL.Image.open(file_path).convert("L")
                matrix = np.array(img)
                shape = matrix.shape
                reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ]) 
                testdata.append(reshaped_matrix)
                e = float(files.index(f)+1)/len(files)
            sys.stdout.write('{}% File Reading Completed .....\r'.format(int(e*100)))
            test_scaler = StandardScaler()
            testdata = test_scaler.fit_transform(np.array(testdata))
            n_components=64
            pca_test = PCA(n_components)
            testdata = pca_test.fit_transform(testdata)
            
            predicted_y = nn.predict(np.array(testdata), w, b, len(nn_structure))
            print "Prediction is: ", predicted_y
            print len(nn_structure)
            
        else:
            num_classes = 2
            
            with open("../mem/weight_catdog.pkl", 'r') as saved:
                w = pkl.load(saved)
            with open("../mem/bias_catdog.pkl", 'r') as saved:
                b = pkl.load(saved)
            with open("../mem/configuration_catdog.pkl", 'r') as saved:
                nn_structure = pkl.load(saved)
            
            files = os.listdir(test_filename)
            for f in files:
            #print "Reading File", f
                file_path = os.path.join(test_filename, f)
                img = PIL.Image.open(file_path).convert("L")
                matrix = np.array(img)
                shape = matrix.shape
                reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ]) 
                testdata.append(reshaped_matrix)
                e = float(files.index(f)+1)/len(files)
            sys.stdout.write('{}% File Reading Completed .....\r'.format(int(e*100)))
            test_scaler = StandardScaler()
            testdata = test_scaler.fit_transform(np.array(testdata))
            n_components=64
            pca_test = PCA(n_components)
            testdata = pca_test.fit_transform(testdata)
            predicted_y = nn.predict(np.array(testdata), w, b, len(nn_structure))
            predicted = ['cat' if x==0 else 'dog' for x in predicted_y]
            print "Prediction is:", predicted
        
    else:
        train_filename = train_data
        test_filename = test_data
        raw_data, raw_labels, num_classes = nn.read_data(train_filename)
        #nn_structure = configuration
        testdata= list()
        if dataset == 'MNIST':
            #configuration = [64]+configuration+[10]
            #print configuration
            nn_structure = [64]+configuration+[10]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(raw_data)
            print "Scaling Completed ....."
            n_components=64
            pca = PCA(n_components)
            data = pca.fit_transform(scaled_data)
            print "Dimensionaly Reduced ....."
            labels = np.array(list(map(int, raw_labels)))
            onehot_labels = nn.one_hot(labels, num_classes)
            w, b, cost_list = nn.train(nn_structure, data, onehot_labels)
            
            with open("../mem/weight_mnist.pkl", 'w') as saver:
                pkl.dump(w, saver)
            
            with open("../mem/bias_mnist.pkl", 'w') as saver:
                pkl.dump(b, saver)
            
            with open("../mem/configuration_mnist.pkl", 'w') as saver:
                pkl.dump(nn_structure, saver)
                
            
            #num_classes = 10
            files = os.listdir(test_filename)
            for f in files:
            #print "Reading File", f
                file_path = os.path.join(test_filename, f)
                img = PIL.Image.open(file_path).convert("L")
                matrix = np.array(img)
                shape = matrix.shape
                reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ]) 
                testdata.append(reshaped_matrix)
                e = float(files.index(f)+1)/len(files)
            sys.stdout.write('{}% File Reading Completed .....\r'.format(int(e*100)))
            test_scaler = StandardScaler()
            testdata = test_scaler.fit_transform(np.array(testdata))
            n_components=64
            pca_test = PCA(n_components)
            testdata = pca_test.fit_transform(testdata)
            predicted_y = nn.predict(np.array(testdata), w, b, len(nn_structure))
            print "Prediction is:", predicted_y
        
        else:
            nn_structure = [64]+configuration+[2]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(raw_data)
            print "Scaling Completed ....."
            n_components=64
            pca = PCA(n_components)
            data = pca.fit_transform(scaled_data)
            print "Dimensionaly Reduced ....."
            print "Cat is class 0 and Dog is class 1 ....."
            labels = np.array([0 if x=='cat' else 1 for x in raw_labels])
            onehot_labels = nn.one_hot(labels, num_classes)
            w, b, cost_list = nn.train(nn_structure, data, onehot_labels)
            
            with open("../mem/weight_catdog.pkl", 'w') as saver:
                pkl.dump(w, saver)
            
            with open("../mem/bias_catdog.pkl", 'w') as saver:
                pkl.dump(b, saver)
            
            with open("../mem/configuration_catdog.pkl", 'w') as saver:
                pkl.dump(nn_structure, saver)
            
            
            files = os.listdir(test_filename)
            for f in files:
            #print "Reading File", f
                file_path = os.path.join(test_filename, f)
                img = PIL.Image.open(file_path).convert("L")
                matrix = np.array(img)
                shape = matrix.shape
                reshaped_matrix = np.reshape(matrix, [shape[0]*shape[1], ]) 
                testdata.append(reshaped_matrix)
                e = float(files.index(f)+1)/len(files)
            sys.stdout.write('{}% File Reading Completed .....\r'.format(int(e*100)))
            test_scaler = StandardScaler()
            testdata = test_scaler.fit_transform(np.array(testdata))
            n_components=64
            pca_test = PCA(n_components)
            testdata = pca_test.fit_transform(testdata)
            predicted_y = nn.predict(np.array(testdata), w, b, len(nn_structure))
            predicted = ['cat' if x==0 else 'dog' for x in predicted_y]
            print "Prediction is:", predicted
            

    
    
if __name__ == '__main__':
    main()