#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:33:04 2019

"""
from sklearn.cluster import AgglomerativeClustering as AC
import numpy as np
import matplotlib.pyplot as plt



def read_expression_file(filename = 'normalized_matrix_after_magic.csv'):
    f = open(filename,'r')
    lines = f.readlines()
    cell_ids = {}
    c_idx = 0
    gene_ids = {}
    g_idx = 0
    n = len(lines) - 1
    m = len(lines[0].split(',')) - 1
    E_matrix = np.zeros((n,m))
    for i in range(len(lines)):
        if i == 0:
            genes = lines[0]
            genes = genes.split(',')
            for j in range(1, len(genes)):
                gene_ids[genes[j]] = g_idx
                g_idx += 1
        else:
            line=lines[i]
            line = line.split(',')
            cell_ids[line[0]] = c_idx
            c_idx += 1
            for j in range(1, len(line)):
                E_matrix[i - 1][j - 1] = line[j]
    return E_matrix, cell_ids, gene_ids



def read_preturb_seq_file(fname = 'covariate_matrix.csv'):
    f= open(fname,'r')
    lines = f.readlines()
    n = len(lines)
    m = len(lines[0].split(','))
    M = np.zeros((n, m))
    for i in range(n):
        line = lines[i]
        line = line.strip('\n')
        line = line.split(',')
        for j in range(len(line)):
            M[i][j] = float(line[j])
    return M



def choose_k(X,k_range):
    X = X.T
    X = X[:32]
    print(X.shape)
    
    X_mean = sum(X)/len(X)
    chs = []
    n = len(X)
    for k in range(2, k_range):
        clf = AC(n_clusters = k, linkage = 'average')
        clf.fit(X)
        labels = clf.labels_
        
        centroids = np.zeros((k, len(X[0])))
        counts = np.zeros((k, 1))
        for i in range(n):
            for l in range(k):
                if l == labels[i]:
                    centroids[l] += X[i]
                    counts[l][0] += 1
        centroids /= counts
        W = 0
        B = 0
        for label in range(k):
            for i in range(len(X)):
                if labels[i] == label:
                    W += np.linalg.norm((X[i] - centroids[label]) , 2) ** 2
            B += counts[label][0] * (np.linalg.norm((centroids[label] - X_mean) ,2)** 2)
        up = B/(k - 1)
        down = W/(n - k)
        chs.append(up/down)
        
    plt.figure()
    plt.plot([i + 2 for i in range(len(chs))],  chs)
    plt.xlabel('k')
    plt.ylabel('ch value')
    plt.title('Choose best k')
    plt.show()

X = np.load('norm0_coeff_cellLabel.npy')
k_range = 11
choose_k(X,k_range)
