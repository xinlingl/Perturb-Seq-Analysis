#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:55:41 2019

"""
import numpy as np
from sklearn.cluster import KMeans
import csv
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures as PF
from time import time
import seaborn as sns



def devide(X,Y,num_val = 50):
    X_val = np.zeros((num_val, len(X[0])))
    Y_val = np.zeros((num_val, len(Y[0])))
    for i in range(num_val):
        idx = np.random.randint(len(X))
        X_val[i] = X[idx]
        Y_val[i] = Y[idx]
        X = np.concatenate((X[: idx], X[idx + 1:]))
        Y = np.concatenate((Y[: idx], Y[idx + 1:]))
    return X, X_val, Y, Y_val



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
                gene_ids[g_idx] = genes[j]
                g_idx += 1
        else:
            line=lines[i]
            line = line.split(',')
            cell_ids[c_idx] = line[0]
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



def kmeans_result(E_matrix):
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(E_matrix)
    preds = kmeans.predict(E_matrix)
    f = open('kmeans_labels.txt','w')
    for p in preds:
        f.write(str(p) + '\n')
    f.close()


    
def bayes_cov_col(Y,X,cols,lm):
    """
    @Y    = Expression matrix, cells x x genes, expecting pandas dataframe
    @X    = Covariate matrix, cells x covariates, expecting pandas dataframe
    @cols = The subset of columns that the EM should be performed over, expecting list
    @lm   = linear model object
    """

    #EM iterateit
    Yhat=pd.DataFrame(lm.predict(X))
    Yhat.index=Y.index
    Yhat.columns=Y.columns
    SSE_all=np.square(Y.subtract(Yhat))
    X_adjust=X.copy()


    df_SSE   = []
    df_logit = []

    for curcov in cols:

        curcells=X[X[curcov]>0].index

        if len(curcells)>2:

            X_notcur=X.copy()
            X_notcur[curcov]=[0]*len(X_notcur)

            X_sub=X_notcur.loc[curcells]

            Y_sub=Y.loc[curcells]

            GENE_var=2.0*Y_sub.var(axis=0)
            vargenes=GENE_var[GENE_var>0].index

            Yhat_notcur=pd.DataFrame(lm.predict(X_sub))
            Yhat_notcur.index=Y_sub.index
            Yhat_notcur.columns=Y_sub.columns

            SSE_notcur=np.square(Y_sub.subtract(Yhat_notcur))
            SSE=SSE_all.loc[curcells].subtract(SSE_notcur)
            SSE_sum=SSE.sum(axis=1)

            SSE_transform=SSE.div(GENE_var+0.5)[vargenes].sum(axis=1)
            logitify=np.divide(1.0,1.0+np.exp(SSE_transform))#sum))

            df_SSE.append(SSE_sum)
            df_logit.append(logitify)

            X_adjust[curcov].loc[curcells]=logitify

    return X_adjust



def LR(X,Y, savename,norm = 0):
    if norm == 0:
        lm=sklearn.linear_model.LinearRegression()
    elif norm == 1:
        lm=sklearn.linear_model.Lasso(alpha = 0.01)
    elif norm == 2:
        lm = sklearn.linear_model.Ridge(alpha = 0.01)
          
    lm.fit(X,Y)
    y_hat = lm.predict(X)
    np.save('norm' + str(norm) + '_' +savename , lm.coef_)
    
    return lm.coef_, lm, y_hat



def plot_PCA(title,Y, with_label = True):
    pca = PCA(n_components = 2)
    pca_coeff = pca.fit_transform(Y)
    if title == 'expression':
        plt.title(title)
    else:
        plt.title(title + ', with label = '+ str(with_label))
    
    plt.scatter(pca_coeff[:,0],pca_coeff[:,1], s = 10)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(title + '_' + str(with_label) + '.png')
    plt.show()


def validate(X_tr, Y_tr, X_val, Y_val,norm = 0, degree = 1):
    if degree > 1:
        pf = PF(degree)
        X_tr = pf.fit_transform(X_tr)
        pf = PF(degree)
        X_val = pf.fit_transform(X_val)
    start = time()
    print(X_tr.shape, X_val.shape)
    _,lm,tr_pred = LR(X_tr,Y_tr, 'None.npy',norm = norm)
    end = time()
    print('Time cost is',end - start)
     
    val_pred = lm.predict(X_val)
    mse_tr = np.sqrt(sum(sum((tr_pred - Y_tr) * (tr_pred - Y_tr)/ len(Y_tr))))
    print('MSE for training is', mse_tr)
    mse_val = np.sqrt(sum(sum((val_pred - Y_val) * (val_pred - Y_val)/ len(Y_val))))
    print('MSE for validation is', mse_val)

    
def get_perturb_id():
    f = open('GSM2396861_k562_ccycle_cbc_gbc_dict.csv')
    lines = f.readlines()
    perturbs = []
    for line in lines:
        line = line.split(',')
        perturbs.append(line[0])
    return perturbs


def gene_mapping(coeff, gene_ids, perturbs):
    coeff = coeff.T
    n = len(coeff)
    m = len(coeff[0])
    result = []
    pairs= []
    vals = []
    geneids = {}
    for i in range(32):
        max_idx = 0
        max_val = -1
        for j in range(m):
            if coeff[i][j] > max_val:
                max_idx = j
                max_val = abs(coeff[i][j])
        geneId = gene_ids[max_idx]#.split('_')[1]
         
        perturb_id = perturbs[i]
        pairs.append(perturb_id + '+' + geneId)
        if geneId in geneids:
            geneids[geneId]  += 1
        else:
            geneids[geneId] = 1
        vals.append(max_val)
        
    plt.xticks(rotation = 270)
    plt.tight_layout()
    plt.bar(pairs, vals)
    plt.xlabel('perturb/gene pair')
    plt.ylabel('absolute coefficient')
    plt.title('Finding the most perturbed genes for each perturbation')
    plt.savefig('largest_abs_val.png')
    plt.show()
    genes_nums= []
    nums = []
    for gene in geneids.keys():
        genes_nums.append(gene + ', counts: ' + str(geneids[gene]))
        nums.append(geneids[gene])
    
    plt.clf()
    plt.pie(nums,autopct = '%1.1f%%',labels = genes_nums)
    plt.title('Counts of most perturbed genes')
    return result


def max_expression(Y, gene_ids):
    sum_expression = np.var(Y,axis = 0)
    sum_expression = list(sum_expression)
    print(len(sum_expression))
    for i in range(len(sum_expression)):
        sum_expression[i] = [sum_expression[i],gene_ids[i]]
    sum_expression.sort(reverse = True)
    print(sum_expression[:10])
    
X = read_preturb_seq_file()
Y,cell_ids, gene_ids = read_expression_file()
coeff = np.load('coeff_cellLabelnorm0.npy')
perturbs = get_perturb_id()
maps = gene_mapping(coeff, gene_ids, perturbs)
max_expression(Y,gene_ids)
