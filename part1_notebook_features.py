#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_curve, roc_auc_score


# ## Load the data

# In[2]:


train_data_2008 = np.loadtxt('data/train_2008.csv', skiprows=1, delimiter=',')


# In[3]:


X_train_2008 = train_data_2008[:,:-1]
Y_train_2008 = train_data_2008[:,-1]


# ## Data pre-process

# In[4]:


def normalize_data_column(x):
    '''
    normalize the input data such that it is centered around zero and has standard deviation of 1.0
    Inputs:
        x: a (N, D) shaped numpy array containing the data points.
    Outputs:
        xp: a (N, D) shaped numpy array containing the normalized data points.
    '''
    xp = np.zeros_like(x)
    
    for idx_D in range(len(x[0,:])): #normalize each column independently
        average = np.mean(x[:,idx_D])
        std_dev = np.std(x[:,idx_D])
        if std_dev > 0:
            xp[:,idx_D] = (x[:, idx_D] - average)/std_dev
        elif average != 0: #if all the elements are the same in that column, make all of them to be one
            xp[:,idx_D] = x[:, idx_D]/average
        else:
            xp[:,idx_D] = x[:, idx_D]
    
    return xp


# In[5]:


#normalize each column of the data
X_train_2008 = normalize_data_column(X_train_2008)


# In[6]:


#split the training data into training and validation dataset
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_2008, Y_train_2008, test_size=0.1, train_size=0.2)


# ## Build and train the model

# In[7]:


#estimator for adaboost
ada_tree_estimator = DecisionTreeRegressor(min_samples_split=2, max_depth=5, max_features='sqrt', splitter='random')
#adaboost regressor
ab = AdaBoostRegressor(ada_tree_estimator, learning_rate=0.03, loss='square', n_estimators=1000)
#fit
ab.fit(X_train, Y_train)


# ## Validation

# In[8]:


## calculate the AUC - area under the curve
Y_train_predict = ab.predict(X_train)
Y_valid_predict = ab.predict(X_valid)
AUC_train = roc_auc_score(Y_train, Y_train_predict)
print(AUC_train)
AUC_valid = roc_auc_score(Y_valid, Y_valid_predict)
print(AUC_valid)


# ## analyze each features

# In[9]:


infile = open('data/train_2008.csv', 'r')
firstLine = infile.readline()
nameOfFeatures = firstLine.split(',')
print(len(nameOfFeatures))


# In[16]:


f_features = open('features_analyze.txt', 'w')
f_features.write('idx, removed feature, AUC-training, AUC-validation, percent change AUC-train, percent change AUC-valid\n')
print('idx, removed feature, AUC-training, AUC-validation, percent change AUC-train, percent change AUC-valid\n')

for id_feature in range(len(nameOfFeatures)-1):    
    X_train_this = np.delete(X_train, id_feature, axis=1)
    X_valid_this = np.delete(X_valid, id_feature, axis=1)
    #estimator for adaboost
    ada_tree_estimator_this = DecisionTreeRegressor(min_samples_split=2, max_depth=5, max_features='sqrt', splitter='random')
    #adaboost regressor
    ab_this = AdaBoostRegressor(ada_tree_estimator_this, learning_rate=0.03, loss='square', n_estimators=1000)
    #fit
    ab_this.fit(X_train_this, Y_train)
    
    Y_train_predict_this = ab_this.predict(X_train_this)
    Y_valid_predict_this = ab_this.predict(X_valid_this)
    AUC_train_this = roc_auc_score(Y_train, Y_train_predict_this)
    AUC_valid_this = roc_auc_score(Y_valid, Y_valid_predict_this)

    percent_AUC_train = 100.0*(AUC_train_this-AUC_train)/AUC_train
    percent_AUC_valid = 100.0*(AUC_valid_this-AUC_valid)/AUC_valid
    print(str(id_feature)+', %s'%nameOfFeatures[id_feature]+', %.4f'%AUC_train_this + ', %.4f'%AUC_valid_this+', %.4f'%percent_AUC_train + ', %.4f'%percent_AUC_valid)
    f_features.write(str(id_feature)+', %s'%nameOfFeatures[id_feature]+', %.4f'%AUC_train_this + ', %.4f'%AUC_valid_this+', %.4f'%percent_AUC_train + ', %.4f'%percent_AUC_valid+'\n')


# In[ ]:




