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
test_data_2008 = np.loadtxt('data/test_2008.csv', skiprows=1, delimiter=',')
test_data_2012 = np.loadtxt('data/test_2012.csv', skiprows=1, delimiter=',')


# In[3]:


X_train_2008 = train_data_2008[:,3:-1]
Y_train_2008 = train_data_2008[:,-1]
X_test_2008 = test_data_2008[:,3:]
X_test_2012 = test_data_2012[:,3:]


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


# In[ ]:


#normalize each column of the data
X_train_2008 = normalize_data_column(X_train_2008)
X_test_2008 = normalize_data_column(X_test_2008)
X_test_2012 = normalize_data_column(X_test_2012)


# In[ ]:


#split the training data into training and validation dataset
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_2008, Y_train_2008, test_size=0.3)


# ## Build and train the model

# In[ ]:


model_parameter = "adaboost_lr0p3_lossSquare_nest1000_minsamplesplit2_maxdepth5_sqrt_random_skipdatacolumn012"


# In[ ]:


#estimator for adaboost
ada_tree_estimator = DecisionTreeRegressor(min_samples_split=2, max_depth=5, max_features='sqrt', splitter='random')
#adaboost regressor
ab = AdaBoostRegressor(ada_tree_estimator, learning_rate=0.03, loss='square', n_estimators=1000)
#fit
ab.fit(X_train, Y_train)


# ## Validation

# In[ ]:


## calculate the AUC - area under the curve
Y_train_predict = ab.predict(X_train)
Y_valid_predict = ab.predict(X_valid)
AUC_train = roc_auc_score(Y_train, Y_train_predict)
print(AUC_train)
AUC_valid = roc_auc_score(Y_valid, Y_valid_predict)
print(AUC_valid)


# In[ ]:


## draw the ROC curve
fpr_train, tpr_train, _ = roc_curve(Y_train, Y_train_predict)
fpr_valid, tpr_valid, _ = roc_curve(Y_valid, Y_valid_predict)
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='blue',
         lw=lw, label='training set, AUC = %.3f'%AUC_train)
plt.plot(fpr_valid, tpr_valid, color='darkorange',
         lw=lw, label='validation set, AUC = %.3f'%AUC_valid)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.legend(loc=0, shadow=True)
plt.title(r'ROC curve')
plt.savefig('plots/ROC_'+model_parameter+'.pdf')


# ## Make the prediction!

# In[ ]:


predict_test_2008 = ab.predict(X_test_2008)
predict_test_2012 = ab.predict(X_test_2012)


# ## Write to submission file

# In[ ]:


ids_2008 = np.arange(len(X_test_2008))
ids_2012 = np.arange(len(X_test_2012))
target_2008 = np.copy(predict_test_2008)
target_2008 = np.copy(predict_test_2008)
target_2012 = np.copy(predict_test_2012)


# In[ ]:


subm_2008 = np.stack([ids_2008,target_2008], axis=1)
subm_2012 = np.stack([ids_2012,target_2012], axis=1)
#let's name the submission file in this way
#submission_2008or2012_date_version_algorithm_person.csv
np.savetxt('submission/submission_2008_02102019_v2_'+model_parameter+'_Zhicai.csv', subm_2008, fmt='%d,%.6f', header='id,target', comments='')
np.savetxt('submission/submission_2012_02102019_v2_'+model_parameter+'_Zhicai.csv', subm_2012, fmt='%d,%.6f', header='id,target', comments='')


# In[ ]:





# In[ ]:




