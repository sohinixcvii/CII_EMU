#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.model_selection as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import functions as fn
import math
pi=math.pi

# In[2]:


#loading dataset
path = 'data/'
npk=np.loadtxt('data/Npk.txt',usecols=(2,3,4,5,6,7))
k = np.loadtxt(path+'k.txt')
nbins = np.loadtxt(path+'nbins.txt')
params=np.loadtxt('data/params_t')


dpk=np.empty(np.shape(npk))
for i in range(len(npk)):
    for j in range(len(npk[0])):
        dpk[i,j]=((k[j]**3)*npk[i,j])/(2*pi**2)
pk=np.log(npk)/10


# In[3]:


#params.shape,npk.shape,nbins.shape,k.shape,pk.shape


# In[4]:


## getting specifics and storing as appropriate data types
epochs=input("Enter number of epochs: ")    #number of epochs
batch=input("Enter batch size: ")           #batch size
thres_acc=input("Threshold accuracy: ")     #threshold accuracy for early stopping

epochs=int(epochs)
batch=int(batch)
thres_acc=float(thres_acc)


# In[5]:


#splitting data set into training set and test set
params_train,params_test,pk_train,pk_test = sm.train_test_split(params,pk, test_size=0.1, random_state=10,shuffle=False)
fn.save_to_file(pk_test,'pk_test')
fn.save_to_file(params_test,'params_test')


#building and training the model
# %time
pred=fn.build_model(params_train,pk_train,params_test,pk_test,epochs,batch,0.2,thres_acc)


# In[9]:


pk_test=np.exp(10*pk_test)
pred=np.exp(10*pred)


# In[12]:


err=(pred-pk_test)/pk_test
# print(np.mean(err)*100)
print("Mean percentage error: ", 100*np.mean(err))


# In[ ]:


fn.save_to_file(pred,'predictions')

