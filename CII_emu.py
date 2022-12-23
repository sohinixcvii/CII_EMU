import warnings
warnings.filterwarnings('ignore')
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
from ann_input import *
import math
pi=math.pi

npk=np.loadtxt(npk_p,usecols=(2,3,4,5,6,7))
k = np.loadtxt(k_p)
nbins = np.loadtxt(n_p)
params=np.loadtxt(path+'params_t')


dpk=np.empty(np.shape(npk))
for i in range(len(npk)):
    for j in range(len(npk[0])):
        dpk[i,j]=((k[j]**3)*npk[i,j])/(2*pi**2)
pk=np.log(npk)/10

#splitting data set into training set and test set
params_train,params_test,pk_train,pk_test = sm.train_test_split(params,pk, test_size=0.1, random_state=10,shuffle=False)
fn.save_to_file(pk_test,'pk_test')
fn.save_to_file(params_test,'params_test')


#building and training the model
pred=fn.build_model(params_train,pk_train,params_test,pk_test,epochs,batch,val_frac,thres_acc,layers,neurons,dropout,dropout_rate,learning_rate,activation)

#inverse transforming
pk_test=np.exp(10*pk_test)
pred=np.exp(10*pred)


#error calculation
err=(pred-pk_test)/pk_test

print("Mean percentage error: ", 100*np.mean(err))

#saving predictions to file
fn.save_to_file(pred,'predictions')
