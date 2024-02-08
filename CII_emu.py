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
import time
pi=math.pi

st=time.time()

dpk=np.loadtxt(dpk_p)
k = np.loadtxt(k_p)
nbins = np.loadtxt(n_p)
params=np.loadtxt(path+'params_t')
# params[:,1]=[0.395]*len(params[:,0])
# pk=np.log(dpk)/10 #for Ma et al 2018 SFR-Mhalo +silva et al LCII-SFR case
pk=np.log(dpk)/15 #for Silva et al case (V2)

print("Params shape: ",np.shape(params)," pk shape: ",np.shape(pk))

#splitting data set into training set and test set
params_train,params_test,pk_train,pk_test = sm.train_test_split(params,pk, test_size=0.1, random_state=10,shuffle=True)
with open('pk_test_silva', 'w+') as pk_fn:
    np.savetxt(pk_fn, np.exp(15*pk_test),fmt='%.5e')
with open('pk_train_silva','w+') as pk_fn:
    np.savetxt(pk_fn, np.exp(15*pk_train),fmt='%.5e')  

with open('params_test_silva', 'w+') as params_fn:
    np.savetxt(params_fn, params_test)

with open('params_train_silva','w+') as params_fn:
    np.savetxt(params_fn, params_train)

#building and training the model
pred=fn.build_model(params_train,pk_train,params_test,pk_test,epochs,batch,val_frac,thres_acc,layers,neurons,dropout,dropout_rate,learning_rate,activation)

#inverse transforming
pk_test=np.exp(15*pk_test)
pred=np.exp(15*pred)

#error calculation
err=(pred-pk_test)/pk_test

print(pk_test[0],pred[0])
print("Mean percentage error: ", 100*np.mean(err))
print("R2 score: ",fn.r2(pk_test,pred))
print("NRMSE is: ",fn.nrmse(pk_test,pred))
#saving predictions to file
with open('predictions_silva','w+') as pred_fn:
    np.savetxt(pred_fn,pred,fmt='%.5e')

tt=time.time()-st
print("Total time taken: ",time.strftime("%H-%M-%S",time.gmtime(tt)))
