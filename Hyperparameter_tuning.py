#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings

#suppressing all the warnings
warnings.filterwarnings('ignore')

import keras_tuner
from tensorflow import keras
import sklearn.model_selection as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import losses
import math
pi=math.pi
from ann_input import *
import csv
import time

st=time.time()

def build_model(hp):
    model=keras.Sequential()
    model.add(keras.layers.Dense(2,activation='relu'))
    for i in range(hp.Int("num_layers",layer_range[0],layer_range[1])):
        model.add(
            keras.layers.Dense(
                #tune number of neurons
                units=hp.Int("units",min_value=neuron_range[0],max_value=neuron_range[1],step=4),
                #tune activation function to use
                activation=hp.Choice("activation",activ_range)
                    #,'softmin','sigmoid','softplus','softsign','selu']
                )
            )
    #Tune whether to use dropout
    #dropout_rate=hp.Float("dropout rate",min_value=drop_rate_range[0],max_value=drop_rate_range[1])
    #if hp.Boolean("dropout"):
    #    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(6,activation='linear'))
    #Defining optimizer learning rate as a hyperparameters
    learning_rate=hp.Float("lr",min_value=lr_range[0],max_value=lr_range[1],sampling='log')
    model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="mse",
                metrics="accuracy"
                )
    return model


# In[3]:
hyper=keras_tuner.HyperParameters()
print("Tuner config: \n \t Number of trials: ",num_trials,"\n \t Number of epochs per trial: ",train_epochs)
if tuner_choice=='HyperBand':
    tuner=keras_tuner.Hyperband(
        hypermodel=build_model,
        objective=target,
        max_epochs=2,
        factor=2,
        hyperband_iterations=3,
        # seed=None,
        # hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        # **kwargs
        )
elif tuner_choice=='BayesOpt':
    tuner = keras_tuner.BayesianOptimization(
                 hypermodel=build_model,
                 objective=target,
                 max_trials=num_trials,
                 executions_per_trial=3,
                 directory='tuning_results',
                 project_name=mod_name
                 )

dpk=np.loadtxt(dpk_p)
k=np.loadtxt(k_p)
n=np.loadtxt(n_p)
params=np.loadtxt(path+'params_t')         #params
print("Shape of k: ",np.shape(k),"Shape of nbins: ", np.shape(n),"Shape of dpk: ",np.shape(dpk))

#Converting to dimensionless power spectrum. Step 1 of reducing dynamic range. 

#Scaling dpk by using custom scaling formalism
pk=np.log(dpk)/15
#Splitting whole dataset into training set and test set. 90% split
params_traino,params_test,pk_traino,pk_test = sm.train_test_split(params,pk, test_size=test_frac,shuffle=True)

#Splitting training set for validation and training (required for hyperparameter optimization)
params_train,params_val,pk_train,pk_val = sm.train_test_split(params_traino,pk_traino, test_size=val_frac,shuffle=True)

#Performing the optimization and printing results. Takes at least 30 mins. 
tuner.search(params_train, pk_train, epochs=train_epochs, validation_data=(params_val, pk_val),verbose=0)
tuner.results_summary()

#res_file=open('HPO','w+')
#csv.writer(res_file, delimiter=' ').writerows(tuner.results_summary())
#res_file.close()

tt=time.time()-st

print("Total time taken: ",time.strftime("%H-%M-%S",time.gmtime(tt)))
