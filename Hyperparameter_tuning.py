#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras_tuner
from tensorflow import keras
import sklearn.model_selection as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import losses
import math
pi=math.pi

def build_model(hp):
    model=keras.Sequential()
    model.add(keras.layers.Dense(2,activation='relu'))
    for i in range(hp.Int("num_layers",1,32)):
        model.add(
            keras.layers.Dense(
                #tune number of neurons
                units=hp.Int("units",min_value=16,max_value=512,step=4),
                #tune activation function to use
                activation=hp.Choice("activation",['relu','elu','softmax','exponential','linear'])
                    #,'softmin','sigmoid','softplus','softsign','selu']
                )
            )
    #Tune whether to use dropout
    dropout_rate=hp.Float("dropout rate",min_value=0.005,max_value=0.5)
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(6,activation='relu'))
    #Defining optimizer learning rate as a hyperparameters
    learning_rate=hp.Float("lr",min_value=1e-8,max_value=1e-4,sampling='log')
    model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="mse",
                metrics=["accuracy"]
                )
    return model


# In[3]:


build_model(keras_tuner.HyperParameters())
kmode=input("Enter k mode axis: ")
mod_name="k"+kmode
kmode=[int(kmode)]

tuner=keras_tuner.Hyperband(
    hypermodel=build_model,
    objective='accuracy',
    max_epochs=10,
    factor=3,
    hyperband_iterations=3,
    # seed=None,
    # hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    # **kwargs
)
# tuner = keras_tuner.BayesianOptimization(
#             hypermodel=build_model,
#             objective='accuracy',
#             max_trials=500,
#             executions_per_trial=3,
#             directory='tuning_results',
#             project_name=mod_name
#             )

npk=np.loadtxt('data/Npk.txt',usecols=(2,3,4,5,6,7))
k=np.loadtxt('data/k.txt')
n=np.loadtxt('data/nbins.txt')
print("Shape of k: ",np.shape(k),"Shape of nbins: ", np.shape(n),"Shape of npk: ",np.shape(npk))
path = 'data/'
params = np.loadtxt(path+'params_LH.txt')                        #params

dpk=np.empty(np.shape(npk))
for i in range(len(npk)):
    for j in range(len(npk[0])):
        dpk[i,j]=((k[j]**3)*npk[i,j])/(2*pi**2)
f=open('data/cii_dpk','w+')
for el in dpk:
    for j in el:
        f.write(str(j)+'\n')
    f.write('\t')
f.truncate()
f.close()
print("Minimum dpk before scaling: ",np.min(np.log(dpk))," Maximum dpk before scaling: ",np.max(np.log(dpk)))

pk=np.log(dpk)/10

print("Minimum dpk after scaling: ",np.min(np.log(pk))," Maximum dpk after scaling: ",np.max(np.log(pk)))

params_traino,params_test,pk_traino,pk_test = sm.train_test_split(params,pk, test_size=0.1,shuffle=False)
params_train,params_val,pk_train,pk_val = sm.train_test_split(params_traino,pk_traino, test_size=0.2,shuffle=False)

scaler_1=MinMaxScaler(feature_range=(0, 1))
scaler_1.fit(params_train)
params_train=scaler_1.transform(params_train)
scaler_4=MinMaxScaler(feature_range=(0, 1))
scaler_4.fit(params_test)
params_test=scaler_4.transform(params_test)

tuner.search(params_train, pk_train, epochs=50, validation_data=(params_val, pk_val))
tuner.results_summary()

# best_model = tuner.get_best_models(num_models=2)[0]
#
# best_model.build(input_shape=(3,3))
# best_model.summary()
