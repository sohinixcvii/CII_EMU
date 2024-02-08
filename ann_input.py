'''Input file for ANN based emulator. Initializes variables. Change this file to implement global changes'''
from tensorflow import keras
import keras_tuner
#Data paths
path= 'data/' #path to input and output data directory
npk_p=path+'cii_power_silva'
k_p=path+'k.txt'
n_p=path+'nbins.txt'
params_p=path+'params_LH.txt'
dpk_p=path+'cii_dpk_silva'

#General specs
test_frac=0.15
val_frac=0.2
train_epochs=20
num_trials=100
#target='accuracy'
target='loss'
#target=keras_tuner.Objective('root_mean_squared_error',direction='min') #target of optimization, ['accuracy','loss']

#Hyperparameter optimization
kmode=0 #int value, kmode index/index array
mod_name= 'loss_mod' #model name to save files with
tuner_choice= 'BayesOpt' #Which tuner to use. 'HyperBand' or 'BayesOpt'
layer_range=[1,64] #search space for number of layers [min,max]
neuron_range=[16,256] #search space for number of neurons [min,max]
lr_range=[1e-8,1e-2] #search space for learning rate [min,max]
drop_rate_range=[0.1,0.5] #search space for dropout rate [min,max]
activ_range=['relu','elu','linear'] #list of activation functions to consider in search


#hyper parameters (from HPO results) version 1:
# target_mod='mse'
# target_mod=keras.metrics.RootMeanSquaredError() #target of build_model function
# layers= 32
# neurons= 144
# dropout= True
# epochs= 1000
# batch= 4
# thres_acc= 0.99
# activation= 'relu'
# dropout_rate= 0.1
# learning_rate= 9.9999999e-05


#Hyperparameters for Silva et al case:----keep
# target_mod='mse'
target_mod=keras.metrics.RootMeanSquaredError() #target of build_model function
layers= 3
neurons=128
dropout= False
epochs= 1000
batch= 4
thres_acc= 0.99
activation= 'relu'
dropout_rate= 0.1
learning_rate=0.005178597403693026

#Hyperparameters for Silva et al case:
# target_mod=['accuracy','mse']
# target_mod=keras.metrics.RootMeanSquaredError() #target of build_model function
# layers= 43
# neurons= 84
# dropout= False
# epochs= 1000
# batch= 4
# thres_acc= 0.99
# activation= 'linear'
# dropout_rate= 0.1
# learning_rate= 0.0007596633140456775

