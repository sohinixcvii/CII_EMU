'''Input file for ANN based emulator. Initializes variables. Change this file to implement global changes'''

#Data paths
path= 'data/' #path to input and output data directory
npk_p=path+'Npk.txt'
k_p=path+'k.txt'
n_p=path+'nbins.txt'
params_p=path+'params_LH.txt'
dpk_p=path+'cii_dpk'

#General specs
test_frac=0.1
val_frac=0.2
train_epochs=2
target='accuracy' #target of optimization, ['accuracy','loss']

#Hyperparameter optimization
kmode=0 #int value, kmode index/index array
mod_name= 'test_mod' #model name to save files with
tuner_choice= 'BayesOpt' #Which tuner to use. 'HyperBand' or 'BayesOpt'
layer_range=[1,32] #search space for number of layers [min,max]
neuron_range=[16,144] #search space for number of neurons [min,max]
lr_range=[1e-8,1e-4] #search space for learning rate [min,max]
drop_rate_range=[0.1,0.5] #search space for dropout rate [min,max]
activ_range=['relu','elu','softmax','exponential','linear'] #list of activation functions to consider in search

#hyper parameters (from HPO results):
layers= 32
neurons= 100
dropout= True
epochs= 100
batch= 4
thres_acc= 0.99
activation= 'elu'
dropout_rate= 0.005
learning_rate= 3.3466e-05