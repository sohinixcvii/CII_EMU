import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from ann_input import *
from sklearn.metrics import *
model=ks.models.Sequential()
dense=ks.layers.Dense

def build_model(params_train,pk_train,params_test,pk_test,epochs,
        batch,validation,thres_acc,layers,neurons,dropout,dropout_rate,lr,activ):
    # opt=ks.optimizers.Adam(learning_rate=lr)
    opt=tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    es=EarlyStopping(monitor='loss',baseline=None,patience=50,verbose=0,min_delta=0.0001,mode='min',)
    mc=ModelCheckpoint('cii_bestmodel',monitor='loss',save_best_only=True,verbose=1)
    model.add(dense(2,input_dim=2,activation='relu'))
    for i in range(layers):
        model.add(dense(neurons,activation=activ))
    if dropout==True:
        model.add(ks.layers.Dropout(rate=dropout_rate))
    model.add(dense(len(pk_test[0]),activation='linear'))
    model.compile(loss='mse',optimizer=opt,metrics=target_mod)
    history=model.fit(params_train,pk_train,validation_split=validation,epochs=epochs,verbose=0,callbacks=[es,mc],batch_size=batch, shuffle=True)
    np.save("cii_history_silva",history.history)
    ks.models.save_model(model,'cii_model_silva.h5')
    train_acc=model.evaluate(params_train,pk_train,verbose=0)
    test_acc=model.evaluate(params_test,pk_test,verbose=0)
    print("Training Accuracy is: ",train_acc[1]*100,"\n Testing accuracy is: ", test_acc[1]*100)
    print("Training loss is: ",train_acc[0]*100,"\n Testing loss is: ",test_acc[0]*100)
    # make class predictions with the model
    predictions = model.predict(params_test)

    return predictions

def history_plot(history):
    plt.style.use('classic')
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.2
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 1.2
    mpl.rcParams['patch.linewidth'] = 1.8
    mpl.rcParams['axes.linewidth'] = 1.8

    plt.figure(figsize=(13, 4), facecolor='w', edgecolor='w')
    plt.suptitle(r'$\rm Accuracy~\&~Loss$', size=23)
    plt.subplot(1, 2, 1)
    plt.grid(True, lw=1)
    plt.ylim(0,1.2)
    plt.xlim(0, len(history.item().get('val_loss'))+10)
    plt.plot(history.item().get('acc'), color='cyan', lw=2)
    plt.plot(history.item().get('val_acc'), color='magenta',lw=2)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.ylabel(r'$\rm Accuracy$', size=18, )
    plt.xlabel(r'$\rm Epochs$', size=18, )

    plt.subplot(1, 2, 2)
    plt.grid(True, lw=1)
    plt.plot(history.item().get('loss'), color='cyan', lw=2)
    plt.plot(history.item().get('val_loss'), color='magenta', lw=2)
    plt.ylabel(r'$\rm Loss$', size=18, color='k', )
    plt.xticks(size=16, color='k', )
    plt.yticks(size=16, color='k')
    #plt.ylim(0, np.max(history.item().get('val_loss'))+10)
    #plt.xlim(0,len(history.item().get('val_loss'))+10)
    plt.xlabel(r'$\rm Epochs$', size=18, color='k')

''' tranformation and inverse tranformation functions used in MCMC files   
def tr(p):
    y=((p-[mh,0])/[ms,1])/4
    return y

def inv_tr(p):
    x=4*np.multiply([ms,1],p)+[mh,0]
    return x

'''

#performance metrics
def r2(y_true, y_pred):
    """
    R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: R2
     """
    return r2_score(y_true, y_pred)

def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Square Error.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: normalized root mean square error
    """
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())

def count_matches_2d(array_a, array_b):
    """
    Counts the number of elements that match in two 2-dimensional arrays.

    Args:
        array_a (list): The first 2-dimensional array.
        array_b (list): The second 2-dimensional array.

    Returns:
        int: The number of matches.
    """

    match_count = 0
    array_a_length = len(array_a)
    array_b_length = len(array_b)

    if array_a_length != array_b_length:
        return match_count

    for i in range(array_a_length):
        if len(array_a[i]) != len(array_b[i]):
            return match_count

        for j in range(len(array_a[i])):
            if array_a[i][j] == array_b[i][j]:
                match_count += 1
    return match_count
        
def sign_pred_acc(data,pred):
    sign_d=np.empty(np.shape(data))
    sign_p=np.empty(np.shape(pred))
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i,j]>0:
                sign_d[i,j]=1
            else:
                sign_d[i,j]=0
            if pred[i,j]>0:
                sign_p[i,j]=1
            else:
                sign_p[i,j]=0
    return count_matches_2d(sign_d,sign_p)
