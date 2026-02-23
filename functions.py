import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from ann_input import *

dense=ks.layers.Dense

def build_model(params_train,pk_train,params_test,pk_test,epochs,
        batch,validation,thres_acc,layers,neurons,dropout,dropout_rate,lr,activ):
    model=ks.models.Sequential()
    opt=ks.optimizers.Adam(learning_rate=lr)
    es=EarlyStopping(monitor='loss',baseline=None,patience=20,verbose=0,min_delta=0.0001,mode='min',)
    mc=ModelCheckpoint('cii_bestmodel',monitor='loss',save_best_only=True,verbose=1)
    model.add(dense(2,input_dim=2,activation='relu'))
    #model.add(dense(28,activation='elu'))
    for i in range(layers):
        model.add(dense(neurons,activation=activ))
    if dropout==True:
        model.add(ks.layers.Dropout(rate=dropout_rate))
    model.add(dense(len(pk_test[0]),activation='linear'))
    model.compile(loss='mse',optimizer=opt,metrics=target_mod)
    history=model.fit(params_train,pk_train,validation_split=validation,epochs=epochs,verbose=0,callbacks=[es,mc],batch_size=batch, shuffle=False)
    np.save("cii_history",history.history)
    ks.models.save_model(model,'cii_model.h5')
    train_acc=model.evaluate(params_train,pk_train,verbose=0)
    test_acc=model.evaluate(params_test,pk_test,verbose=0)
    print("Training Accuracy is: ",train_acc[1]*100,"\n Testing accuracy is: ", test_acc[1]*100)
    print("Training loss is: ",train_acc[0]*100,"\n Testing loss is: ",test_acc[0]*100)
    # make class predictions with the model
    predictions = model.predict(params_test)

    return predictions

def save_to_file(data,label):
    if label=='predictions':
        with open("predictions","w") as fn_p:
            for el in data:
                for n in el:
                    fn_p.write(str(n)+"\t")
                fn_p.write("\n")
            fn_p.truncate()
        print("\n Predictions saved to file! \n")
    elif label=='pk_test':
        fn_pk=open("pk_test",'w')
        for el in data:
            for n in el:
                fn_pk.write(str(n)+"\t")
            fn_pk.write("\n")
        fn_pk.truncate()
        fn_pk.close()
        print("\n Power-spectrum test set saved to file. \n")
    elif label=='params_test':
        fn_params=open("params_test",'w')
        for el in data:
            for n in el:
                fn_params.write(str(n)+"\t")
            fn_params.write("\n")
        fn_params.truncate()
        fn_params.close()
        print("\n Parameters test set saved to file. \n")

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
