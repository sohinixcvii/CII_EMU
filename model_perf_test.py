import numpy as np
from tensorflow import keras as ks

from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
import functions as fn

params=np.loadtxt('data/params_t')
dpk=np.loadtxt('data/cii_dpk_silva')
# dpk=np.log(dpk)/15

model = ks.models.load_model('cii_model_silva.h5')
model_th=[]
for el in tqdm(params):
    par = np.reshape(el, (1, 2))
    cii=model.predict(par,verbose=0)[0]
    cii=np.exp(15*cii)
    model_th.append(np.array(cii))

print("R2 is: ",fn.r2(dpk,model_th))
print("RMSE is: ",fn.rmse(dpk,model_th))
print("NRMSE is: ",fn.nrmse(dpk,model_th))
e=(model_th-dpk)/dpk
print("Average fractional error in prediction: ",np.mean(e))

params_test=np.loadtxt('params_test_silva')
pk_test=np.loadtxt('pk_test_silva')
pred_test=np.loadtxt('predictions_silva')
print("Test R2: ",fn.r2(pk_test,pred_test))
print("Test NRMSE: ",fn.nrmse(pk_test,pred_test))

params_train=np.loadtxt("params_train_silva")
pk_train=np.loadtxt('pk_train_silva')
model_th=[]
for el in tqdm(params_train):
    par = np.reshape(el, (1, 2))
    cii=model.predict(par,verbose=0)[0]
    cii=np.exp(15*cii)
    model_th.append(np.array(cii))

print("Train r2: ",fn.r2(pk_train,model_th))
print("Train NRMSE: ",fn.nrmse(pk_train,model_th))
e=(model_th-pk_train)/pk_train
print("Mean error in prediction: ",np.mean(e))
