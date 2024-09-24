'''Import Libraries'''
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''Data Processing'''
df=pd.read_csv('Jiaxipera_TT1116Y.csv')
X=df.drop(['Capacity'],axis=1).values
y=df['Capacity'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

'''Neural Network'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
model=Sequential()
model.add(Dense(3,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=62,epochs=1000)

losses=pd.DataFrame(model.history.history)
losses.plot(figsize=(12,8))

'''Evaluation Test Data'''
from sklearn import metrics
def measure_accuracy(original,predicted,train=True):
    mae=metrics.mean_absolute_error(original,predicted)
    mse=metrics.mean_squared_error(original,predicted)
    rmse=np.sqrt(metrics.mean_squared_error(original,predicted))
    r2_square=metrics.r2_score(original,predicted)
    evs=metrics.explained_variance_score(original,predicted)
    if train:
        print("Training Result:")
        print("------------------")
        print('MAE:{0:0.3f}'.format(mae))
        print('MSE:{0:0.3f}'.format(mse))
        print('RMSE:{0:0.3f}'.format(rmse))
        print('Explained Variance Score: {0:0.3f}'.format(evs))
        print('R2 Square:{0:0.3f}'.format(r2_square))
    elif not train: 
        print("Testing Result:")
        print("------------------")
        print('MAE:{0:0.3f}'.format(mae))
        print('MSE:{0:0.3f}'.format(mse))
        print('RMSE:{0:0.3f}'.format(rmse))
        print('Explained Variance Score: {0:0.3f}'.format(evs))
        print('R2 Square:{0:0.3f}'.format(r2_square))
        
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)
#plt.figure(figsize=(8,6))
#plt.scatter(y_test,y_test_pred)
#errors=y_test.reshape(36,1)-y_test_pred
#plt.figure(figsize=(8,6))

#sns.displot(errors)
#plt.plot(y_test,y_test,'r')

measure_accuracy(y_train,y_train_pred,train=True)
measure_accuracy(y_test,y_test_pred,train=False)

'''Input Valu'''
print('***************************')
input=np.array([49,-30,3000]).reshape(1,3)
input_scaler=scaler.transform(input)
input_predict=model.predict(input_scaler)
cap=input_predict[0,0]
print('Input Condition')
print('----------------')
print('Condenser Temperature : %0.3f'%input[0,0])
print('Evaporator Temperature : %0.3f'%input[0,1])
print('Compressor RPM : %0.3f'%input[0,2])
print('Capacity (w): %0.3f'%cap)

'''Mass Flow Rate Calculation'''

Tsuction=40+273
Tcond=input[0,0]+273
Teva=input[0,1]+273
Pcond=CP.PropsSI('P','T',Tcond,'Q',0,'IsoButane')
Peva=CP.PropsSI('P','T',Teva,'Q',0,'IsoButane')
Ha=(CP.PropsSI('H','P',Peva,'T',Tsuction,'IsoButane'))/1e3
Hd=(CP.PropsSI('H','P',Pcond,'T',Tsuction,'IsoButane'))/1e3
del_H=Ha-Hd
# Mass flow rate (MRF)
MRF=(cap*3.6)/del_H
print('Mass Flow Rate: %0.3f'%MRF)