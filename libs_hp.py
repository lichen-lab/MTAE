from __future__ import division, print_function, absolute_import
from collections import defaultdict, OrderedDict
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import h5py
import pickle
from scipy import stats
import keras_tuner as kt
#from keras_tuner import HyperModel

import seaborn as sns
from pathlib import Path
import shutil
import math



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers



from keras.layers import *
from keras.regularizers import l1_l2
from keras.initializers import Constant
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input,GRU
from keras.layers import *

from keras import optimizers, Sequential
from keras.models import Model, load_model


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import ElasticNet  
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import pearsonr



def standard_data(X):
    X2=X.copy()
    for i in range(X.shape[1]):
        scalers = StandardScaler()
        X2[:, i, :] = scalers.fit_transform(X[:, i, :]) 
    return X2




#type=cn or mci or comb
def load_data(hdf5='temporal500.h5',test_split = 0.3,comparetype='cn',seed=1):
    print ('Loading data...')
    f = h5py.File(hdf5, 'r')
    #print(list(f.keys()))
    if comparetype=='cn':
      X1 = f['X_cn_to_cn']
      X2=f['X_cn_to_mci']
      X1= np.array(X1)
      X2= np.array(X2)
    elif comparetype=='mci':
      X1 = f['X_mci_to_mci']
      X2=f['X_mci_to_dem']      
      X1= np.array(X1)
      X2= np.array(X2)  
    elif comparetype=='comb':
      X1_1 = f['X_cn_to_cn']
      X2_1=f['X_cn_to_mci']
      X1_1= np.array(X1_1)
      X2_1= np.array(X2_1)
      X1_2 = f['X_mci_to_mci']
      X2_2=f['X_mci_to_dem']
      X1_2= np.array(X1_2)
      X2_2= np.array(X2_2)
      X1=np.concatenate([X1_1,X1_2],axis=-1)
      X2=np.concatenate([X2_1,X2_2],axis=-1)

    X1=np.transpose(X1, (2, 1, 0))
    X2=np.transpose(X2, (2, 1, 0))
    Y1=np.zeros(X1.shape[0])
    Y2=np.ones(X2.shape[0])
    

    train_size1 = int(len(X1) * (1 - test_split))
    train_size2 = int(len(X2) * (1 - test_split))

    X_train1 = X1[:train_size1]
    X_train2 = X2[:train_size2]
    Y_train1 = Y1[:train_size1]
    Y_train2 = Y2[:train_size2]
    X_test1 =  X1[train_size1:]
    X_test2 =  X2[train_size2:]
    Y_test1 =  Y1[train_size1:]
    Y_test2 =  Y2[train_size2:]
    
    X_train=np.concatenate([X_train1,X_train2],axis=0)
    X_test=np.concatenate([X_test1,X_test2],axis=0)
    Y_train=np.concatenate([Y_train1,Y_train2])
    Y_test=np.concatenate([Y_test1,Y_test2])
    
    #shuffle label and matrix
    np.random.seed(seed)
    id_train=np.random.permutation(X_train.shape[0])
    id_test=np.random.permutation(X_test.shape[0])
    X_train=X_train[id_train]
    Y_train=Y_train[id_train]
    X_test=X_test[id_test]
    Y_test=Y_test[id_test]
    
    X=np.concatenate([X_train,X_test],axis=0)
    Y=np.concatenate([Y_train,Y_test])
    
    print('Training size: ',X_train.shape)

    return X,Y,X_train, Y_train, X_test, Y_test
 




'''
LSTMAE
'''





def createLSTMAE_hp(hp,X_train,AEtype):
    
    timesteps,n_features=X_train.shape[1],X_train.shape[2]
    inputs = Input(shape=(timesteps,n_features))
    
    #encoder and prediction    
    encoder=LSTM(units=hp['encoder_units1'], activation='relu',return_sequences=True)(inputs)
    encoder=LSTM(units=hp['encoder_units2'], activation='relu',return_sequences=False)(encoder)
    
    # prediction
    x = Dense(hp['dense_units'], activation='relu')(encoder)
    pred = Dense(2, activation="softmax",name='classification')(x)

    # reconstruct decoder
    decoder = RepeatVector(timesteps)(encoder)
    decoder = LSTM(hp['decoder_units'],activation='relu', return_sequences=True)(decoder)

    decoder = TimeDistributed(Dense(n_features),name='autoencoder')(decoder)
   
    model = Model(inputs=inputs, outputs=[pred,decoder])

    if AEtype=='define':
        model.compile(loss={'classification': 'binary_crossentropy', 
                        'autoencoder': 'mse'},
                  loss_weights={'classification': hp['alpha'],
                                 'autoencoder': 1-hp['alpha']},
                  
                  optimizer='adam',
                  metrics={'classification': 'accuracy', 'autoencoder': 'mse'})

    elif AEtype=='custom':
        '''
        use customized loss
        '''
        ### define customized function for combined loss
        def make_comb_loss(alpha):
            bce = tf.keras.losses.BinaryCrossentropy()
            mse=tf.keras.losses.MeanSquaredError()
            def comb_loss(y_true, y_pred):
                return (1 - hp['alpha']) * mse(y_true, y_pred) + hp['alpha'] * bce(y_true, y_pred)
            return comb_loss
        
        
        comb_loss=make_comb_loss(alpha)
        
        
        model.compile(loss=comb_loss,
                  optimizer='adam',
                  metrics={'classification': 'accuracy', 'autoencoder': 'mse'})
                  #metrics='accuracy')
        
    
    return model
    










'''
CNNAE
'''


def createCNNAE_hp(hp,X_train,AEtype):
    
    # Convolutional Encoder (avoid using duplicated names in encoder and decoder)
    timesteps,n_features=X_train.shape[1],X_train.shape[2]
    inputs = Input(shape=(timesteps,n_features))
    
    encoder = Conv1D(hp['encoder_conv_units1'], 2, activation='relu', padding='same')(inputs)
    encoder = MaxPooling1D(2, padding='same')(encoder)

    
    # Classification
    x = Flatten()(encoder)
    x = Dense(hp['encoder_dense_units1'], activation='relu')(x)
    pred = Dense(2, activation='softmax', name='classification')(x)
    
    
    # Decoder
    #new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0]])
    decoder=UpSampling1D(2)(encoder)
    decoder = Conv1DTranspose(hp['decoder_conv_units1'], 2, strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv1DTranspose(n_features, 2, activation='relu', padding='same', name='autoencoder')(decoder)
    model = Model(inputs=inputs, outputs=[pred, decoder])
    
    
    if AEtype=='define':
        model.compile(loss={'classification': 'binary_crossentropy', 
                        'autoencoder': 'mse'},
                  loss_weights={'classification': hp['alpha'],
                                 'autoencoder': 1-hp['alpha']},
                  
                  optimizer='adam',
                  metrics={'classification': 'accuracy', 'autoencoder': 'mse'})
              #metrics={'classification': 'accuracy', 'autoencoder': ['binary_crossentropy', 'mse']})

    elif AEtype=='custom':
        '''
        use customized loss
        '''
        ### define customized function for combined loss
        def make_comb_loss(alpha):
            bce = tf.keras.losses.BinaryCrossentropy()
            mse=tf.keras.losses.MeanSquaredError()
            def comb_loss(y_true, y_pred):
                return (1 - hp['alpha']) * mse(y_true, y_pred) + hp['alpha'] * bce(y_true, y_pred)
            return comb_loss
        
        
        comb_loss=make_comb_loss(alpha)
        
        
        model.compile(loss=comb_loss,
                  optimizer='adam',
                  metrics={'classification': 'accuracy', 'autoencoder': 'mse'})
                  #metrics='accuracy')
        
    
    return model
    




def trainModel(model,X_train,y_train,method,AEtype='define'):
    
    if 'AE' in method and 'AE0' not in method:
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        ]
        
        
        history=model.fit(X_train, 
              {'classification': y_train, 'autoencoder': X_train},
              batch_size=16, epochs=200, validation_split = 0.2, verbose = 0 #,
              #callbacks=callbacks
             )
        
    else:
        
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
        ]
        

        history = model.fit(X_train, y_train, 
                            callbacks=callbacks,
                            batch_size=16, epochs=100, validation_split = 0.2, verbose = 0)
        
    return model, history
        
        





def runHpModel(X_train, y_train, X_test, y_test,method,hparams,comparetype,isnorm,AEtype):

    if method=='LSTMAE':          
        model=createLSTMAE_hp(hparams,X_train,AEtype)
    elif method=='CNNAE':    
        model=createCNNAE_hp(hparams,X_train,AEtype)


    model,history=trainModel(model,X_train,y_train, method,AEtype)
    auc_test,acc_test,f1_test,mcc_test,auprc_test=evalModel(model, X_test,y_test,method)
    
    
    return [auc_test,acc_test,f1_test,mcc_test,auprc_test]
         

         




def evalModel(model,X_test,y_test,method):
    
    '''
    evaluation
    '''
    if 'AE' in method:
        y_test_prob,X_test_hat = model.predict(X_test)
    else:
        y_test_prob = model.predict(X_test)


    if y_test_prob.ndim==2:
        y_test_classes=np.argmax(y_test_prob, axis=-1)
        fpr, tpr, thresholds = roc_curve(y_test[:,0], y_test_prob[:,0])
        auc_test = auc(fpr, tpr)
        acc_test=accuracy_score(y_test_classes, np.argmax(y_test, axis=-1))
        f1_test = f1_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
        mcc_test = matthews_corrcoef(y_test_classes, np.argmax(y_test, axis=-1))
        auprc_test = average_precision_score(y_test[:,0], y_test_prob[:,0])
    elif y_test_prob.ndim==1:
        threshold=np.mean(y_test_prob)
        y_test_classes=np.copy(y_test_prob)
        y_test_classes[y_test_prob<threshold]=0
        y_test_classes[y_test_prob>=threshold]=1
        fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=-1), y_test_prob)
        auc_test = auc(fpr, tpr)
        acc_test=accuracy_score(y_test_classes, np.argmax(y_test, axis=-1))
        f1_test = f1_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
        mcc_test = matthews_corrcoef(y_test_classes, np.argmax(y_test, axis=-1))
        auprc_test = average_precision_score(y_test[:,0], y_test_prob)

        

    auc_test=round(auc_test,3)
    acc_test=round(acc_test,3)
    f1_test=round(f1_test,3)
    mcc_test=round(mcc_test,3)
    auprc_test=round(auprc_test,3)
    
    #print([auc_test,acc_test,f1_test,mcc_test])
    
    return [auc_test,acc_test,f1_test,mcc_test,auprc_test]

    


hdf5='temporal500.h5'
methods=['LSTMAE','CNNAE']
nrep=10;test_split=0.2;comparetype='cn'
isnorm='no';AEtype='define'


def benchmark(hdf5='temporal500.h5',
              methods=['LSTMAE','CNNAE'],
              nrep=10,test_split=0.2,comparetype='cn',
              isnorm='no',AEtype='define'):
    

    nmethod=len(methods)
    shape=(nrep,nmethod)
    
    auc_all=np.zeros(shape)
    acc_all=np.zeros(shape)
    f1_all=np.zeros(shape)
    mcc_all=np.zeros(shape)
    auprc_all=np.zeros(shape)

    
    for irep in range(nrep):
        print('Experiment',irep+1)
        X, Y, X_train, y_train, X_test, y_test=load_data(hdf5=hdf5,test_split = test_split,comparetype=comparetype,seed=irep)
        Y=tf.keras.utils.to_categorical(Y)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
            
        if isnorm=='yes':
            X=standard_data(X)
            X_train=standard_data(X_train)
            X_test=standard_data(X_test)
            
            
        for imethod in range(nmethod):   
            print(methods[imethod])

            

            hparams = np.load(methods[imethod].strip('_1t').strip('_2t')+'.'+comparetype+'.'+isnorm+'.npy',allow_pickle='TRUE').item()
            auc_test,acc_test,f1_test,mcc_test,auprc_test=runHpModel(X_train, y_train, X_test, y_test,methods[imethod],hparams,comparetype,isnorm,AEtype)

            
            auc_all[irep,imethod]=auc_test
            acc_all[irep,imethod]=acc_test
            f1_all[irep,imethod]=f1_test
            mcc_all[irep,imethod]=mcc_test
            auprc_all[irep,imethod]=auprc_test
            

        
        
    auc_all_df = pd.DataFrame(data = auc_all, columns = methods)
    acc_all_df = pd.DataFrame(data = acc_all, columns = methods)
    f1_all_df = pd.DataFrame(data = f1_all, columns = methods)
    mcc_all_df = pd.DataFrame(data = mcc_all, columns = methods)
    auprc_all_df = pd.DataFrame(data = auprc_all, columns = methods)

    
    # print('AUC')
    # print(auc_all_df)
    # print('ACC')
    # print(acc_all_df)
    # print('F1')
    # print(f1_all_df)
    # print('MCC')
    # print(mcc_all_df)
    # print('AUPRC')
    # print(auprc_all_df)
    
    # auc_all_df = pd.DataFrame(data = auc_all, columns = methods)
            
    return [auc_all,acc_all,f1_all,mcc_all,auprc_all_df]




    


def boxplot(df,comparetype,metric,isnorm):
    sns.set(font_scale=0.5)
    tmp=pd.melt(df)
    ax = sns.boxplot(x="variable", y="value", data=tmp)
    ax.set_title(metric)
    plt.setp(ax.get_xticklabels(), rotation=20)
    ax.get_figure().savefig(comparetype+'.'+metric+'.'+isnorm+'.png')
    plt.clf()




def boxplotpair(methods,comparetype,metric):
    sns.set(font_scale=0.5)
    hf_no = h5py.File(comparetype+'.no.h5', 'r')
    no=pd.DataFrame(hf_no.get(metric+'_no'),columns=methods)
    no=pd.melt(no)
    no['norm']='no'
    hf_yes = h5py.File(comparetype+'.yes.h5', 'r')
    yes=pd.DataFrame(hf_yes.get(metric+'_yes'),columns=methods)
    yes=pd.melt(yes)
    yes['norm']='yes'

    df=pd.concat([yes,no],axis=0)
    ax = sns.boxplot(x="variable", y="value", hue='norm',data=df)
    ax.set_title(metric)
    plt.setp(ax.get_xticklabels(), rotation=50)
    ax.get_figure().savefig(comparetype+'.'+metric+'.both.png')
    sns.despine(offset=10, trim=True)
    plt.clf()
    
    



def outputResult(auc_all,acc_all,f1_all,mcc_all,auprc_all,methods,comparetype,isnorm):

    auc_all_df = pd.DataFrame(data = auc_all, columns = methods)
    acc_all_df = pd.DataFrame(data = acc_all, columns = methods)
    f1_all_df = pd.DataFrame(data = f1_all, columns = methods)
    mcc_all_df = pd.DataFrame(data = mcc_all, columns = methods)
    auprc_all_df = pd.DataFrame(data = auprc_all, columns = methods)


    
    
    boxplot(auc_all_df,metric='auc',comparetype=comparetype,isnorm=isnorm)
    boxplot(f1_all_df,metric='f1',comparetype=comparetype,isnorm=isnorm)
    boxplot(acc_all_df,metric='acc',comparetype=comparetype,isnorm=isnorm)
    boxplot(mcc_all_df,metric='mcc',comparetype=comparetype,isnorm=isnorm)
    boxplot(auprc_all_df,metric='auprc',comparetype=comparetype,isnorm=isnorm)




