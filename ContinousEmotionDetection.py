
from scipy.sparse import *
import numpy as np

import scipy.io as sio
from scipy.stats.stats import pearsonr
from sklearn import preprocessing


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Masking, concatenate


from sklearn.metrics import mean_squared_error
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras import backend as K




def loadData (dataFolder,annotedTrial):     
    dataNum = 0
    for cnt,ParTrial in enumerate(annotedTrial):
        dataAddress ="{}P{}-features-resamp4hz-trial-{}.mat".format(dataFolder,ParTrial[0],ParTrial[1])
        if cnt==0:
                    features_label =sio.loadmat(dataAddress)
        features_label[cnt] =sio.loadmat(dataAddress)
        dataNum +=np.shape((features_label[cnt]['face_feats']))[0]
    return features_label, dataNum




def normalizeDataSubjet(annotedTrial,features_label,dataNum):   
    faceFe = np.zeros((dataNum,38))
    eegFe = np.zeros((dataNum,128))
    target = np.zeros((dataNum,1))
    startTrial = int(0)
    trialNum = int(np.shape(annotedTrial)[0])
    lengthTrial = []
    partNumSample = 0
    partTrial = []
    for cnt,ParTrial in enumerate(annotedTrial):
        
        lengthTrial.append(int(np.shape((features_label[cnt]['face_feats']))[0]))
        endTrial = startTrial + int(lengthTrial[cnt])
        
        faceFe[startTrial:endTrial,:]  = features_label[cnt]['face_feats'][:,0:38]
        eegFe[startTrial:endTrial,:]   = features_label[cnt]['eeg_band_feats_full']
        target [startTrial:endTrial,:] = np.transpose(features_label[cnt]['target'])
        
        startTrial = endTrial
        
        if cnt>0:
            if annotation['trials_included'][cnt,0]==annotation['trials_included'][cnt-1,0] and cnt!= trialNum-1 :
                partNumSample += int(lengthTrial[cnt])

            else:
                if cnt == trialNum-1:
                    partNumSample += int(lengthTrial[cnt])
                start = int(np.sum(partTrial))
                faceFe[start:int(start+partNumSample),:] = preprocessing.scale(faceFe[start:int(start+partNumSample),:])
                eegFe[start:int(start+partNumSample),:] = preprocessing.scale(eegFe[start:int(start+partNumSample),:])
                partTrial.append(partNumSample)
                partNumSample = int(lengthTrial[cnt])

        else:
            partNumSample = int(lengthTrial[cnt])
            
    return faceFe, eegFe, target, lengthTrial 
    




def array2sequence (npArray,maxLen,seqNum,lengthTrial):   
    tensorSeq = np.zeros((seqNum,maxLen,int(np.shape(npArray)[1])))
  
    startTrial = 0
    for cnt,lenTrial in enumerate(lengthTrial): 
        endTrial = startTrial + int(lenTrial)
        
        tensorSeq[cnt, :int(lenTrial), :]  = npArray[startTrial:endTrial,:]
        
        startTrial = endTrial
    return tensorSeq




def maskSequence (maxLen,seqNum):   
    maskSeq = np.zeros((seqNum,maxLen))
  
    startTrial = 0
    for cnt,lenTrial in enumerate(lengthTrial): 
        
        maskSeq[cnt,:int(lenTrial)] = np.ones((1,int(lenTrial)))

    return maskSeq


def splitData( tensorFaceFe, tensorEegFe, tensorTarget, maskFe, lengthTrial, trainShare,valShare ):    
    # split into train, validation and test sets
    trainSize = int(len(tensorFaceFe) * trainShare)
    valSize =  int(len(tensorFaceFe) * valShare)
    testSize = int(len(tensorFaceFe)-trainSize-valSize)

    indices = np.arange(len(tensorFaceFe),dtype='int')

    np.random.shuffle(indices)

    dataFace = tensorFaceFe[indices,:,:]
    dataEeg  = tensorEegFe[indices,:,:]
    labels   = tensorTarget[indices,:,:]
    dataMask = maskFe[indices,:]
    lenTrRnd= lengthTrial[indices]
    
    trainFace, valFace, testFace = dataFace[0:trainSize,:,:], dataFace[trainSize:trainSize+valSize,:,:], dataFace[trainSize+valSize:,:,:]

    trainEeg, valEeg, testEeg = dataEeg[0:trainSize,:,:], dataEeg[trainSize:trainSize+valSize,:,:], dataEeg[trainSize+valSize:,:,:]
    lenSeqTrain,lenSeqVal ,lenSeqTest = lenTrRnd[0:trainSize], lenTrRnd[trainSize:trainSize+valSize], lenTrRnd[trainSize+valSize:]

    labelTrain = labels[0:trainSize,:,:]
    labelVal = labels[trainSize:trainSize+valSize,:,:]
    labelTest = labels[trainSize+valSize:len(tensorFaceFe),:,:]
    maskTrain = dataMask[0:trainSize,:,]
    
    
    return trainFace,valFace,testFace, trainEeg,valEeg,testEeg, labelTrain,labelVal,labelTest,           maskTrain, lenSeqTrain,lenSeqVal ,lenSeqTest


def custom_loss(y_true, y_pred):
    loss = []
    dot = lambda a, b: K.batch_dot(a, b, axes=1)
    center = lambda x: x- K.mean(x)
    cov = lambda a, b: K.mean(dot(center(a),center(b)))
    correlation = lambda x: cov(x[0], x[1]) /(K.std(x[0])*K.std(x[1]))
    for cnt in range(np.shape(y_pred)[2]):
        # 3D tensor
        loss.append(correlation((y_true[cnt,:,:],y_pred[cnt,:,:]))-2*K.mean(K.square(y_true[cnt,:,:]-y_pred[cnt,:,:])))

    return  1- np.sum(loss)/int(np.shape(y_pred)[2])


def networkSeq(inputTensor, outputTensor, maxLen, lstmSize1, lstmSize2, denseSize1):
    featSize = np.shape(inputTensor)[2]
    outputSize = np.shape(outputTensor)[2]
    model = Sequential()
    model.add(Masking(mask_value=0.,batch_input_shape=(None, maxLen, featSize)))
    model.add(LSTM(lstmSize1 , batch_input_shape=(None, maxLen, featSize), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(lstmSize2, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(32)))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(outputSize,activation='linear')))
    model.compile(loss='mse', optimizer='Nadam',sample_weight_mode="temporal")
    return model



def trainNetworkSeq(model,trainData,valData,labelTrain,labelVal, epochs,batchSize,maskTrain=None,mode='min',pathSaveNetWeight="weights.best.hdf5"):
    # checkpoint
    filepath=pathSaveNetWeight
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode=mode)
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(trainData,labelTrain, validation_data=(valData,labelVal),epochs=epochs, batch_size=batchSize, callbacks=callbacks_list, sample_weight=maskTrain)
    model.load_weights(filepath)
    return model



def evaluteSeq(model,testdata, labelreal,batchSize,lenSeq) :   
        
    # predict test sequeces
    labelPredic = model.predict(testdata, batch_size=batchSize, verbose=0)
    
    mse, rmse, pearsonCof = metricSeq(labelreal, labelPredic, lenSeq)
    
    return mse, rmse, pearsonCof, labelPredic




def metricSeq(labelreal, labelPredic, lenSeq):
    # evalute the predict sequences (the sequences have different lengths)
    #min_max_scaler = preprocessing.MinMaxScaler()
    mse = []
    pearson = []
    for cnt in range(len(labelreal)):
        predictSeq = labelPredic[cnt,:int(lenSeq[cnt]),:]
        realSeq    = labelreal[cnt,:int(lenSeq[cnt]),:]    
        #A = min_max_scaler.fit_transform(A)-0.5
        #B = min_max_scaler.fit_transform(B)-0.5
        mse.append(mean_squared_error(predictSeq, realSeq))
        pearson.append(pearsonr(predictSeq, realSeq)[0])
        
    return np.mean(mse), np.sqrt(np.mean(mse)), np.mean(pearson)




def printRedsult(mse,rmse,pearsonCof,title) :
    print()
    print("**********",title,"**********")
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('p = ', pearsonCof)
    print()




if __name__ == '__main__':
    
    #split data parameter
    trainShare = 0.6
    valShare   = 0.3
    
    # Parameter define
    lstmEegLa1 = 64
    lstmEegLa2 = 32
    denseEeg   = 32

    lstmFaceLa1 = 19
    lstmFaceLa2 = 10
    denseFace   = 10
    pathSaveNetWeightEeg="weights.bestEEG.hdf5"
    pathSaveNetWeightFace="weights.bestFace.hdf5"
    pathSaveNetWeightFLF="weights.bestFLF.hdf5"
    epochs = 25
    batchSize = 20

    dataFolder = './data/Features/'
    annotationAdd = './data/lable_continous_Mahnob.mat'
    annotation = sio.loadmat(annotationAdd)
    features_label, dataNum = loadData(dataFolder,annotation['trials_included'])
    faceFe, eegFe, target, lengthTrial = normalizeDataSubjet(annotation['trials_included'],features_label,dataNum)
    maxLen = int(np.max(lengthTrial))
    seqNum = len(annotation['trials_included'])

    tensorFaceFe = array2sequence (faceFe,maxLen,seqNum,lengthTrial)
    tensorEegFe  = array2sequence (eegFe,maxLen,seqNum,lengthTrial)
    tensorTarget = array2sequence (target,maxLen,seqNum,lengthTrial)
    maskFe = maskSequence (maxLen,seqNum)

    trainFace,valFace,testFace, trainEeg,valEeg,testEeg, labelTrain,labelVal,labelTest, maskTrain, lenSeqTrain,lenSeqVal             ,lenSeqTest = splitData( tensorFaceFe, tensorEegFe, tensorTarget, maskFe, np.array(lengthTrial), trainShare,valShare )

    # define and train networks
    print('Train EEG network')
    modelEeg  = networkSeq(tensorEegFe,  tensorTarget, maxLen, lstmEegLa1,  lstmEegLa2,  denseEeg)
    #print(modelEeg.summary())
    modelEeg = trainNetworkSeq(modelEeg,trainEeg,valEeg,labelTrain,labelVal, epochs,batchSize,maskTrain,'min',pathSaveNetWeightEeg)

    print('Train Face network')
    modelFace = networkSeq(tensorFaceFe, tensorTarget, maxLen, lstmFaceLa1, lstmFaceLa2, denseFace)
    #print(modelFace.summary())
    modelFace = trainNetworkSeq(modelFace,trainFace,valFace,labelTrain,labelVal, epochs,batchSize,maskTrain,'min',pathSaveNetWeightFace)

    # feature level fusion

    #Fuse the features
    tensorFLF = np.concatenate((tensorFaceFe, tensorEegFe), axis=2)
    trainFLF  = np.concatenate((trainFace, trainEeg), axis=2)
    valFLF  = np.concatenate((valFace, valEeg), axis=2)
    testFLF  = np.concatenate((testFace, testEeg), axis=2)

    print('Train feature level fusion network')
    modelFLF = networkSeq(tensorFLF, tensorTarget, maxLen, lstmFaceLa1+lstmEegLa1, lstmFaceLa2+lstmEegLa2, denseFace+denseEeg)
    modelFLF = trainNetworkSeq(modelFLF,trainFLF,valFLF,labelTrain,labelVal, epochs,batchSize,maskTrain,'min',pathSaveNetWeightFLF)

    # results
    mseEeg,rmseEeg,pearsonCofEeg,prdictTestEeg  = evaluteSeq(modelEeg,testEeg,  labelTest,batchSize,lenSeqTest)
    printRedsult(mseEeg,rmseEeg,pearsonCofEeg,'EEG') 

    mseFace,rmseFace,pearsonCofFace,prdictTestFace = evaluteSeq(modelFace,testFace, labelTest,batchSize,lenSeqTest)
    printRedsult(mseFace,rmseFace,pearsonCofFace,'Face') 

    mseFLF,rmseFLF,pearsonCofFLF,prdictTestFLF = evaluteSeq(modelFLF,  testFLF,  labelTest,batchSize,lenSeqTest)
    printRedsult(mseFLF,rmseFLF,pearsonCofFLF,'FLF') 

    #decision lebel fusion
    prdictTestDLF = 0.5*(prdictTestEeg+prdictTestFace)
    mseDLF, rmseDLF, pearsonCofDLF = metricSeq(labelTest, prdictTestDLF, lenSeqTest)
    printRedsult(mseDLF,rmseDLF,pearsonCofDLF,'DLF') 








