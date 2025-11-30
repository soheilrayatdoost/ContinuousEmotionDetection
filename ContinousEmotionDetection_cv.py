from scipy.sparse import *
import numpy as np

import scipy.io as sio
from scipy.stats import pearsonr
from sklearn import preprocessing


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Masking, concatenate, TimeDistributed


from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K




def loadData (dataFolder,annotedTrial):     
    dataNum = 0
    for cnt,ParTrial in enumerate(annotedTrial):
        dataAddress ="{}P{}-features-resamp4hz-trial-{}.mat".format(dataFolder,ParTrial[0],ParTrial[1])
        if cnt==0:
                    features_label =sio.loadmat(dataAddress)
        features_label[cnt] =sio.loadmat(dataAddress)
        dataNum +=np.shape((features_label[cnt]['face_feats']))[0]
    return features_label, dataNum




def normalizeDataSubjet(annotedTrial,features_label,dataNum,annotation):   
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




def maskSequence (maxLen,seqNum,lengthTrial):   
    maskSeq = np.zeros((seqNum,maxLen))
  
    startTrial = 0
    for cnt,lenTrial in enumerate(lengthTrial): 
        
        maskSeq[cnt,:int(lenTrial)] = np.ones((1,int(lenTrial)))

    return maskSeq


def createKFolds(numSamples, k=10, shuffle=True, randomSeed=42):
    """Create k-fold split indices for cross-validation."""
    indices = np.arange(numSamples)
    
    if shuffle:
        if randomSeed is not None:
            np.random.seed(randomSeed)
        np.random.shuffle(indices)
    
    foldSizes = np.full(k, numSamples // k, dtype=int)
    foldSizes[:numSamples % k] += 1
    
    folds = []
    current = 0
    for foldSize in foldSizes:
        start, stop = current, current + foldSize
        testIndices = indices[start:stop]
        trainIndices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((trainIndices, testIndices))
        current = stop
    
    return folds


def splitDataKFold(tensorFaceFe, tensorEegFe, tensorTarget, maskFe, lengthTrial, trainIndices, testIndices, valSplit=0.2):
    """Split data for one fold of cross-validation."""
    # Extract test set
    testFace = tensorFaceFe[testIndices,:,:]
    testEeg  = tensorEegFe[testIndices,:,:]
    labelTest = tensorTarget[testIndices,:,:]
    lenSeqTest = lengthTrial[testIndices]
    
    # Split training set into train and validation
    numTrain = len(trainIndices)
    numVal = int(numTrain * valSplit)
    
    trainIndicesShuffled = trainIndices.copy()
    np.random.shuffle(trainIndicesShuffled)
    
    valIndices = trainIndicesShuffled[:numVal]
    trainIndicesFinal = trainIndicesShuffled[numVal:]
    
    # Extract training set
    trainFace = tensorFaceFe[trainIndicesFinal,:,:]
    trainEeg  = tensorEegFe[trainIndicesFinal,:,:]
    labelTrain = tensorTarget[trainIndicesFinal,:,:]
    maskTrain = maskFe[trainIndicesFinal,:]
    lenSeqTrain = lengthTrial[trainIndicesFinal]
    
    # Extract validation set
    valFace = tensorFaceFe[valIndices,:,:]
    valEeg  = tensorEegFe[valIndices,:,:]
    labelVal = tensorTarget[valIndices,:,:]
    lenSeqVal = lengthTrial[valIndices]
    
    return trainFace, valFace, testFace, trainEeg, valEeg, testEeg, labelTrain, labelVal, labelTest, maskTrain, lenSeqTrain, lenSeqVal, lenSeqTest


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
    model.add(Masking(mask_value=0., input_shape=(maxLen, featSize)))
    model.add(LSTM(lstmSize1, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(lstmSize2, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(32)))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(outputSize,activation='linear')))
    model.compile(loss='mse', optimizer='nadam')
    return model



def trainNetworkSeq(model,trainData,valData,labelTrain,labelVal, epochs,batchSize,maskTrain=None,mode='min',pathSaveNetWeight="weights.best.keras"):
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
    mse = []
    pearson = []
    for cnt in range(len(labelreal)):
        predictSeq = labelPredic[cnt,:int(lenSeq[cnt]),:]
        realSeq    = labelreal[cnt,:int(lenSeq[cnt]),:]    
        mse.append(mean_squared_error(predictSeq, realSeq))
        pearson.append(pearsonr(predictSeq.flatten(), realSeq.flatten())[0])
        
    return np.mean(mse), np.sqrt(np.mean(mse)), np.mean(pearson)


def printFoldResults(foldNum, results):
    """Print results for a single fold."""
    print()
    print(f"{'='*70}")
    print(f"FOLD {foldNum} RESULTS")
    print('='*70)
    
    for model, (mse, rmse, pearson) in results.items():
        print(f"\n{model}:")
        print(f"  MSE     = {mse:.6f}")
        print(f"  RMSE    = {rmse:.6f}")
        print(f"  Pearson = {pearson:.6f}")


def aggregateFoldResults(foldResults):
    """Aggregate results across all folds."""
    models = foldResults[0].keys()
    aggregated = {}
    
    for model in models:
        mseValues = [fold[model][0] for fold in foldResults]
        rmseValues = [fold[model][1] for fold in foldResults]
        pearsonValues = [fold[model][2] for fold in foldResults]
        
        aggregated[model] = (
            np.mean(mseValues), np.std(mseValues),
            np.mean(rmseValues), np.std(rmseValues),
            np.mean(pearsonValues), np.std(pearsonValues)
        )
    
    return aggregated


def printAggregatedResults(aggregated):
    """Print aggregated results across all folds."""
    print()
    print(f"{'='*70}")
    print("10-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
    print('='*70)
    
    for model, (mseMean, mseStd, rmseMean, rmseStd, pearsonMean, pearsonStd) in aggregated.items():
        print(f"\n{model}:")
        print(f"  MSE     = {mseMean:.6f} ± {mseStd:.6f}")
        print(f"  RMSE    = {rmseMean:.6f} ± {rmseStd:.6f}")
        print(f"  Pearson = {pearsonMean:.6f} ± {pearsonStd:.6f}")
    
    print()
    print(f"{'='*70}")


if __name__ == '__main__':
    
    # Parameter define
    lstmEegLa1 = 64
    lstmEegLa2 = 32
    denseEeg   = 32

    lstmFaceLa1 = 19
    lstmFaceLa2 = 10
    denseFace   = 10
    epochs = 100
    batchSize = 20

    dataFolder = './data/Features/'
    annotationAdd = './data/lable_continous_Mahnob.mat'
    
    print("Loading annotations...")
    annotation = sio.loadmat(annotationAdd)
    
    print("Loading feature data...")
    features_label, dataNum = loadData(dataFolder,annotation['trials_included'])
    
    print("Normalizing data...")
    faceFe, eegFe, target, lengthTrial = normalizeDataSubjet(annotation['trials_included'],features_label,dataNum,annotation)
    
    maxLen = int(np.max(lengthTrial))
    seqNum = len(annotation['trials_included'])

    print("Converting to sequences...")
    tensorFaceFe = array2sequence(faceFe,maxLen,seqNum,lengthTrial)
    tensorEegFe  = array2sequence(eegFe,maxLen,seqNum,lengthTrial)
    tensorTarget = array2sequence(target,maxLen,seqNum,lengthTrial)
    maskFe = maskSequence(maxLen,seqNum,lengthTrial)

    # Create 10-fold splits
    print("\nCreating 10-fold cross-validation splits...")
    kFolds = createKFolds(seqNum, k=10, shuffle=True, randomSeed=42)
    
    # Store results for all folds
    allFoldResults = []
    
    # Perform cross-validation
    for foldNum, (trainIndices, testIndices) in enumerate(kFolds, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {foldNum}/10")
        print('='*70)
        
        # Split data for this fold
        trainFace, valFace, testFace, trainEeg, valEeg, testEeg, labelTrain, labelVal, labelTest, maskTrain, lenSeqTrain, lenSeqVal, lenSeqTest = splitDataKFold(
            tensorFaceFe, tensorEegFe, tensorTarget, maskFe,
            np.array(lengthTrial), trainIndices, testIndices, valSplit=0.2
        )

        # Train EEG network
        print(f'\nTrain EEG network (Fold {foldNum})')
        modelEeg = networkSeq(tensorEegFe, tensorTarget, maxLen, lstmEegLa1, lstmEegLa2, denseEeg)
        modelEeg = trainNetworkSeq(modelEeg, trainEeg, valEeg, labelTrain, labelVal, epochs, batchSize, maskTrain, 'min', f'weights_fold{foldNum}_EEG_old.keras')

        # Train Face network
        print(f'\nTrain Face network (Fold {foldNum})')
        modelFace = networkSeq(tensorFaceFe, tensorTarget, maxLen, lstmFaceLa1, lstmFaceLa2, denseFace)
        modelFace = trainNetworkSeq(modelFace, trainFace, valFace, labelTrain, labelVal, epochs, batchSize, maskTrain, 'min', f'weights_fold{foldNum}_Face_old.keras')

        # Feature level fusion
        print(f'\nTrain feature level fusion network (Fold {foldNum})')
        tensorFLF = np.concatenate((tensorFaceFe, tensorEegFe), axis=2)
        trainFLF  = np.concatenate((trainFace, trainEeg), axis=2)
        valFLF  = np.concatenate((valFace, valEeg), axis=2)
        testFLF  = np.concatenate((testFace, testEeg), axis=2)

        modelFLF = networkSeq(tensorFLF, tensorTarget, maxLen, lstmFaceLa1+lstmEegLa1, lstmFaceLa2+lstmEegLa2, denseFace+denseEeg)
        modelFLF = trainNetworkSeq(modelFLF, trainFLF, valFLF, labelTrain, labelVal, epochs, batchSize, maskTrain, 'min', f'weights_fold{foldNum}_FLF_old.keras')

        # Evaluate
        print(f'\nEvaluating Fold {foldNum}...')
        mseEeg, rmseEeg, pearsonCofEeg, prdictTestEeg = evaluteSeq(modelEeg, testEeg, labelTest, batchSize, lenSeqTest)
        mseFace, rmseFace, pearsonCofFace, prdictTestFace = evaluteSeq(modelFace, testFace, labelTest, batchSize, lenSeqTest)
        mseFLF, rmseFLF, pearsonCofFLF, prdictTestFLF = evaluteSeq(modelFLF, testFLF, labelTest, batchSize, lenSeqTest)

        # Decision level fusion
        prdictTestDLF = 0.5*(prdictTestEeg+prdictTestFace)
        mseDLF, rmseDLF, pearsonCofDLF = metricSeq(labelTest, prdictTestDLF, lenSeqTest)
        
        # Store results
        foldResults = {
            'EEG': (mseEeg, rmseEeg, pearsonCofEeg),
            'Face': (mseFace, rmseFace, pearsonCofFace),
            'FLF': (mseFLF, rmseFLF, pearsonCofFLF),
            'DLF': (mseDLF, rmseDLF, pearsonCofDLF)
        }
        
        printFoldResults(foldNum, foldResults)
        allFoldResults.append(foldResults)
    
    # Aggregate and print final results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    aggregatedResults = aggregateFoldResults(allFoldResults)
    printAggregatedResults(aggregatedResults)
    
    print("\n10-FOLD CROSS-VALIDATION COMPLETE!")
