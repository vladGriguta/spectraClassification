#!/usr/bin/env python
# coding: utf-8

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
import glob
import multiprocessing
freeProc = 2
n_proc=multiprocessing.cpu_count()-freeProc
from sklearn.metrics import confusion_matrix #, classification_report, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.initializers import random_uniform
from keras import layers
from keras.optimizers import RMSprop
import gc
gc.collect()

# load and save functions
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pd.DataFrame(pickle.load(f))



def read_data(locationSpectra):
    filenames = glob.glob(locationSpectra+'*pkl')
    
    cut_off = 5000
    X = np.zeros((4900,cut_off,2))
    #wavelength = np.zeros((len(filenames),cut_off))
    X_scaled = np.zeros((4900,cut_off,2))
    y = []
    sc = MinMaxScaler()
    counter_excluded = 0
    for i in range(4900):
        df_current = load_obj(filenames[i])
        l = len(df_current['model'])
        
        wavelength = np.power(10,df_current['loglam'][0:l])
        flux = df_current['model'][0:l]
        flux_scaled = np.array(sc.fit_transform(np.array(flux).reshape(-1,1))).reshape(len(flux))
        
        X[i][0:l] = np.stack((wavelength,flux),axis=1)
        
        # Scale result in new array
        X_scaled[i][0:l] = np.stack((wavelength,flux_scaled),axis=1)
        
        y.append(df_current['information'].iloc[0])
    X = X[0:(len(X)-counter_excluded)]
    X = np.array(X)
    wavelength = wavelength[0:(len(X)-counter_excluded)]
    y = np.array(y)
    
    return X,y,X_scaled


def paralellised_append(elem_X,elem_X_new,min_X,step):
    for j in range(elem_X.shape[0]):
        if(elem_X[j,0]>0):
            elem_X_new[int((elem_X[j,0] - min_X) / step)].append(elem_X[j,1])
    return elem_X_new

def paralellised_assign(elem_X_new):
    for j in range(elem_X_new.shape[0]):
        elem_X_new[j] = np.array(elem_X_new[j])
        if(elem_X_new[j].size > 0):
            elem_X_new[j] = np.mean(np.array(elem_X_new[j]))
        else:
            elem_X_new[j] = 0.
    return elem_X_new

def prepare_cnn_entries(X,n_nodes=100):
    """
    This function transforms the wavelength input to a discrete input so that
    it can be fed to the CNN.
    """
    idx = X[:,:,0] > 0
    X_nozeros = X[idx]
    min_X = np.min(X_nozeros[:,0])
    max_X = np.max(X_nozeros[:,0])
    del X_nozeros
    gc.collect()  

    
    
    wavelength_all,step = np.linspace(min_X,max_X, num=n_nodes,retstep=True)
    #wavelength_edges = np.histogram_bin_edges(wavelength_all,bins=n_nodes)

    X_new = [[[] for i in range(n_nodes)] for j in range(X.shape[0])]

    
    print('Start converting to discrete wavelength inputs...')    
    for i in range(X.shape[0]):
        if(i % int(X.shape[0] / 10) == 0):
            print('Part 1.... ' + str(round(100*i/X.shape[0],0)) + ' % completed')
        X_new[i] = paralellised_append(X[i],X_new[i],min_X,step)
    
    #del X
    gc.collect()
    
    X_new = np.array(X_new)
    print(np.count_nonzero(X_new))
    
    for i in range(X_new.shape[0]):
        if(i % int(X_new.shape[0] / 10) == 0):
            print('Part 2.... ' + str(round(100*i/X_new.shape[0])) + ' % completed')
        X_new[i] = paralellised_assign(X_new[i])
            
    print(np.count_nonzero(X_new))
    
    return X_new
        

def encode_data(y):
    # encode class values as integers
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y,encoder

def decode(dummy_y,encoder):
    """
    Function that takes the dummy variable and its encoder and transforms it
    back to the initial form
    """
    # from dummy back to class names
    encoded_y = np.zeros(len(dummy_y))
    for i in range(len(dummy_y)):
        encoded_y[i] = int(np.argmax(dummy_y[i]))
    classes_y =  encoder.inverse_transform(encoded_y.astype(int))
    
    return classes_y,encoded_y


# Defining a function to plot smoother plots of validation and accuracy - this often helps to identify a trend better
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points


#This longer code gets a prettier confusion matrix.
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, annot=annot, fmt='', ax=ax) # Changing cm_perc to cm give a heatmap in terms of the percentages instead of absolute number 
    plt.savefig(filename)

def plot_confusion_matrix(cm, target_names, location):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + target_names)
    ax.set_yticklabels([''] + target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
        # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(location+'ConfusionMatrix')
    plt.close()
    return ax


if __name__ == '__main__':
    # load all spectra in internal memory 
    locationSpectra = 'spectra_matched_multiproc/'
    location_plots = 'CNN_plots_newInput/'
    if not os.path.exists(location_plots):
        os.makedirs(location_plots) 
    
    _,y,X = read_data(locationSpectra)
    
    n_nodes = 300
    X = prepare_cnn_entries(X,n_nodes)
    X = X.reshape((X.shape[0],X.shape[1],1))
    print('Successfull')

    
    dummy_y,encoder_y = encode_data(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.1,  random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    
    
    # Enter target names 
    target_names = ['GALAXY', 'QSO', 'STAR']
    numberTargets = len(target_names) # Useful later (for the final dense layer of CNN).
    
    
    # Reshape for CNN. (Honestly, I'm not sure why this makes a difference) 
    """
    X_train = np.reshape(X_train, (np.size(X_train,0),1, np.size(X_train,1),np.size(X_train,2)))
    X_test = np.reshape(X_test, (np.size(X_test,0),1, np.size(X_test,1),np.size(X_test,2)))
    X_val = np.reshape(X_val, (np.size(X_val,0),1, np.size(X_val,1),np.size(X_val,2)))
    """
    
    # Choose hyperparameters
    no_epochs = 25
    batch_size = 32
    learning_rate = 0.0100
    dropout_rate = 0.05

    # Design the Network
    model = Sequential()
    model.add(layers.Conv1D(64, 6, activation='relu',  input_shape=X_train[0].shape)) # Input shape is VERY fiddly. May need to try different things. 
    model.add(Dropout(dropout_rate))
    model.add(layers.Conv1D(128, 6, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dense(numberTargets, activation='softmax'))
    print(model.summary())
    
    """
    # Design the Network
    model = Sequential()
    model.add(layers.SeparableConv2D(32, (1, 2), input_shape=X_train[0].shape, activation='relu'))
    model.add(Dropout(dropout_rate))
    #model.add(layers.BatchNormalization()) # BatchNormalization and dropout work poorly together, though - Ioffe & Szegedy 2015; Li et al. 2018 
    model.add(layers.SeparableConv2D(64, (1, 3), activation='sigmoid'))
    model.add(layers.SeparableConv2D(128, (1, 4), activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(numberTargets, activation='softmax'))
    """
    
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss='categorical_crossentropy', # May need to change to binary_crossentropy or categorical_crossentropy
                  metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=no_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    predictions = model.predict_proba(X_test)
    predicted_labels = predictions.argmax(axis=1) # Converts probabilities (e.g. 0.035 0.001 0.704 0.260) to labels (e.g. 0 0 1 0)
    
    
    # Evaluation 
    plt.close()
    # Now our model is built and trained and tested; we look at results
    # First, the loss and accuracy on training+validation data.
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy (Unsmoothed)')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(location_plots+'accuracy.png')
    plt.close()
    
    
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(location_plots+'loss.png')
    plt.close()
    
    plt.figure()
    plt.plot(smooth_curve(history.history['acc']), 'bo', label='Smoothed training acc', alpha=0.5)
    plt.plot(smooth_curve(history.history['val_acc']), 'b', label='Smoothed validation acc')
    plt.title('Training and validation Accuracy (smoothed')
    plt.legend()
    plt.savefig(location_plots+'accuracy_smoothed.png')
    plt.close()
    
    
    
    plt.figure()
    plt.plot(smooth_curve(history.history['loss']), 'bo', label='Smoothed training loss', alpha=0.5)
    plt.plot(smooth_curve(history.history['val_loss']), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss (Smoothed)')
    plt.legend()
    plt.savefig(location_plots+'loss_smoothed.png')
    plt.close()
    
    
    # Now let's see the results on Test data; rather than just training and validation sets
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Note that in multi-class and imbalanced classification problems, test accuracy is not an ideal metric')
    cm = confusion_matrix(y_test.argmax(axis=1), predicted_labels)
    print("Confusion matrix:\n{}".format(cm))

    y_true = y_test.argmax(axis=1)
    #cm_analysis(y_true, predicted_labels, filename=location_plots+'confMatrix.png', labels=[0,1,2])
    plot_confusion_matrix(cm, target_names, location=location_plots)
    plt.close()