import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #, classification_report, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import keras
from keras import backend as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.initializers import random_uniform
from keras import layers
from keras.optimizers import RMSprop#, SGD, Adagrad, Adadelta, Adam, Adamax, Nadam

        
# Read in data
data = np.load('/FilePath/DataFileName')
labels = np.load('/FilePath/LabelsFileName')

# Split data into training, testing, validation datasets. 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1,  random_state=42) # 90%/10%/0% train/test/val. Change test_size to change the proportions of the data split. 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111111111, random_state=1) # 80%/10%/10%.

# Enter target names 
target_names = ['Class1', 'Class2', 'Class3'] 
numberTargets = len(target_names) # Useful later (for the final dense layer of CNN).

# Reshape for CNN. (Honestly, I'm not sure why this makes a difference) 
X_train = np.reshape(X_train, (np.size(X_train,0), np.size(X_train,1), 1))
X_test = np.reshape(X_test, (np.size(X_test,0), np.size(X_test,1), 1))
X_val = np.reshape(X_val, (np.size(X_val,0), np.size(X_val,1), 1))

# One-hot-encode labels for CNN 
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)
y_val = keras.utils.np_utils.to_categorical(y_val)


# Choose hyperparameters
no_epochs = 20
batch_size = 128
learning_rate = 0.0100
dropout_rate = 0.45

# Design the Network
model = Sequential()
model.add(layers.Conv1D(32, 6, activation='relu',  input_shape=X_train[0].shape)) # Input shape is VERY fiddly. May need to try different things. 
model.add(Dropout(dropout_rate))
model.add(layers.Conv1D(64, 4, activation='relu'))
model.add(layers.Conv1D(128, 6, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(layers.Conv1D(64, 4, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(Dense(numberTargets, activation='softmax'))
print(model.summary())

model.compile(optimizer=RMSprop(lr=learning_rate),
              loss='categorical_crossentropy', # May need to change to binary_crossentropy or categorical_crossentropy
              metrics=['acc'])
history = model.fit(X_train, y_train, epochs=no_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
predictions = model.predict_proba(X_test)
predicted_labels = predictions.argmax(axis=1) # Converts probabilities (e.g. 0.035 0.001 0.704 0.260) to labels (e.g. 0 0 1 0)


# Evaluation 

# Now our model is built and trained and tested; we look at results
# First, the loss and accuracy on training+validation data.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy (Unsmoothed)')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


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

plt.figure()
plt.plot(smooth_curve(history.history['acc']), 'bo', label='Smoothed training acc', alpha=0.5)
plt.plot(smooth_curve(history.history['val_acc']), 'b', label='Smoothed validation acc')
plt.title('Training and validation Accuracy (smoothed')
plt.legend()
plt.show()

plt.figure()
plt.plot(smooth_curve(history.history['loss']), 'bo', label='Smoothed training loss', alpha=0.5)
plt.plot(smooth_curve(history.history['val_loss']), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss (Smoothed)')
plt.legend()
plt.show()


# Now let's see the results on Test data; rather than just training and validation sets
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Note that in multi-class and imbalanced classification problems, test accuracy is not an ideal metric')


confusion = confusion_matrix(y_test.argmax(axis=1), predicted_labels)
print("Confusion matrix:\n{}".format(confusion))


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
    
    
y_true = y_test.argmax(axis=1)
cm_analysis(y_true, predicted_labels, filename='Example', labels=[0,1,2])