# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,KFold,RandomizedSearchCV
pd.set_option('display.max_columns',None)
x=pd.read_csv("/kaggle/input/feat-engg/train_mod.csv")
y=pd.read_csv("/kaggle/input/feat-engg/target_mod.csv")
test=pd.read_csv("/kaggle/input/feat-engg/test_mod.csv")
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

rus=RandomUnderSampler(random_state=0)
smote=SMOTE(sampling_strategy='minority')
X_sm,y_sm=rus.fit_resample(x,y)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X_sm,y_sm,test_size=0.2,stratify=y_sm, random_state=1)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
X_train.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train.shape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train.shape, X_test.shape
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

epochs = 20
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)
def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
plot_learningCurve(history, 100)
y_pred = model.predict(X_train)

y_pred
ans=[]
    
for i in y_pred:
    if(i>=0.5):
        ans.append(1)
    else:
        ans.append(0)
ans
f1_score(y_train,ans)
ans=[]
ypred = model.predict(X_test)
for i in ypred:
    if(i>=0.5):
        ans.append(1)
    else:
        ans.append(0)
len(ans)
f1_score(y_test,ans)
test=np.array(test)
test = test.reshape(test.shape[0], test.shape[1], 1)
test.shape
ans=[]
ypred=model.predict(test)
for i in ypred:
    if(i>=0.5):
        ans.append(1)
    else:
        ans.append(0)
len(ans)
pd.DataFrame(ans).to_csv('nn_pred.csv')
epochs = 100
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
plot_learningCurve(history, epochs)
y_pred = model.predict(X_train)
ans=[]
for i in y_pred:
    if(i>0.5):
        ans.append(1)
    else:
        ans.append(0)
f1_score(y_train,ans)
y_pred = model.predict(X_test)
ans=[]
for i in y_pred:
    if(i>0.5):
        ans.append(1)
    else:
        ans.append(0)
f1_score(y_test,ans)
ans=[]
ypred=model.predict(test)
for i in ypred:
    if(i>=0.5):
        ans.append(1)
    else:
        ans.append(0)
len(ans)
pd.DataFrame(ans).to_csv('nn_pred.csv')


from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv1D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape= X_train[0].shape
    ),
    keras.layers.Conv1D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(1, activation='sigmoid')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',project_name="Credit Card Fraud")
tuner_search.search(X_train,y_train,epochs=3,validation_split=0.1)


