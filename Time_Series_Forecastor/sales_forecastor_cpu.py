import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

print(tf.__version__)

#TF 2.16
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.models import load_model

def df_to_Xy(df, window_size):
    
    df_np = np.array(df)
    X = []
    y= []

    for i in range(len(df_np) - window_size):
        row = [r for r in df_np[i:i+window_size]]
        X.append(row)
        label = df_np[i+window_size][0]
        y.append(label)
    
    print(X)
    print(y)

    print("")
    return np.array(X), np.array(y)

def plot_predictions(model, X, y, start=0, end=100):
  predictions = model.predict(X)
  
  print(predictions.flatten(),"\n\n")
  temp_preds = predictions.flatten()
  
  print(y)
  temp_actuals = y

  h = tf.keras.losses.Huber()
  print(h(temp_actuals,temp_preds))

  df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
                          'Temperature Actuals':temp_actuals,
                          })
  plt.plot(df['Temperature Predictions'][start:end])
  plt.plot(df['Temperature Actuals'][start:end])
  plt.show()

  print(df[start:end])
  
  return df[start:end]


csv_path = r"C:\Users\kerem\Desktop\Pavoreal_Local\Data\jena_climate_2009_2016.csv.zip"

df = pd.read_csv(csv_path, encoding='ISO-8859-1')
print(df.head(), "\n")
print(df)
#df = df[5::5] #start at 5, take every fifth

df.index = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
print(df[:10])

temp_df = pd.DataFrame({"Temperature": df["T (degC)"]})
temp_df["Seconds"] = temp_df.index.map(pd.Timestamp.timestamp)
print(temp_df)

day  = 60*60*24 #SECS IN DAY
year = 356.2425*day #SECS IN YEAR

temp_df["Day sin"] = np.sin(temp_df["Seconds"]*(2*np.pi/day))
temp_df["Day cos"] = np.cos(temp_df["Seconds"]*(2*np.pi/day))
temp_df["Year sin"] = np.sin(temp_df["Seconds"]*(2*np.pi/year))
temp_df["Year cos"] = np.cos(temp_df["Seconds"]*(2*np.pi/year))
temp_df = temp_df.drop("Seconds", axis=1) #DONT NEED SECONDS ANYMORE
print(temp_df)
print(len(temp_df))

WINDOW_SIZE=7
X, y = df_to_Xy(temp_df, WINDOW_SIZE)
print(X.shape)
print(y.shape)

X_train, y_train = X[:50460], y[:50460]
X_val, y_val = X[50460:67280], y[50460:67280]
X_test, y_test = X[67280:], y[67280:]

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

#---------BUILD MODEL ARCHITECTURE------------
model = tf.keras.Sequential([
    
    tf.keras.layers.InputLayer((7,5)),
    #tf.keras.layers.Reshape((7,5,1)),
    tf.keras.layers.Conv1D(30, kernel_size=2),         
    #tf.keras.layers.Lambda(lambda x : tf.keras.backend.squeeze(x,2)),  #KERNEL SIZEI DEGISTIR BELKI 2D YAPILABİLİR CONVA BAK
    tf.keras.layers.LSTM(30,return_sequences=True),
    tf.keras.layers.LSTM(30),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")

])

print(model.summary())

#--------USE LR SCHEDULE TO ARRANGE LEARNING RATE-------------
"""
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch/20))
epochs = 100
model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5,momentum=0.9), metrics=["mae"])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[lr_schedule]) #GIVE LR SCHEDULE CALLBACK

loss = history.history["loss"]
lr = history.history['lr']

plt.semilogx(lr, loss)
plt.show()
"""

#--------DEFINE REAL LR AND TRAIN REAL MODEL-------------
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    
    tf.keras.layers.InputLayer((7,5)),
    #tf.keras.layers.Reshape((7,5,1)),
    tf.keras.layers.Conv1D(30, kernel_size=2),         
    #tf.keras.layers.Lambda(lambda x : tf.keras.backend.squeeze(x,2)),  #KERNEL SIZEI DEGISTIR BELKI 2D YAPILABİLİR CONVA BAK
    tf.keras.layers.LSTM(30,return_sequences=True),
    tf.keras.layers.LSTM(30),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")

])

print(model.summary())

#PATH TO SAVE MODEL
check_path = r"C:\Users\kerem\Desktop\Pavoreal_Local\Time_Series_Forecastor\forecastor_mark1.h5"  #SAVE AS H5 TO BE ABLE TO RETRIEVE IN TF 2.16

#HYPERPARAMETERS
epochs = 150

#CHECKPOINT, COMPILE, FIT

#cp1 = tf.keras.callbacks.ModelCheckpoint(check_path, save_best_only=True) 
model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9), metrics=["mae"])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs) # CAN GIVE CHECKPOINT CALLBACK
model.save(r"C:\Users\kerem\Desktop\Pavoreal_Local\Time_Series_Forecastor\forecastor_mark1.h5", save_format="h5")

#PLOT LOSS
loss = history.history["loss"]
plt.plot(range(150), loss)
plt.show()

print("----",loss[len(loss)-1],"----")

#--------------RELOAD MODEL AND PREDICT----------------
model1 = tf.keras.models.load_model(check_path)

start = 0
end = 500
plot_predictions(model1, X_test, y_test, start, end)