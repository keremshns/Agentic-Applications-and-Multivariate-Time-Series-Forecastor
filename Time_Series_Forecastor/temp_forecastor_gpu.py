import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

#TF 2.16
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.models import load_model


print(tf.config.list_physical_devices('GPU'), "\n")
gpus = tf.config.list_physical_devices('GPU')

#ALLOCATE MINIMAL NEEDED MEMORY /  limit GPU memory growth when using gpu
'''if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)'''


def df_to_Xy(df, window_size):
    
    df_np = np.array(df)
    X = []
    y= []

    for i in range(len(df_np) - window_size):
        row = [r for r in df_np[i:i+window_size]]
        X.append(row)
        label = df_np[i+window_size][0]
        y.append(label)
    
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


csv_path = r"C:\Users\kerem\Desktop\Pavoreal_Local\Data\jena_climate_2009_2016.csv\jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)
print(df.head())
df = df[5::5] #start at 5, take every fifth

df.index = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
print(df[:10])

temp_df = pd.DataFrame({"Temperature": df["T (degC)"]})
print(temp_df)
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

print("SHAPE OF TRAIN TEST SPLITS:\n")
print("TRAIN X, TRAIN Y:")
print(X_train.shape)
print(y_train.shape)
print("VALID X, VALID Y:")
print(X_val.shape)
print(y_val.shape)
print("TEST X, TEST Y:")
print(X_test.shape)
print(y_test.shape)

#---------BUILD MODEL ARCHITECTURE------------
model = tf.keras.Sequential([
    
    tf.keras.layers.InputLayer((7,5)),
    #tf.keras.layers.Reshape((7,5,1)),
    tf.keras.layers.Conv1D(30, kernel_size=2, strides=2),         
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
check_path = r"C:\Users\kerem\Desktop\Pavoreal_Local\Time_Series_Forecastor\forecastor_mark1"

#HYPERPARAMETERS
epochs = 150

#CHECKPOINT, COMPILE, FIT

cp1 = tf.keras.callbacks.ModelCheckpoint(check_path, save_best_only=True) 
model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9), metrics=["mae"])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[cp1]) #GIVE CHECKPOINT CALLBACK

#PLOT LOSS
loss = history.history["loss"]
plt.plot(range(150), loss)
plt.show()

print("----",loss[len(loss)-1],"----") #LAST LOSS

#--------------RELOAD MODEL AND PREDICT----------------
model1 = tf.keras.models.load_model(r"C:\Users\kerem\Desktop\Pavoreal_Local\Time_Series_Forecastor\forecastor_mark1")

start = 0
end = 500
plot_predictions(model1, X_test, y_test, start, end)


