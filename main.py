import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import keras
from keras import layers
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("dataRefined.csv")
wins = df.W
df.drop("W", axis=1, inplace=True)
df.drop("Season", axis=1, inplace=True)
df.drop("Team", axis=1, inplace=True)
df.drop("Rk", axis=1, inplace=True)

# print(wins)
# print(df.head)

x_train, x_test, y_train, y_test = train_test_split(df, wins, test_size=0.2, random_state=30)

print(x_train.shape)

inputs = keras.Input(shape=(13,),name="InputLayer")

kernal_initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=10)
bias_initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=10)

x = keras.layers.Dense(10, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(inputs)
x = keras.layers.Dense(64, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(x)
x = keras.layers.Dense(64, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(x)
x = keras.layers.Dense(64, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu', kernel_initializer=kernal_initializer, bias_initializer=bias_initializer)(x)
outputs = keras.layers.Dense(17, activation="softmax")(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=1000)

history = model.evaluate(x_test, y_test)

model.save('./model.keras')