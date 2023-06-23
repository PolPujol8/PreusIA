import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Carrega dades
house_df = pd.read_csv("precios_hogares.csv")

#Visualització
sns.scatterplot(data=house_df, x='sqft_living', y='price')
#plt.show()


#Correlació
f, ax = plt.subplots(figsize=(20,20))
numeric_columns = house_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_columns.corr(), annot=True)
#plt.show()

#Limpieza 
selected_fetures = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

X = numeric_columns[selected_fetures]
Y = numeric_columns['price']

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
X_scaled= scaler.fit_transform(X)

#Normalizando output
y = Y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)


#Entrenament
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, Y_tests = train_test_split(X_scaled, y_scaled, test_size=0.25)

#Definir model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape = (7, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
          
#model.summary()
                    
model.compile(optimizer='Adam', loss='mean_squared_error')
              
epochs_hist = model.fit(x_train, y_train, epochs=100, batch_size = 50, validation_split = 0.2)

#Evaluando modelo
epochs_hist.history.keys()

#Gràfico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progres del model durent entrenament')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation loss')
plt.legend(['Training loss', 'Validation loss'])

#Predicció
#Definir Casa amb inputs
x_tests_1 = np.array([[ 4, 4, 1960, 5000, 1, 2000, 3000]])

scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(x_tests_1)

y_predict_1 = model.predict(X_test_scaled_1)

#Revertir escalat per apreciar el preu correctament
y_predict_1= scaler.inverse_transform(y_predict_1)         

                    