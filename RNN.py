import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

X_train = pd.read_csv('rnn_x_train.csv')
X_test = pd.read_csv('rnn_x_test.csv')
X_train = X_train.drop(['id'], axis=1)
X_test = X_test.drop(['id'],axis=1)

numbers = set(X_train.venue.values) | set(X_test.venue.values)
d_numbers = {key:val for key, val in zip(numbers,range(len(numbers)))}

X_train.venue = X_train.venue.apply(lambda x: d_numbers[x])
X_test.venue = X_test.venue.apply(lambda x: d_numbers[x])


from sklearn.utils import shuffle
X_train = shuffle(X_train, random_state=42)
X_test = shuffle(X_test, random_state=42)

df_test = X_test.copy()
df_train = X_train.copy()

import gc
del X_train, X_test
gc.collect

X_train, y_train = df_train.iloc[:,:7], df_train.iloc[:,7:]
X_test, y_test = df_test.iloc[:,:7], df_test.iloc[:,7:]

print(X_test)
# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, 1))

# Create a list of RNN Simple Memory Cells
encoder = keras.layers.SimpleRNN(units=512, return_state=True,
                                            activation='relu',
                                             dropout=0.2)

encoder_outputs_and_states = encoder(encoder_inputs)

# Keep the states
encoder_states = encoder_outputs_and_states[1:]

# Decoder will have zeros as an input
decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder = keras.layers.SimpleRNN(units=512, return_sequences=True, return_state=True,
                                            activation='relu',
                                             dropout=0.2)

# Output state of encoder is input state of decoder
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# The output of decoder
decoder_outputs = decoder_outputs_and_states[0]

# A dense layer to apply activation function
decoder_dense = keras.layers.Dense(1)

decoder_outputs = decoder_dense(decoder_outputs)



epocsh = 100
batch_size = 256

optimizer = keras.optimizers.RMSprop(lr=0.00001)

# Create a model using the functional API provided by Keras.
# The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
# A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])


model.fit([X_train.to_numpy()[..., np.newaxis], 
                        np.zeros((y_train.to_numpy().shape[0],y_train.to_numpy().shape[1],1))] ,
                        y_train.to_numpy()[..., np.newaxis],
                        epochs=epocsh)
                  #callbacks=[keras.callbacks.EarlyStopping(verbose=0,patience=30)]
                        #verbose=0                        
                        
from tensorflow.keras.models import load_model

model.save('RNN_citation_prediction.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('RNN_citation_prediction.h5')

score, acc = model.evaluate([X_test.to_numpy()[..., np.newaxis], 
                                  np.zeros((y_test.to_numpy().shape[0],y_test.to_numpy().shape[1],1))],
                           y_test.to_numpy()[..., np.newaxis])
print('MSE:', score)
print('RMSE:', acc)


y_test_predicted = model.predict([X_test.to_numpy()[..., np.newaxis], 
                                  np.zeros((y_test.to_numpy().shape[0],y_test.to_numpy().shape[1],1))])
                                  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, explained_variance_score

print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,0], y_test_predicted[:,0])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,0], y_test_predicted[:,0])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,0], y_test_predicted[:,0])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,0], y_test_predicted[:,0])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,1], y_test_predicted[:,1])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,1], y_test_predicted[:,1])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,1], y_test_predicted[:,1])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,1], y_test_predicted[:,1])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,2], y_test_predicted[:,2])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,2], y_test_predicted[:,2])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,2], y_test_predicted[:,2])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,2], y_test_predicted[:,2])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,3], y_test_predicted[:,3])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,3], y_test_predicted[:,3])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,3], y_test_predicted[:,3])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,3], y_test_predicted[:,3])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,4], y_test_predicted[:,4])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,4], y_test_predicted[:,4])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,4], y_test_predicted[:,4])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,4], y_test_predicted[:,4])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,5], y_test_predicted[:,5])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,5], y_test_predicted[:,5])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,5], y_test_predicted[:,5])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,5], y_test_predicted[:,5])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,6], y_test_predicted[:,6])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,6], y_test_predicted[:,6])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,6], y_test_predicted[:,6])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,6], y_test_predicted[:,6])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,7], y_test_predicted[:,7])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,7], y_test_predicted[:,7])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,7], y_test_predicted[:,7])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,7], y_test_predicted[:,7])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,8], y_test_predicted[:,8])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,8], y_test_predicted[:,8])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,8], y_test_predicted[:,8])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,8], y_test_predicted[:,8])))



print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test.iloc[:,9], y_test_predicted[:,9])))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test.iloc[:,9], y_test_predicted[:,9])))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test.iloc[:,9], y_test_predicted[:,9])))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(y_test.iloc[:,9], y_test_predicted[:,9])))
                                 
