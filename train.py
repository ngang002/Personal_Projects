import numpy as np 
import recursive_neural_network as RNN 

import os
import header as HD
from tensorflow import keras
from keras.layers import Dense, Bidirectional, Dropout, LSTM
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as K

tfkl = tf.keras.layers

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel('ERROR')


def build_train_model(X_train, pixels, features, NLAYERS, \
	previous_timesteps=HD._previous_timesteps, next_timesteps=HD._next_timesteps, lr=HD._ALPHA):
	
	# First Create the neutral network
	stock_RNN = RNN.LSTMNet(pixels=pixels, features=features, learning_rate=lr, \
		input_time_steps=previous_timesteps, output_timesteps=next_timesteps)

	# Add the first layer which should have the same # of features as X_train/val 
	stock_RNN.add_initialLayer()

	# Add additional layers here
	for ln in range(NLAYERS):
		stock_RNN.add_layer(ln)

	# Add the final layers here (should add one more LSMNT w/o being bidirections)
	# and then a dense layer with the same shape as the number of timesteps in the 
	# future
	stock_RNN.add_finalLayer()
	stock_RNN.summary()


	model = stock_RNN
	opt = keras.optimizers.legacy.Adam(learning_rate=lr)
	model.compile(optimizer=opt, loss='mean_squared_error')    
	model.build(input_shape=(None, pixels, features))

	return model

def train(X_train, y_train, X_val, y_val, scaler, ticker, output_dir, HYPER_PARAMETER_DICT):

	epochs = HYPER_PARAMETER_DICT['Epochs']
	batch_size = HYPER_PARAMETER_DICT['Batch Size']
	NLAYERS = HYPER_PARAMETER_DICT['NLAYERS']
	lr = HYPER_PARAMETER_DICT['Learning Rate']
	previous_timesteps = HYPER_PARAMETER_DICT['Previous Timesteps']
	next_timesteps = HYPER_PARAMETER_DICT['Next Timesteps']

	pixels = X_train.shape[1]
	features = X_train.shape[2]
	print(f"Number of Pixels {pixels} and Features {features} going into train")

	model = build_train_model(X_train, pixels, features, NLAYERS=NLAYERS, \
						previous_timesteps=previous_timesteps, next_timesteps=next_timesteps, lr=lr)

	# Define a callback to save the model when validation loss improves
	model_file_name = output_dir+'models/'+ticker+'_model/'
	checkpoint_callback = ModelCheckpoint(
		model_file_name,  # Filepath to save the model
		monitor='val_loss',  # Metric to monitor (validation loss)
		save_best_only=True,  # Save only if the monitored metric improves
		mode='min',  # Mode for improvement (minimize validation loss)
		verbose=1,  # Verbosity (1: display messages, 0: no messages)
		save_weights_only=True
	)

	early_stopping_callback = EarlyStopping(
		monitor='val_loss',  # Monitor validation loss
		patience=HD._PATIENCE_EPOCHS,         # Stop after 10 consecutive epochs without improvement
		restore_best_weights=True  # Restore the best model weights when stopping
	)

	history_file_name = f'{output_dir}_{ticker}_history.json'
	history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
			callbacks=[checkpoint_callback, early_stopping_callback], 
			epochs=epochs, batch_size=batch_size, shuffle=True)

	with open(history_file_name, 'w') as file:
		json.dump(history.history, file)