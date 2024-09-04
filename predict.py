import recursive_neural_network as RNN 
import numpy as np

import os
import header as HD
from tensorflow import keras
from keras.layers import Dense, Bidirectional, Dropout, LSTM
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as K

def build_predict_model(X_train, pixels, features, NLAYERS, \
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

def predict(full_dat, pixels, features, scaler, ticker, output_dir, HYPER_PARAMETER_DICT):

	epochs = HYPER_PARAMETER_DICT['Epochs']
	batch_size = HYPER_PARAMETER_DICT['Batch Size']
	NLAYERS = HYPER_PARAMETER_DICT['NLAYERS']
	lr = HYPER_PARAMETER_DICT['Learning Rate']
	previous_timesteps = HYPER_PARAMETER_DICT['Previous Timesteps']
	next_timesteps= HYPER_PARAMETER_DICT['Next Timesteps']


	print(f"Shape of full_dat={np.shape(full_dat)}")
	model_file_name = output_dir+'models/'+ticker+'_model/'
	# model = LSTMNet(num_of_layers, lstm_units, output_time_steps)
	
	# We have to no rebuild the model to predict
	# this is done so that we can simply just run the model rather 
	# than retraining over again for the same result
	model = build_predict_model(full_dat, pixels, features, previous_timesteps=previous_timesteps, next_timesteps=next_timesteps, NLAYERS=4, lr=lr)
	print(model_file_name)

	##############################################################################
	model.load_weights(model_file_name)
	full_scaled = full_dat # scaler.transform(full_dat.reshape((-1, features))).reshape((1, -1, features))

	X_full, y_full = [], []
	print(f"Looping over range{previous_timesteps, full_scaled.shape[0]}")
	for i in range(previous_timesteps, full_scaled.shape[0]):
		X_full.append(full_scaled[i-previous_timesteps:i,:])
		#y_full.append(full_scaled[i:i+next_timesteps, 0])
	X_full, y_full = np.array(X_full), np.array(y_full)

	print(f'Input shape {X_full.shape}')
	predictions = model.predict(X_full)


	predictions_filename = f'{output_dir}_{ticker}_predictions.dat'
	with open(predictions_filename, 'w') as file:
		json.dump(predictions.tolist(), file)
	

def predict_stack(output_dir, ticker, scaler, HYPER_PARAMETER_DICT):

	previous_timesteps = HYPER_PARAMETER_DICT['Previous Timesteps']
	next_timesteps= HYPER_PARAMETER_DICT['Next Timesteps']

	predictions_filename = f'{output_dir}_{ticker}_predictions.dat'
	with open(predictions_filename, 'r') as file:
		predictions = json.load(file)
	
	predictions = np.array(predictions)
	print(scaler.data_max_[0], scaler.data_min_[0])
	predictions_rescaled = predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
	print(np.amin(predictions_rescaled), np.amax(predictions_rescaled))

	stack = np.full(((predictions.shape[0]), previous_timesteps+predictions.shape[0]+next_timesteps-1), np.nan)
	for _ in range(previous_timesteps, predictions.shape[0]):	
		# print(f"Adding at Index {_} from {_+previous_timesteps}-{previous_timesteps+_+next_timesteps} and the length we're adding is {len(predictions[_])}")
		# print(np.shape(predictions[_]))
		stack[_,_+previous_timesteps:_+previous_timesteps+next_timesteps] = predictions_rescaled[_]

	#print(np.amin(stack), np.amax(stack))
	stack = stack.T
	mean_stack = np.nanmean(stack, axis=1)
	stack_low, stack_high = np.nanpercentile(stack, [16, 84], axis=1)
	# print(np.shape(mean_stack), np.shape(stack_low), np.shape(stack_high))
	return mean_stack, stack_low, stack_high


