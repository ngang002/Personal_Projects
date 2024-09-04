import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import multiprocessing
from  datetime import datetime
import header as HD

def preprocess_data(filename, validation_date):

	"""
		filename: name of the csv in which the stock data is held 
		validation_dates: dates beyond which we are using the validate the data 

	"""
	stock_df = pd.read_csv(filename)
	stock_df = stock_df[['Date', 'Open', 'Close', 'Low', 'High', 'Volume']]
	stock_df['Date'] = pd.to_datetime(stock_df['Date'])
	columns = stock_df.columns
	features = len(columns)-1
	print(f"Number of Features: {features}")
	
	scaler = MinMaxScaler(feature_range=(0., 1.))
	stock_df.loc[:, stock_df.columns != 'Date'] = scaler.fit_transform(stock_df.loc[:, stock_df.columns != 'Date'])
	print()
	print(stock_df.info)

	# print(f"Validation date is: {validation_date}")
	train_dat = stock_df[(stock_df['Date'] < validation_date)]
	val_dat   = stock_df[(stock_df['Date'] >= validation_date)]
	print(f"Number of timesteps in train: {len(train_dat)} and validation: {len(val_dat)}")
	# print(train_stock_df)
	full_dat = stock_df.set_index('Date').values
	train_dat = train_dat.set_index('Date').values
	val_dat = val_dat.set_index('Date').values
	print(f"Shape of train_dat: {np.shape(train_dat)} \n val_dat: {np.shape(val_dat)} \n full_dat: {np.shape(full_dat)}")

	X_train, X_val, y_train, y_val = [], [], [], []
	# We want to segment the "lifespan" of a stock into sub-sections 
	# so that we can train on mutliple different time steps
	# this will require taking the many segments
	dt_prev = HD._previous_timesteps
	dt_next = HD._next_timesteps
 
	full_dat = full_dat[:,:]
	for i in range(HD._previous_timesteps, train_dat.shape[0]-dt_next):
		X_train.append(train_dat[i-dt_prev:i, :])
		y_train.append(train_dat[i:i+dt_next, 0])

	for i in range(HD._previous_timesteps, val_dat.shape[0]-dt_next):        
		X_val.append(val_dat[i-dt_prev:i, :])
		y_val.append(val_dat[i:i+dt_next, 0])

	print(np.shape(X_train), np.shape(X_val), np.shape(y_train), np.shape(y_val), np.array(full_dat))

	return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(full_dat), scaler

