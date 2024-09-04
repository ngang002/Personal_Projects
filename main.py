import numpy as np
import header as HD 

import time
import os, sys
import yfinance as yf
from datetime import datetime, timedelta
import argparse


import preprocess_dataset as pp 
import data_exploration as de
import recursive_neural_network as RNN
from train import train
from predict import predict, predict_stack
from plot_utils import *

def main(ticker):
    
	columns = ['Close', 'Volume', 'High', 'Low', 'Open']

	validation_days = int(HD._VAL_DAYS)
	offset_today_by_days = 0
	previous_years = 5
	patience_epochs = 10

	end_date, end_date_str = HD._get_EndDate()
	start_date, start_date_str = HD._get_date(end_date, 365*previous_years)
	val_date, val_date_str = HD._get_date(end_date, validation_days)
	print(f"We are starting at {start_date} and ending at {end_date} and the {val_date}")
	print(f"We are starting at {start_date_str} and ending at {end_date_str} and the {val_date_str}")


	output_dir = f'./ticker_dirs/{ticker}/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		os.makedirs(output_dir+'models/')
		print(f"Directory '{output_dir}' created successfully.")
	else:
		print(f"Directory '{output_dir}' already exists.")


	# Download stock data with 1-day resolution
	filename = f"{output_dir}/{ticker}.csv"
	time_series_data = yf.download(ticker, start=start_date, 
	                               end=end_date, interval="1d")

	# Print the downloaded data
	print(f"Writing to {filename}")
	time_series_data.to_csv(filename)
	# print(time_series_data.info)
	# Time to explore the data and see what we are working with 
	# de.plot_df(time_series_data)


	HYPER_PARAMETER_DICT = {'Epochs': HD._EPOCHS, 
							'Batch Size': HD._BATCH_SIZE, 
							'NLAYERS': HD._NLAYERS,
							'Learning Rate': HD._ALPHA, 
							'Previous Timesteps': HD._previous_timesteps,
							'Next Timesteps': HD._next_timesteps }
	# preprocessing the stock data 
	filename = output_dir+ticker+".csv"
	X_train, y_train, X_val, y_val, full_dat, scaler = pp.preprocess_data(filename, val_date)
	train_len, val_len = len(X_train), len(X_val)
	# In this step we are going ot be training the stock datasets
	train(X_train, y_train, X_val, y_val, scaler, ticker, output_dir, HYPER_PARAMETER_DICT)
	
	# Plot the loss history of the training and validation losses 
	plot_History(output_dir, ticker)

	predict(full_dat, HYPER_PARAMETER_DICT['Previous Timesteps'], full_dat.shape[1], scaler, ticker, output_dir, HYPER_PARAMETER_DICT)
	mean_stack, stack_low, stack_high = predict_stack(output_dir, ticker, scaler, HYPER_PARAMETER_DICT)
	plot_Predictions(time_series_data, mean_stack, stack_low, stack_high, train_len, val_len, output_dir, ticker)

if __name__ == '__main__':
	
	main(sys.argv[1])