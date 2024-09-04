import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import json
import header as HD

def plot_History(output_dir, ticker):


	fig, ax = plt.subplots(1, 1, figsize=(10, 8))
	fig.subplots_adjust(wspace=0., hspace=0)
	
	history_file_name = f'{output_dir}_{ticker}_history.json'	


	with open(history_file_name, 'r') as file:
		history = json.load(file)
	
	epochs = np.arange(len(history['loss']))
	ax.plot(epochs, history['loss'], '-b', label='Training Loss')
	ax.plot(epochs, history['val_loss'], '-r', label='Validation Loss')
	ax.set_xlabel('Epochs', fontsize=18), ax.set_ylabel('Loss', fontsize=18)
	ax.set_yscale('log')
	fig.savefig(f'{output_dir}/loss_funtion.png')


def plot_Predictions(original_dat, mean, low, upp, train_len, val_len, output_dir, ticker):


	fig, ax = plt.subplots(1, 1, figsize=(10, 8))
	fig.subplots_adjust(wspace=0., hspace=0)
	


	original_opening = original_dat['Open'].values
	original_time = np.arange(len(original_opening))

	mean = mean
	mean_time = np.arange(len(mean)) # np.arange(HD._previous_timesteps, HD._previous_timesteps+len(mean))

	print(np.shape(low), np.shape(upp))
	epochs = np.arange(HD._EPOCHS)
	ax.plot(original_time, original_opening, '-k')
	
	print(train_len, val_len)
	print(f"Number of days in original_dat={len(original_dat)} \t in Mean Stock Stack={len(mean)+120}")
	ax.plot(mean_time[:train_len], mean[:train_len], '-r')
	ax.plot(mean_time[train_len:train_len+val_len], mean[train_len:train_len+val_len], '-y')
	ax.plot(mean_time[train_len+val_len:], mean[train_len+val_len:], '-g')
	
	ax.fill_between(mean_time[:train_len], y1=low[:train_len], y2=upp[:train_len], color='r', alpha=0.4)
	ax.fill_between(mean_time[train_len:train_len+val_len], \
						y1=low[train_len:train_len+val_len], y2=upp[train_len:train_len+val_len], color='y', alpha=0.4)
	ax.fill_between(mean_time[train_len+val_len:], y1=low[train_len+val_len:], y2=upp[train_len+val_len:], color='g', alpha=0.4)
	

	ax.set_xlabel('Time [days]', fontsize=18), ax.set_ylabel('Opening Stock Price', fontsize=18)
	fig.savefig(f'{output_dir}/{ticker}_prediction.png')