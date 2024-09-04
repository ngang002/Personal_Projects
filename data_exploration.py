import numpy as np 

import matplotlib.pyplot as plt


def plot_df(df):

	df = df.reset_index() 
	fig, ax = plt.subplots(figsize=(5,5))
	print(df)
	# ax.plot(df['Date'], df['Open'], ls =':', c='k')
	# ax.plot(df['Date'], df['High'], ls ='--', c='k')
	# ax.plot(df['Date'], df['Low'], ls ='--', c='k')
	
	ax.plot(df['Date'], df['Close'], ls ='-', c='k')
	ax.set_xlabel('Dates')
	ax.set_ylabel('Stock Price $')

	ax2 = ax.twinx()
	ax2.plot(df['Date'], df['Volume'], ls ='--', c='k')
	ax2.set_ylabel('Volume')
	plt.show()