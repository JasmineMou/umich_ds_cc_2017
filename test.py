train_path = './train_data'

## get a list of unique labels
import os
files = os.listdir(train_path)
labels = set(f.split('_')[0] for f in files)
# print(labels) # {'BbClarinet', 'TenorTrombone', 'Cello', 'BassTrombone', 'Xylophone', 'BassClarinet', 'Viola', 'Marimba', 'Violin', 'EbClarinet'}

from pylab import *
from scipy.io import wavfile


## plot the wave
import matplotlib.pyplot as plt

def viz(data, fn, freqArray_kHz, power_dB):
	plt.figure()

	plt.subplot(121)
	plt.plot(data)
	plt.xlabel("Time (ms)")
	plt.ylabel("Amplitude")
	plt.title(fn)

	plt.subplot(122)
	plt.plot(freqArray_kHz, power_dB, color='k')
	plt.xlabel('Frequency (kHz)')
	plt.ylabel('Power (dB)')
	plt.title(fn)

	plt.show() ## features: frequency of low and high; max and min amplitude.
	plt.close()


## generate train data
import random
def cross_validate(l):
	'''
		Given the label name, return [[train_data], [test_data]].
		train:test = 4:1. 
	'''
	l_files = [f for f in files if f.split('_')[0]==l]
	random.shuffle(l_files)
	n_test = int(len(l_files)/5)
	l_train = l_files[0:-n_test] 
	l_test = l_files[-n_test:]

	return l_train, l_test

# # test cross_validate()
# l = 'Xylophone'
# cross_validate(l)

def freq_power(rate, data):
	'''
		Given the data, return the freqArray_kHz, power_dB. 
		Applied with Fast Fourier Transform. 
	'''
	n = len(data)
	p = fft(data)
	n_uniq_pts = int(ceil((n+1)/2.0))
	power = (abs(p[0:n_uniq_pts])/float(n))**2
	if(n%2>0):
		## odd number of points fft
		power[1:len(power)] = power[1:len(power)]*2
	else:
		## even number of points fft
		power[1:len(power)-1] = power[1:len(power)-1]*2

	freqArray_kHz = arange(0, n_uniq_pts, 1.0)*(rate/n)/1000
	power_dB = 10*log10(power)

	return freqArray_kHz, power_dB

## 
import pandas as pd
def features(fn):
	'''
		Given a single wav file, return its features, and plot.
		Possible features: amp_range, freq_range.
	'''
	rate, data = wavfile.read(os.path.join(train_path, fn))
	duration = data.shape[0]/rate

	freqArray_kHz, power_dB = freq_power(rate, data)
	df_freq_power = pd.DataFrame({'freqArray_kHz':freqArray_kHz, 'power_dB':power_dB})

	return rate, data, duration, df_freq_power

	# print(fn, data.shape, duration)
	# viz(data, fn, freqArray_kHz, power_dB)


## merge to generate a average signal for each label

for label in labels:
	l_files = [f for f in files if f.split('_')[0]==label]
	dfs = []
	for fn in l_files:
		rate, data, duration, df_freq_power = features(fn)
		dfs.append(df_freq_power)
	df_concat = pd.concat(dfs)
	by_row_index = df_concat.groupby(df_concat.index)
	df_means = by_row_index.mean()
	df_means.sort(['freqArray_kHz'])
	print(df_means)

	df_means.plot(x='freqArray_kHz', y='power_dB')
	plt.title(label)
	plt.savefig("{}.png".format(label))
	# plt.show()
	# plt.close()




