import scipy.signal
import numpy as np
import pickle

seed = 1234
np.random.seed(seed)

# Each segment is 37.5 ms, downsample to 1 ms (resulting 4000-dim)
fs = 4000000
cw_len = 1
num_samples = int(fs * cw_len / 1000.0)

with open('train.csv') as f:
	signals = []
	down_samples = []
	targets = []
	for i, line in enumerate(f):
		if i == 0: continue
		line = line.strip()
		signal, time = line.split(',')
		signal = int(signal)
		time = float(time)
		signals.append(signal)

		if len(signals) == 150000:
			print('Processing {} of {}...'.format(str(len(down_samples)), str(4194)))
			down_sample = scipy.signal.resample(signals, num_samples)
			down_samples.append(down_sample)
			targets.append(time)

			signals = []

	down_samples = np.array(down_samples)
	targets = np.array(targets)

print('Reading done.')

permutation = np.random.permutation(down_samples.shape[0])
down_samples = down_samples[permutation]
targets = targets[permutation]

with open('prepared_data/train_signals', 'wb') as outf:
	train_signals = down_samples[:int(0.9 * len(down_samples))]
	pickle.dump(train_signals, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('train_signals done.')


with open('prepared_data/train_labels', 'wb') as outf:
	train_labels = targets[:int(0.9 * len(down_samples))]
	pickle.dump(train_labels, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('train_labels done.')

with open('prepared_data/dev_signals', 'wb') as outf:
	dev_signals = down_samples[int(0.9 * len(down_samples)):]
	pickle.dump(dev_signals, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('dev_signals done.')

with open('prepared_data/dev_labels', 'wb') as outf:
	dev_labels = targets[int(0.9 * len(down_samples)):]
	pickle.dump(dev_labels, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('dev_labels done.')




# for i in range(0, 629145481, 150000 - 7500):


# 1.100000091014408e-09 sec

# 150000 * 1.100000091014408e-09 sec = 1.6500001365216121 * e-04 sec = 16.5 ms


# For earthquake prediction: split into chunks of 16.5 ms (150000 samples) with 7500 samples overlap

# wc -l train.csv                 ~/Downloads(master✗)@Jimmys-MacBook-Air.local
# 629145481 train.csv



# 4194 train
# 2624 test

# scipy.signal.resample