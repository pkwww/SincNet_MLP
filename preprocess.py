import scipy.signal
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import datetime

seed = 1234
np.random.seed(seed)

# Down sample, each original is 150000
num_samples = 40000

print('***** Started processing test set at {} *****'.format(datetime.datetime.now()))
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
signals = []
down_samples = []
targets = []
for i, seg_id in enumerate(tqdm(submission.index)):
	seg = pd.read_csv('raw_data/test/' + seg_id + '.csv')
	signals = seg['acoustic_data'].values
	down_sample = scipy.signal.resample(signals, num_samples)
	down_samples.append(down_sample)
	targets.append(0.0)

with open('prepared_data/test_signals', 'wb') as outf:
	pickle.dump(down_samples, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('test_signals done.')

with open('prepared_data/test_labels', 'wb') as outf:
	pickle.dump(targets, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('test_labels done.')


print('***** Started processing training set at {} *****'.format(datetime.datetime.now()))
with open('raw_data/train.csv') as f:
	signals = []
	times = []
	down_samples = []
	targets = []
	for i, line in enumerate(f):
		if i == 0: continue
		line = line.strip()
		signal, time = line.split(',')
		signal = int(signal)
		time = float(time)
		signals.append(signal)
		times.append(time)

		if len(signals) == 150000:
			print('Processing {} of {}...'.format(str(len(down_samples)+1), str(4194)))
			down_sample = scipy.signal.resample(signals, num_samples)
			down_samples.append(down_sample)
			targets.append(times[len(times)//2])

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

print('***** Pre processing done at {} *****'.format(datetime.datetime.now()))