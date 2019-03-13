import scipy.signal
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import datetime
import os
from scipy.stats import kurtosis, skew

USE_MIDDLE_AS_TARGET = True
USE_FEATURES = True
FOLDER_NAME='sliding_features_mid'

def extract_features(z):
	z = np.array(z)
	return np.array([z.mean(axis=0), 
				 z.min(axis=0),
				 z.max(axis=0),
				 z.std(axis=0),
				 np.quantile(z, 0.01, axis=0),
				 np.quantile(z, 0.05, axis=0),
				 np.quantile(z, 0.95, axis=0),
				 np.quantile(z, 0.99, axis=0),]
				 # kurtosis(z, bias=False),
				 # np.var(z),
				 # skew(z),
				 # np.median(z),
				 # np.mean(np.abs(z - z.mean())),
				 # np.abs(z).mean(),
				 # np.abs(z).std()]
				 )


seed = 1234
np.random.seed(seed)

# Down sample, each original is 150000
num_samples = 40000
step_interval = 1000 # 150000 / 1000 => 150 time stesp in RNN

os.makedirs('prepared_data/' + FOLDER_NAME, exist_ok=True)

print('***** Started processing test set at {} *****'.format(datetime.datetime.now()))
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
signals = []
down_samples = []
targets = []
for i, seg_id in enumerate(tqdm(submission.index)):
	seg = pd.read_csv('raw_data/test/' + seg_id + '.csv')
	signals = seg['acoustic_data'].values

	if USE_FEATURES:
		down_sample = []
		for j in range(0, len(signals), step_interval):
			feature = extract_features(signals[j:j+step_interval])
			down_sample.append(feature)

	else:
		down_sample = signals
		# down_sample = scipy.signal.resample(signals, num_samples)
	down_samples.append(down_sample)
	targets.append(0.0)
down_samples = np.array(down_samples)
targets = np.array(targets)
print(down_samples.shape)
print(targets.shape)

with open('prepared_data/'+FOLDER_NAME+'/test_signals', 'wb') as outf:
	pickle.dump(down_samples, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('test_signals done.')

with open('prepared_data/'+FOLDER_NAME+'/test_labels', 'wb') as outf:
	pickle.dump(targets, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('test_labels done.')


print('***** Started processing training set at {} *****'.format(datetime.datetime.now()))

# Juan: Grabbing segments of 150k sliding them 4095 elements at a time
down_samples = []
targets = []

for enumerator, offset in enumerate(range(0, 150000, 3750)):
	print('***** Processing {} out of {} at {} *****'.format(enumerator, len(range(0, 150000, 3750)), datetime.datetime.now()))
	have_subtracted = False

	with tqdm(total=4194) as pbar:
		with open('raw_data/train.csv') as f:
			signals = []
			times = []
			
			for i, line in enumerate(f):
				if i == 0: continue
				line = line.strip()
				signal, time = line.split(',')
				signal = int(signal)
				time = float(time)
				signals.append(signal)
				times.append(time)

				if not have_subtracted and len(signals) == offset:
					have_subtracted = True
					signals = []
					times = []

				if len(signals) == 150000:
					if USE_FEATURES:
						down_sample = []
						for j in range(0, len(signals), step_interval):
							feature = extract_features(signals[j:j+step_interval])
							down_sample.append(feature)
					else:
						down_sample = signals
						#down_sample = scipy.signal.resample(signals, num_samples)

					down_samples.append(down_sample)
					if USE_MIDDLE_AS_TARGET:
						targets.append(times[len(times)//2])
					else:
						targets.append(times[-1])

					signals = []
					times = []
					pbar.update(1)



down_samples = np.array(down_samples)
targets = np.array(targets)

print('Reading done.')
print(down_samples.shape)
print(targets.shape)

permutation = np.random.permutation(down_samples.shape[0])
down_samples = down_samples[permutation]
targets = targets[permutation]

with open('prepared_data/'+FOLDER_NAME+'/train_signals', 'wb') as outf:
	train_signals = down_samples[:int(0.9 * len(down_samples))]
	pickle.dump(train_signals, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('train_signals done.')


with open('prepared_data/'+FOLDER_NAME+'/train_labels', 'wb') as outf:
	train_labels = targets[:int(0.9 * len(down_samples))]
	pickle.dump(train_labels, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('train_labels done.')

with open('prepared_data/'+FOLDER_NAME+'/dev_signals', 'wb') as outf:
	dev_signals = down_samples[int(0.9 * len(down_samples)):]
	pickle.dump(dev_signals, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('dev_signals done.')

with open('prepared_data/'+FOLDER_NAME+'/dev_labels', 'wb') as outf:
	dev_labels = targets[int(0.9 * len(down_samples)):]
	pickle.dump(dev_labels, outf, protocol=pickle.HIGHEST_PROTOCOL)		
print('dev_labels done.')

print('***** Pre processing done at {} *****'.format(datetime.datetime.now()))