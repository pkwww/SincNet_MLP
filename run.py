# run.py
# Chiehmin Wei
# University of Edinburgh 

# Mar 2019

# Description: 
# This code performs an earthquake experiment with SincNet.
 
# How to run it:
# python run.py --cfg=cfg/SincNet_TIMIT.cfg
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
from tqdm import tqdm
import numpy as np
from dnn_models import FunTimes, SincNet as CNN
from dataset import EarthquakeDataset
from tqdm import tqdm

from data_io import ReadList,read_conf,str_to_bool
from collections import OrderedDict, defaultdict
from torch.serialization import default_restore_location
import datetime

import pandas as pd

def move_to_cuda(sample):
		if torch.is_tensor(sample):
				return sample.cuda()
		elif isinstance(sample, list):
				return [move_to_cuda(x) for x in sample]
		elif isinstance(sample, dict):
				return {key: move_to_cuda(value) for key, value in sample.items()}
		else:
				return sample


def save_checkpoint(options, save_dir, model, optimizer, epoch, valid_loss):
		os.makedirs(save_dir, exist_ok=True)
		last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
		save_checkpoint.last_epoch = max(last_epoch, epoch)
		prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
		save_checkpoint.best_loss = min(prev_best, valid_loss)

		state_dict = {
				'epoch': epoch,
				'val_loss': valid_loss,
				'best_loss': save_checkpoint.best_loss,
				'last_epoch': save_checkpoint.last_epoch,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'options': options,
		}
		if valid_loss < prev_best:
				torch.save(state_dict, os.path.join(save_dir, 'checkpoint_best.pt'))
		if last_epoch < epoch:
				torch.save(state_dict, os.path.join(save_dir, 'checkpoint_last.pt'))

def load_checkpoint(save_dir, restore_file, model, optimizer):
		checkpoint_path = os.path.join(save_dir, restore_file)
		if os.path.isfile(checkpoint_path):
				state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
				model.load_state_dict(state_dict['model'])
				optimizer.load_state_dict(state_dict['optimizer'])
				save_checkpoint.best_loss = state_dict['best_loss']
				save_checkpoint.last_epoch = state_dict['last_epoch']
				print('Loaded checkpoint {}'.format(checkpoint_path))
				return state_dict


num_workers=2
clip_norm=4.0

print('Reading config file...')

# Reading cfg file
options=read_conf()

#[data]
train_src_dir=options.train_src_dir
train_tgt_dir=options.train_tgt_dir
dev_src_dir=options.dev_src_dir
dev_tgt_dir=options.dev_tgt_dir
test_src_dir=options.test_src_dir
test_tgt_dir=options.test_tgt_dir
output_folder=options.output_folder
save_dir=options.save_dir
restore_file=options.restore_file

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))

#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))

#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)
cuda=options.cuda == 'True'
patience=int(options.patience)

# Folder creation
os.makedirs(output_folder, exist_ok=True)
		
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.L1Loss()

	
# Converting context and shift in samples
wlen=4000
# wshift=int(fs*cw_shift/1000.00)

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
					'fs': fs,
					'cnn_N_filt': cnn_N_filt,
					'cnn_len_filt': cnn_len_filt,
					'cnn_max_pool_len':cnn_max_pool_len,
					'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
					'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
					'cnn_use_laynorm':cnn_use_laynorm,
					'cnn_use_batchnorm':cnn_use_batchnorm,
					'cnn_act': cnn_act,
					'cnn_drop':cnn_drop,          
					}
CNN_net = CNN(CNN_arch)

DNN1_arch = {'input_dim': CNN_net.out_dim,
					'fc_lay': fc_lay,
					'fc_drop': fc_drop, 
					'fc_use_batchnorm': fc_use_batchnorm,
					'fc_use_laynorm': fc_use_laynorm,
					'fc_use_laynorm_inp': fc_use_laynorm_inp,
					'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
					'fc_act': fc_act,
					}

DNN2_arch = {'input_dim':fc_lay[-1] ,
					'fc_lay': class_lay,
					'fc_drop': class_drop, 
					'fc_use_batchnorm': class_use_batchnorm,
					'fc_use_laynorm': class_use_laynorm,
					'fc_use_laynorm_inp': class_use_laynorm_inp,
					'fc_use_batchnorm_inp':class_use_batchnorm_inp,
					'fc_act': class_act,
					}

model = FunTimes(CNN_arch, DNN1_arch, DNN2_arch, use_sinc_net=True)

if cuda:
	cost = cost.cuda()
	model = model.cuda()

print('FunTimes: {:d} parameters'.format(sum(p.numel() for p in model.parameters())))

# Instantiate optimizer and learning rate scheduler
# optimizer = optim.Adam(model.parameters(), lr)
optimizer = optim.RMSprop(model.parameters(), lr=lr,alpha=0.95, eps=1e-8) 

# Load last checkpoint if one exists
state_dict = load_checkpoint(save_dir, restore_file, model, optimizer)
last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
	
# Track validation performance for early stopping
bad_epochs = 0
best_validate = float('inf')

train_dataset = EarthquakeDataset(train_src_dir, train_tgt_dir)
dev_dataset = EarthquakeDataset(dev_src_dir, dev_tgt_dir)

for epoch in range(last_epoch + 1, N_epochs):
	print('***** Started epoch {} at {} *****'.format(epoch, datetime.datetime.now()))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
																					shuffle=True, num_workers=num_workers, drop_last=True)
	model.train()

	stats = OrderedDict()
	stats['loss'] = 0
	stats['grad_norm'] = 0
	stats['clip'] = 0
	# Display progress
	progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

	 # Iterate over the training set
	for i, sample in enumerate(progress_bar):
			if cuda:
					sample = move_to_cuda(sample)
			if len(sample) == 0:
					continue

			output = model(sample['signals'])
			loss = cost(output, sample['target'])

			optimizer.zero_grad()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
			optimizer.step()

			# Update statistics for progress bar
			total_loss = loss.item()
			stats['loss'] += total_loss
			stats['grad_norm'] += grad_norm
			stats['clip'] += 1 if grad_norm > clip_norm else 0
			progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
															 refresh=True)
	print('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
						value / len(progress_bar)) for key, value in stats.items())))

	print('***** Finished epoch {} at {} *****'.format(epoch, datetime.datetime.now()))
	# Validation
	model.eval()

	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, 
																						shuffle=False, num_workers=num_workers, drop_last=True)

	stats = OrderedDict()
	stats['valid_loss'] = 0
	
	# Iterate over the validation set
	for i, sample in enumerate(dev_loader):
			if cuda:
					sample = move_to_cuda(sample)
			if len(sample) == 0:
					continue
			with torch.no_grad():
					# Compute loss
					output = model(sample['signals'])
					loss = cost(output, sample['target'])
			# Update tracked statistics
			stats['valid_loss'] += loss.item()
	
	# Calculate validation perplexity
	stats['valid_loss'] = stats['valid_loss']
	valid_perplexity = np.exp(stats['valid_loss'])
	
	print(
			'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
			' | valid_perplexity {:.3g}'.format(valid_perplexity))

	model.train()
	# Save checkpoints
	save_checkpoint(options, save_dir, model, optimizer, epoch, valid_perplexity)  # lr_scheduler

	# Check whether to terminate training
	if valid_perplexity < best_validate:
			best_validate = valid_perplexity
			bad_epochs = 0
	else:
			bad_epochs += 1
	if bad_epochs >= patience:
			print('No validation set improvements observed for {:d} epochs. Early stop!'.format(patience))
			break




# Test
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

test_dataset = EarthquakeDataset(test_src_dir, test_tgt_dir)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
																						shuffle=False, num_workers=num_workers, drop_last=False)

print ("Evaluate model with Test Set")
for i, sample in enumerate(tqdm(test_loader)):
		# print('Processing {} of {} tests'.format(str(i), str(len(test_loader))))
		if cuda:
				sample = move_to_cuda(sample)
		if len(sample) == 0:
				continue
		with torch.no_grad():
				output = model(sample['signals']).numpy()
		for ii, j in enumerate(range(i * batch_size, (i + 1) * batch_size)):
			submission.time_to_failure[j] = output[ii][0]

print(submission.head())
# Save
submission.to_csv('SincNet_submission.csv')
