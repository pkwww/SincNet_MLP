# run.py
# Chiehmin Wei
# University of Edinburgh 

# Mar 2019

# Description: 
# This code performs an earthquake experiment with SincNet.
 
# How to run it:
# python run.py --cfg=cfg/LSTM_Earthquake.cfg --model=Transformer_features
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
from tqdm import tqdm
import numpy as np
from dnn_models import FunTimesTransformer, FunTimesLSTM, FunTimesCNN, SincNet, ConvNet, EZConv

from dataset import EarthquakeDataset
from tqdm import tqdm

from data_io import ReadList,read_conf,str_to_bool
from collections import OrderedDict, defaultdict
from torch.serialization import default_restore_location
import datetime

import pandas as pd
from optparse import OptionParser
from adamw import AdamW

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

parser=OptionParser()
parser.add_option("--model") 
parser.add_option("--cfg")
(options,args)=parser.parse_args()
architecture = options.model

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

#[cnn]
wlen=int(options.wlen)
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))

#[transformer]
tr_embed_dim=int(options.tr_embed_dim)
tr_max_positions=int(options.tr_max_positions)
tr_pos=options.tr_pos
tr_num_layers=int(options.tr_num_layers)
tr_num_heads=int(options.tr_num_heads)
tr_filter_size=int(options.tr_filter_size)
tr_hidden_size=int(options.tr_hidden_size)
tr_dropout=float(options.tr_dropout)
tr_attention_dropout=float(options.tr_attention_dropout)
tr_relu_dropout=float(options.tr_relu_dropout)

#[lstm]
lstm_embed_dim=int(options.lstm_embed_dim)
lstm_hidden_size=int(options.lstm_hidden_size)
lstm_num_layers=int(options.lstm_num_layers)
lstm_bidirectional=options.lstm_bidirectional=='True'
lstm_dropout_in=float(options.lstm_dropout_in)
lstm_dropout_out=float(options.lstm_dropout_out)

#[dnn_before]
fc1_lay_use=options.fc1_lay_use=='True'
fc1_lay=list(map(int, options.fc1_lay.split(',')))
fc1_drop=list(map(float, options.fc1_drop.split(',')))
fc1_use_laynorm_inp=str_to_bool(options.fc1_use_laynorm_inp)
fc1_use_batchnorm_inp=str_to_bool(options.fc1_use_batchnorm_inp)
fc1_use_batchnorm=list(map(str_to_bool, options.fc1_use_batchnorm.split(',')))
fc1_use_laynorm=list(map(str_to_bool, options.fc1_use_laynorm.split(',')))
fc1_act=list(map(str, options.fc1_act.split(',')))

#[dnn_after]
fc2_lay=list(map(int, options.fc2_lay.split(','))) + [1]
fc2_drop=list(map(float, options.fc2_drop.split(','))) + [0.0]
fc2_use_laynorm_inp=str_to_bool(options.fc2_use_laynorm_inp) 
fc2_use_batchnorm_inp=str_to_bool(options.fc2_use_batchnorm_inp)
fc2_use_batchnorm=list(map(str_to_bool, options.fc2_use_batchnorm.split(','))) + [False]
fc2_use_laynorm=list(map(str_to_bool, options.fc2_use_laynorm.split(','))) + [False]
fc2_act=list(map(str, options.fc2_act.split(','))) + ['softplus']


#[optimization]
optimizer_to_use=options.optimizer
weight_decay=float(options.weight_decay)
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
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


if fc1_lay_use:
	MLP_before = {'input_dim': wlen,
				'fc_lay': fc1_lay,
				'fc_drop': fc1_drop, 
				'fc_use_batchnorm': fc1_use_batchnorm,
				'fc_use_laynorm': fc1_use_laynorm,
				'fc_use_laynorm_inp': fc1_use_laynorm_inp,
				'fc_use_batchnorm_inp':fc1_use_batchnorm_inp,
				'fc_act': fc1_act,
				}
else:
	MLP_before = None

if architecture == 'Transformer_features':
	MLP_after = {'input_dim': tr_hidden_size,
			'fc_lay': fc2_lay,
			'fc_drop': fc2_drop, 
			'fc_use_batchnorm': fc2_use_batchnorm,
			'fc_use_laynorm': fc2_use_laynorm,
			'fc_use_laynorm_inp': fc2_use_laynorm_inp,
			'fc_use_batchnorm_inp':fc2_use_batchnorm_inp,
			'fc_act': fc2_act,
			}
	model = FunTimesTransformer(MLP_before, MLP_after,
		tr_embed_dim, tr_max_positions, tr_pos, tr_num_layers,
		tr_num_heads, tr_filter_size, tr_hidden_size, tr_dropout, 
		tr_attention_dropout, tr_relu_dropout, cuda)

elif architecture == 'LSTM_raw':
	MLP_after = {'input_dim': lstm_hidden_size + lstm_hidden_size * lstm_bidirectional,
			'fc_lay': fc2_lay,
			'fc_drop': fc2_drop, 
			'fc_use_batchnorm': fc2_use_batchnorm,
			'fc_use_laynorm': fc2_use_laynorm,
			'fc_use_laynorm_inp': fc2_use_laynorm_inp,
			'fc_use_batchnorm_inp':fc2_use_batchnorm_inp,
			'fc_act': fc2_act,
			}
	model = FunTimesLSTM(MLP_before, MLP_after, lstm_embed_dim, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, lstm_dropout_in, lstm_dropout_out, raw=True)

elif architecture == 'LSTM_features':
	MLP_after = {'input_dim': lstm_hidden_size + lstm_hidden_size * lstm_bidirectional,
			'fc_lay': fc2_lay,
			'fc_drop': fc2_drop, 
			'fc_use_batchnorm': fc2_use_batchnorm,
			'fc_use_laynorm': fc2_use_laynorm,
			'fc_use_laynorm_inp': fc2_use_laynorm_inp,
			'fc_use_batchnorm_inp':fc2_use_batchnorm_inp,
			'fc_act': fc2_act,
			}
	model = FunTimesLSTM(MLP_before, MLP_after, lstm_embed_dim, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, lstm_dropout_in, lstm_dropout_out, raw=False)

else:
	if architecture in ['SincNet_raw', 'CNN_raw']:
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
		if architecture == 'SincNet_raw':
			CNN_net = SincNet(CNN_arch)
		else:
			CNN_net = ConvNet(CNN_arch)

		MLP_after = {'input_dim': CNN_net.out_dim,
					'fc_lay': fc_lay,
					'fc_drop': fc_drop, 
					'fc_use_batchnorm': fc_use_batchnorm,
					'fc_use_laynorm': fc_use_laynorm,
					'fc_use_laynorm_inp': fc_use_laynorm_inp,
					'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
					'fc_act': fc_act,
					}
		model = FunTimesCNN(MLP_before, MLP_after, CNN_arch, use_sinc_net=architecture=='SincNet_raw')

	elif architecture == 'CNN_features':
		CNN_arch = {'input_dim': fc1_lay[-1],
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
		MLP_after = {'input_dim': lstm_hidden_size + lstm_hidden_size * lstm_bidirectional,
			'fc_lay': fc2_lay,
			'fc_drop': fc2_drop, 
			'fc_use_batchnorm': fc2_use_batchnorm,
			'fc_use_laynorm': fc2_use_laynorm,
			'fc_use_laynorm_inp': fc2_use_laynorm_inp,
			'fc_use_batchnorm_inp':fc2_use_batchnorm_inp,
			'fc_act': fc2_act,
			}
		model = EZConv(MLP_before, MLP_after, CNN_arch)
	
	else:
		print('Received: {}'.format(architecture))
		raise Exception('Model must be one of: Transformer_features, LSTM_raw, LSTM_features, CNN_raw, CNN_features, SincNet_raw')
		

if cuda:
	cost = cost.cuda()
	model = model.cuda()

print('FunTimes: {:d} parameters'.format(sum(p.numel() for p in model.parameters())))

# Instantiate optimizer and learning rate scheduler
if optimizer_to_use == 'AMSGrad':
	optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
elif optimizer_to_use == 'AdamW':
	optimizer = AdamW(model.parameters(), lr, weight_decay=weight_decay)
elif optimizer_to_use == 'Adam':
	optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
elif optimizer_to_use == 'RMSProp':
	optimizer = optim.RMSprop(model.parameters(), lr,alpha=0.95, eps=1e-8, weight_decay=weight_decay) 
else:
	print('Optimizer selected: {}'.format(optimizer_to_use))
	raise Exception('Optimizer once be one of: AMSGrad, AdamW, Adam, RMSProp')

# Load last checkpoint if one exists
state_dict = None
state_dict = load_checkpoint(save_dir, restore_file, model, optimizer)
last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
	
# Track validation performance for early stopping
bad_epochs = 0
best_validate = float('inf')

train_dataset = EarthquakeDataset(train_src_dir, train_tgt_dir)
dev_dataset = EarthquakeDataset(dev_src_dir, dev_tgt_dir)

print('***** Started training at {} *****'.format(datetime.datetime.now()))
for epoch in range(last_epoch + 1, N_epochs):
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

			signals = sample['signals']
			#signals = sample['signals'].unsqueeze(-1)
			output = model(signals)
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

	# Validation
	model.eval()

	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

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
					#signals = sample['signals'].unsqueeze(-1)
					signals = sample['signals']
					output = model(signals)
					loss = cost(output, sample['target'])
			# Update tracked statistics
			stats['valid_loss'] += loss.item()
	
	# Calculate validation perplexity
	stats['valid_loss'] = stats['valid_loss'] / len(dev_loader)
	valid_perplexity = np.exp(stats['valid_loss'])
	
	print(
			'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
			' | valid_perplexity {:.3g}'.format(valid_perplexity))

	model.train()
	# Save checkpoints
	save_checkpoint(options, save_dir, model, optimizer, epoch, valid_perplexity)  # lr_scheduler

	# Check whether to terminate training
	if valid_perplexity <= best_validate:
			best_validate = valid_perplexity
			bad_epochs = 0
	else:
			bad_epochs += 1
	if bad_epochs >= patience:
			print('No validation set improvements observed for {:d} epochs. Early stop!'.format(patience))
			print('Best perplexity is {}'.format(best_validate))
			break




# Test
load_checkpoint(save_dir, 'checkpoint_best.pt', model, optimizer)
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

test_dataset = EarthquakeDataset(test_src_dir, test_tgt_dir)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

print('***** Finished training at {} *****'.format(datetime.datetime.now()))
print ("Evaluate model with Test Set")
for i, sample in enumerate(tqdm(test_loader)):
		# print('Processing {} of {} tests'.format(str(i), str(len(test_loader))))
		if cuda:
				sample = move_to_cuda(sample)
		if len(sample) == 0:
				continue
		with torch.no_grad():
				#signals = sample['signals'].unsqueeze(-1)
				signals = sample['signals']
				output = model(signals).cpu().numpy()
		for ii, j in enumerate(range(i * batch_size, (i + 1) * batch_size)):
			submission.time_to_failure[j] = output[ii][-1]

print(submission.head())
# Save
outfile = '{}_submission.csv'.format(architecture)
submission.to_csv(outfile)
print ("Prediction saved as {}".format(outfile))