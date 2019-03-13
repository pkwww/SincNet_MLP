import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math

def flip(x, dim):
	xsize = x.size()
	dim = x.dim() + dim if dim < 0 else dim
	x = x.contiguous()
	x = x.view(-1, *xsize[dim:])
	x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
										-1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
	return x.view(xsize)


def sinc(band,t_right):
	y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
	y_left= flip(y_right,0)

	y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

	return y
		

class SincConv_fast(nn.Module):
	"""Sinc-based convolution
	Parameters
	----------
	in_channels : `int`
			Number of input channels. Must be 1.
	out_channels : `int`
			Number of filters.
	kernel_size : `int`
			Filter length.
	sample_rate : `int`, optional
			Sample rate. Defaults to 16000.
	Usage
	-----
	See `torch.nn.Conv1d`
	Reference
	---------
	Mirco Ravanelli, Yoshua Bengio,
	"Speaker Recognition from raw waveform with SincNet".
	https://arxiv.org/abs/1808.00158
	"""

	@staticmethod
	def to_mel(hz):
		return 2595 * np.log10(1 + hz / 700)

	@staticmethod
	def to_hz(mel):
		return 700 * (10 ** (mel / 2595) - 1)

	def __init__(self, out_channels, kernel_size, sample_rate, in_channels=1,
							 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

		super(SincConv_fast,self).__init__()

		if in_channels != 1:
			#msg = (f'SincConv only support one input channel '
			#       f'(here, in_channels = {in_channels:d}).')
			msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
			raise ValueError(msg)

		self.out_channels = out_channels
		self.kernel_size = kernel_size
		
		# Forcing the filters to be odd (i.e, perfectly symmetrics)
		if kernel_size%2==0:
			self.kernel_size=self.kernel_size+1
				
		self.stride = stride
		self.padding = padding
		self.dilation = dilation

		if bias:
			raise ValueError('SincConv does not support bias.')
		if groups > 1:
			raise ValueError('SincConv does not support groups.')

		self.sample_rate = sample_rate
		self.min_low_hz = min_low_hz
		self.min_band_hz = min_band_hz

		# initialize filterbanks such that they are equally spaced in Mel scale
		low_hz = 30
		high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

		mel = np.linspace(self.to_mel(low_hz),
											self.to_mel(high_hz),
											self.out_channels + 1)
		hz = self.to_hz(mel)
		

		# filter lower frequency (out_channels, 1)
		self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

		# filter frequency band (out_channels, 1)
		self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

		# Hamming window
		#self.window_ = torch.hamming_window(self.kernel_size)
		n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
		self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


		# (kernel_size, 1)
		n = (self.kernel_size - 1) / 2.0
		self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes




	def forward(self, waveforms):
		"""
		Parameters
		----------
		waveforms : `torch.Tensor` (batch_size, 1, n_samples)
				Batch of waveforms.
		Returns
		-------
		features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
				Batch of sinc filters activations.
		"""

		self.n_ = self.n_.to(waveforms.device)

		self.window_ = self.window_.to(waveforms.device)

		low = self.min_low_hz  + torch.abs(self.low_hz_)
		
		high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
		band=(high-low)[:,0]
		
		f_times_t_low = torch.matmul(low, self.n_)
		f_times_t_high = torch.matmul(high, self.n_)

		band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
		band_pass_center = 2*band.view(-1,1)
		band_pass_right= torch.flip(band_pass_left,dims=[1])
		
		
		band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

		
		band_pass = band_pass / (2*band[:,None])
		

		self.filters = (band_pass).view(
				self.out_channels, 1, self.kernel_size)

		return F.conv1d(waveforms, self.filters, stride=self.stride,
										padding=self.padding, dilation=self.dilation,
										 bias=None, groups=1) 


				
				
class sinc_conv(nn.Module):

	def __init__(self, N_filt,Filt_dim,fs):
		super(sinc_conv,self).__init__()

		# Mel Initialization of the filterbanks
		low_freq_mel = 80
		high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
		mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
		f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
		b1=np.roll(f_cos,1)
		b2=np.roll(f_cos,-1)
		b1[0]=30
		b2[-1]=(fs/2)-100
						
		self.freq_scale=fs*1.0
		self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
		self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

		
		self.N_filt=N_filt
		self.Filt_dim=Filt_dim
		self.fs=fs
				

	def forward(self, x):
			
		filters=Variable(torch.zeros((self.N_filt,self.Filt_dim))).cuda()
		N=self.Filt_dim
		t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()
		
		
		min_freq=50.0;
		min_band=50.0;
		
		filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
		filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)
	 
		n=torch.linspace(0, N, steps=N)

		# Filter window (hamming)
		window=0.54-0.46*torch.cos(2*math.pi*n/N);
		window=Variable(window.float().cuda())

		
		for i in range(self.N_filt):
										
				low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
				low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
				band_pass=(low_pass2-low_pass1)

				band_pass=band_pass/torch.max(band_pass)

				filters[i,:]=band_pass.cuda()*window

		out=F.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim))

		return out
		

def act_fun(act_type):
	if act_type=="softplus":
		return nn.Softplus()

	if act_type=="relu":
		return nn.ReLU()
						
	if act_type=="tanh":
		return nn.Tanh()
						
	if act_type=="sigmoid":
		return nn.Sigmoid()
					 
	if act_type=="leaky_relu":
		return nn.LeakyReLU(0.2)
						
	if act_type=="elu":
		return nn.ELU()
										 
	if act_type=="softmax":
		return nn.LogSoftmax(dim=1)
				
	if act_type=="linear":
		return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
						
						
class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm,self).__init__()
		self.gamma = nn.Parameter(torch.ones(features))
		self.beta = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MLP(nn.Module):
	def __init__(self, options):
		super(MLP, self).__init__()
		
		self.input_dim=int(options['input_dim'])
		self.fc_lay=options['fc_lay']
		self.fc_drop=options['fc_drop']
		self.fc_use_batchnorm=options['fc_use_batchnorm']
		self.fc_use_laynorm=options['fc_use_laynorm']
		self.fc_use_laynorm_inp=options['fc_use_laynorm_inp']
		self.fc_use_batchnorm_inp=options['fc_use_batchnorm_inp']
		self.fc_act=options['fc_act']
		
	 
		self.wx  = nn.ModuleList([])
		self.bn  = nn.ModuleList([])
		self.ln  = nn.ModuleList([])
		self.act = nn.ModuleList([])
		self.drop = nn.ModuleList([])
	 

	 
		# input layer normalization
		if self.fc_use_laynorm_inp:
			self.ln0=LayerNorm(self.input_dim)
			
		# input batch normalization    
		if self.fc_use_batchnorm_inp:
			self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
			 
			 
		self.N_fc_lay=len(self.fc_lay)
				 
		current_input=self.input_dim
		
		# Initialization of hidden layers
		
		for i in range(self.N_fc_lay):
			# dropout
			self.drop.append(nn.Dropout(p=self.fc_drop[i]))
		
			# activation
			self.act.append(act_fun(self.fc_act[i]))
			 
			 
			add_bias=True
			 
			# layer norm initialization
			self.ln.append(LayerNorm(self.fc_lay[i]))
			self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
			 
			if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
				add_bias=False
			 
						
			# Linear operations
			self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
			 
			# weight initialization
			self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
			self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
			 
			current_input=self.fc_lay[i]
			 
			 
	def forward(self, x):

			
		# Applying Layer/Batch Norm
		if bool(self.fc_use_laynorm_inp):
			x=self.ln0((x))
			
		if bool(self.fc_use_batchnorm_inp):
			x=self.bn0((x))
			
		for i in range(self.N_fc_lay):

			if self.fc_act[i]!='linear':
					
				if self.fc_use_laynorm[i]:
					x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
				
				if self.fc_use_batchnorm[i]:
					x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
				
				if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
					x = self.drop[i](self.act[i](self.wx[i](x)))
				 
			else:
				if self.fc_use_laynorm[i]:
					x = self.drop[i](self.ln[i](self.wx[i](x)))
				
				if self.fc_use_batchnorm[i]:
					x = self.drop[i](self.bn[i](self.wx[i](x)))
				
				if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
					x = self.drop[i](self.wx[i](x)) 
				
		return x


class MLP_for_me(nn.Module):
	def __init__(self, options):
		super(MLP_for_me, self).__init__()
		
		self.input_dim=int(options['input_dim'])
		self.fc_lay=options['fc_lay']
		self.fc_drop=options['fc_drop']
		self.fc_use_batchnorm=options['fc_use_batchnorm']
		self.fc_use_laynorm=options['fc_use_laynorm']
		self.fc_use_laynorm_inp=options['fc_use_laynorm_inp']
		self.fc_use_batchnorm_inp=options['fc_use_batchnorm_inp']
		self.fc_act=options['fc_act']
		
	 
		self.wx  = nn.ModuleList([])
		self.bn  = nn.ModuleList([])
		self.ln  = nn.ModuleList([])
		self.act = nn.ModuleList([])
		self.drop = nn.ModuleList([])
	 

	 
		# input layer normalization
		if self.fc_use_laynorm_inp:
			self.ln0=LayerNorm(self.input_dim)
			
		# input batch normalization    
		if self.fc_use_batchnorm_inp:
			self.bn0=nn.BatchNorm1d(self.input_dim,momentum=0.05)
			 
			 
		self.N_fc_lay=len(self.fc_lay)
				 
		current_input=self.input_dim
		
		# Initialization of hidden layers
		
		for i in range(self.N_fc_lay):
			# dropout
			self.drop.append(nn.Dropout(p=self.fc_drop[i]))
		
			# activation
			self.act.append(act_fun(self.fc_act[i]))

			add_bias=True
			 
			# layer norm initialization
			self.ln.append(LayerNorm(self.fc_lay[i]))
			self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
			 
			if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
				add_bias=False
			 
						
			# Linear operations
			self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
			 
			# weight initialization
			self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
			self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
			 
			current_input=self.fc_lay[i]
			 
			 
	def forward(self, x):	
		# Applying Layer/Batch Norm
		if bool(self.fc_use_laynorm_inp):
			x=self.ln0((x))
			
		if bool(self.fc_use_batchnorm_inp):
			x=self.bn0(x.transpose(-1, -2)).transpose(-1, -2)
		
		for i in range(self.N_fc_lay):

			if self.fc_act[i]!='linear':
				if self.fc_use_laynorm[i]:
					x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
				
				elif self.fc_use_batchnorm[i]:
					x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x).transpose(-1, -2)).transpose(-1, -2)))

				elif self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
					x = self.drop[i](self.act[i](self.wx[i](x)))
				 
			else:
				if self.fc_use_laynorm[i]:
					x = self.drop[i](self.ln[i](self.wx[i](x)))
				
				if self.fc_use_batchnorm[i]:
					x = self.drop[i](self.bn[i](self.wx[i](x).transpose(-1, -2)).transpose(-1, -2))
				
				if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
					x = self.drop[i](self.wx[i](x)) 
		
		return x


class SincNet(nn.Module):
		
	def __init__(self,options):
		super(SincNet,self).__init__()
	
		self.cnn_N_filt=options['cnn_N_filt']
		self.cnn_len_filt=options['cnn_len_filt']
		self.cnn_max_pool_len=options['cnn_max_pool_len']
		 
		 
		self.cnn_act=options['cnn_act']
		self.cnn_drop=options['cnn_drop']
		 
		self.cnn_use_laynorm=options['cnn_use_laynorm']
		self.cnn_use_batchnorm=options['cnn_use_batchnorm']
		self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
		self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
		 
		self.input_dim=int(options['input_dim'])
		 
		self.fs=options['fs']
		 
		self.N_cnn_lay=len(options['cnn_N_filt'])
		self.conv  = nn.ModuleList([])
		self.bn  = nn.ModuleList([])
		self.ln  = nn.ModuleList([])
		self.act = nn.ModuleList([])
		self.drop = nn.ModuleList([])
		 
					 
		if self.cnn_use_laynorm_inp:
			self.ln0=LayerNorm(self.input_dim)
				 
		if self.cnn_use_batchnorm_inp:
			self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
				 
		current_input=self.input_dim 
		 
		for i in range(self.N_cnn_lay):
			 
			N_filt=int(self.cnn_N_filt[i])
			len_filt=int(self.cnn_len_filt[i])
			 
			# dropout
			self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
			 
			# activation
			self.act.append(act_fun(self.cnn_act[i]))
									
			# layer norm initialization         
			self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

			self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
					

			if i==0:
				self.conv.append(SincConv_fast(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs))
						
			else:
				self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
				
			current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

			 
		self.out_dim=current_input*N_filt



	def forward(self, x):
		batch=x.shape[0]
		seq_len=x.shape[1]
		 
		if bool(self.cnn_use_laynorm_inp):
			x=self.ln0((x))
			
		if bool(self.cnn_use_batchnorm_inp):
			x=self.bn0((x))
			
		x=x.view(batch,1,seq_len)

		 
		for i in range(self.N_cnn_lay):
				 
			if self.cnn_use_laynorm[i]:
				if i==0:
					x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
				else:
					x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
				
			if self.cnn_use_batchnorm[i]:
				x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

			if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
				x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

		 
		x = x.view(batch,-1)

		return x

class ConvNet(nn.Module):
		
	def __init__(self,options):
		super(ConvNet,self).__init__()
	
		self.cnn_N_filt=options['cnn_N_filt']
		self.cnn_len_filt=options['cnn_len_filt']
		self.cnn_max_pool_len=options['cnn_max_pool_len']
		 
		 
		self.cnn_act=options['cnn_act']
		self.cnn_drop=options['cnn_drop']
		 
		self.cnn_use_laynorm=options['cnn_use_laynorm']
		self.cnn_use_batchnorm=options['cnn_use_batchnorm']
		self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
		self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
		 
		self.input_dim=int(options['input_dim'])
		 
		self.fs=options['fs']
		 
		self.N_cnn_lay=len(options['cnn_N_filt'])
		self.conv  = nn.ModuleList([])
		self.bn  = nn.ModuleList([])
		self.ln  = nn.ModuleList([])
		self.act = nn.ModuleList([])
		self.drop = nn.ModuleList([])
		 
					 
		if self.cnn_use_laynorm_inp:
			self.ln0=LayerNorm(self.input_dim)
				 
		if self.cnn_use_batchnorm_inp:
			self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
				 
		current_input=self.input_dim 
		 
		for i in range(self.N_cnn_lay):
			 
			N_filt=int(self.cnn_N_filt[i])
			len_filt=int(self.cnn_len_filt[i])
			 
			# dropout
			self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
			 
			# activation
			self.act.append(act_fun(self.cnn_act[i]))
									
			# layer norm initialization  
			self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

			self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
					

			if i==0:
				self.conv.append(nn.Conv1d(8, self.cnn_N_filt[i], self.cnn_len_filt[i]))
						
			else:
				self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
				
			current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

			 
		self.out_dim=current_input*N_filt



	def forward(self, x):
		x = x.transpose(1, 2)
		batch=x.shape[0]
		seq_len=x.shape[1]
		 
		if bool(self.cnn_use_laynorm_inp):
			x=self.ln0((x))
			
		if bool(self.cnn_use_batchnorm_inp):
			x=self.bn0((x))
		
		x=x.view(batch,1,seq_len)
		 
		for i in range(self.N_cnn_lay):
				 
			if self.cnn_use_laynorm[i]:
				x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
				
			if self.cnn_use_batchnorm[i]:
				x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

			if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
				x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

		 
		x = x.view(batch,-1)

		return x



class LSTM(nn.Module):
	def __init__(self,
				 embed_dim=8,
				 hidden_size=128,
				 num_layers=4,
				 bidirectional=True,
				 dropout_in=0.25,
				 dropout_out=0.25):

		super(LSTM, self).__init__()

		self.dropout_in = dropout_in
		self.dropout_out = dropout_out
		self.bidirectional = bidirectional
		self.hidden_size = hidden_size
		self.out_dim = 2 * hidden_size if bidirectional else hidden_size

		
		dropout_lstm = dropout_out if num_layers > 1 else 0.
		self.lstm = nn.LSTM(input_size=embed_dim,
												hidden_size=hidden_size,
												num_layers=num_layers,
												dropout=dropout_lstm,
												bidirectional=bidirectional)

	def forward(self, src_embeddings):
		""" Performs a single forward pass through the instantiated encoder sub-network. """
		# Embed tokens and apply dropout
		batch_size, src_time_steps, embed_dim = src_embeddings.size()
		src_lengths = [src_time_steps] * batch_size
		_src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

		# Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
		src_embeddings = _src_embeddings.transpose(0, 1)

		# Pack embedded tokens into a PackedSequence
		packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths)

		# Pass source input through the recurrent layer(s)
		packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings)

		# Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
		lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
		lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
		assert list(lstm_output.size()) == [src_time_steps, batch_size, self.out_dim]  # sanity check
		
		lstm_output = lstm_output.transpose(0, 1)

		return lstm_output



def make_positions(tensor, padding_idx, left_pad):
	"""Replace non-padding symbols with their position numbers.
	Position numbers begin at padding_idx+1.
	Padding symbols are ignored, but it is necessary to specify whether padding
	is added on the left side (left_pad=True) or right side (left_pad=False).
	"""
	max_pos = padding_idx + 1 + tensor.size(1)
	if not hasattr(make_positions, 'range_buf'):
		make_positions.range_buf = tensor.new()
	make_positions.range_buf = make_positions.range_buf.type_as(tensor)
	if make_positions.range_buf.numel() < max_pos:
		torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
	mask = tensor.ne(padding_idx)
	positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
	if left_pad:
		positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
	return tensor.clone().masked_scatter_(mask, positions[mask])

class LearnedPositionalEmbedding(nn.Embedding):
	"""This module learns positional embeddings up to a fixed maximum size.
	Padding symbols are ignored, but it is necessary to specify whether padding
	is added on the left side (left_pad=True) or right side (left_pad=False).
	"""

	def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
		super().__init__(num_embeddings, embedding_dim, padding_idx)
		self.left_pad = left_pad

	def forward(self, input, incremental_state=None):
		"""Input is expected to be of size [bsz x seqlen]."""
		if incremental_state is not None:
			# positions is the same for every token when decoding a single step
			positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
		else:
			positions = make_positions(input.data, self.padding_idx, self.left_pad)
		return super().forward(Variable(positions))

	def max_positions(self):
		"""Maximum number of supported positions."""
		return self.num_embeddings - self.padding_idx - 1


class SinusoidalPositionalEmbedding(nn.Module):
	"""This module produces sinusoidal positional embeddings of any length.
	Padding symbols are ignored, but it is necessary to specify whether padding
	is added on the left side (left_pad=True) or right side (left_pad=False).
	"""

	def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.padding_idx = padding_idx
		self.left_pad = left_pad
		self.register_buffer(
			'weights',
			SinusoidalPositionalEmbedding.get_embedding(
				init_size,
				embedding_dim,
				padding_idx,
			),
		)

	@staticmethod
	def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
		"""Build sinusoidal embeddings.
		This matches the implementation in tensor2tensor, but differs slightly
		from the description in Section 3.5 of "Attention Is All You Need".
		"""
		half_dim = embedding_dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim) * -emb)
		emb = torch.arange(num_embeddings).unsqueeze(1) * emb.unsqueeze(0)
		emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
		if embedding_dim % 2 == 1:
			# zero pad
			emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
		if padding_idx is not None:
			emb[padding_idx, :] = 0
		return emb

	def forward(self, input, incremental_state=None):
		"""Input is expected to be of size [bsz x seqlen]."""
		# recompute/expand embeddings if needed
		bsz, seq_len = input.size()
		max_pos = self.padding_idx + 1 + seq_len
		if max_pos > self.weights.size(0):
			self.weights = SinusoidalPositionalEmbedding.get_embedding(
				max_pos,
				self.embedding_dim,
				self.padding_idx,
			).type_as(self.weights)
		weights = Variable(self.weights)

		if incremental_state is not None:
			# positions is the same for every token when decoding a single step
			return weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

		positions = Variable(make_positions(input.data, self.padding_idx, self.left_pad))
		return weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)

	def max_positions(self):
		"""Maximum number of supported positions."""
		return int(1e5)  # an arbitrary large number


class LayerNormalization(nn.Module):
	"""Layer normalization for module"""

	def __init__(self, hidden_size, eps=1e-6, affine=True):
		super(LayerNormalization, self).__init__()

		self.affine = affine
		self.eps = eps
		if self.affine:
			self.gamma = nn.Parameter(torch.ones(hidden_size))
			self.beta = nn.Parameter(torch.zeros(hidden_size))

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
	m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
	m.weight.data.normal_(0, 0.1)
	return m

def residual(x, y, dropout, training):
	"""Residual connection"""
	y = F.dropout(y, p=dropout, training=training)
	return x + y

def Linear(in_features, out_features, bias=True, dropout=0):
	"""Weight-normalized Linear layer (input: N x T x C)"""
	m = nn.Linear(in_features, out_features, bias=bias)
	m.weight.data.uniform_(-0.1, 0.1)
	if bias:
		m.bias.data.uniform_(-0.1, 0.1)
	return m

def split_heads(x, num_heads):
	"""split x into multi heads
	Args:
		x: [batch_size, length, depth]
	Returns:
		y: [[batch_size, length, depth / num_heads] x heads]
	"""
	sz = x.size()
	# x -> [batch_size, length, heads, depth / num_heads]
	x = x.view(sz[0], sz[1], num_heads, sz[2] // num_heads)
	# [batch_size, length, 1, depth // num_heads] * 
	heads = torch.chunk(x, num_heads, 2)
	x = []
	for i in range(num_heads):
		x.append(torch.squeeze(heads[i], 2))
	return x

def combine_heads(x):
	"""combine multi heads
	Args:
		x: [batch_size, length, depth / num_heads] x heads
	Returns:
		x: [batch_size, length, depth]
	"""
	return torch.cat(x, 2)


def dot_product_attention(q, k, v, bias, dropout, to_weights=False):
	"""dot product for query-key-value
	Args:
		q: query antecedent, [batch, length, depth]
		k: key antecedent,   [batch, length, depth]
		v: value antecedent, [batch, length, depth]
		bias: masked matrix
		dropout: dropout rate
		to_weights: whether to print weights
	"""
	# [batch, length, depth] x [batch, depth, length] -> [batch, length, length]
	logits = torch.bmm(q, k.transpose(1, 2).contiguous())
	if bias is not None:
		logits += bias
	size = logits.size()
	weights = F.softmax(logits.view(size[0] * size[1], size[2]), dim=1)
	weights = weights.view(size)
	if to_weights:
		return torch.bmm(weights, v), weights
	else:
		return torch.bmm(weights, v)


class FeedForwardNetwork(nn.Module):
	def __init__(self, hidden_size, filter_size, dropout):
		super(FeedForwardNetwork, self).__init__()
		self.fc1 = Linear(hidden_size, filter_size, bias=False)
		self.fc2 = Linear(filter_size, hidden_size, bias=False)
		self.dropout = dropout

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.fc2(x)
		return x


class MultiheadAttention(nn.Module):
	"""Multi-head attention mechanism"""
	def __init__(self, 
				 key_depth, value_depth, output_depth,
				 num_heads, dropout=0.1):
		super(MultiheadAttention, self).__init__()

		self._query = Linear(key_depth, key_depth, bias=False)
		self._key = Linear(key_depth, key_depth, bias=False)
		self._value = Linear(value_depth, value_depth, bias=False)
		self.output_perform = Linear(value_depth, output_depth, bias=False)

		self.num_heads = num_heads
		self.key_depth_per_head = key_depth // num_heads
		self.dropout = dropout
		
	def forward(self, query_antecedent, memory_antecedent, bias, to_weights=False):
		if memory_antecedent is None:
			memory_antecedent = query_antecedent
		q = self._query(query_antecedent)
		k = self._key(memory_antecedent)
		v = self._value(memory_antecedent)
		q *= self.key_depth_per_head ** -0.5
		
		# split heads
		q = split_heads(q, self.num_heads)
		k = split_heads(k, self.num_heads)
		v = split_heads(v, self.num_heads)

		x = []
		avg_attn_scores = None
		for i in range(self.num_heads):
			results = dot_product_attention(q[i], k[i], v[i],
											bias,
											self.dropout,
											to_weights)
			if to_weights:
				y, attn_scores = results
				if avg_attn_scores is None:
					avg_attn_scores = attn_scores
				else:
					avg_attn_scores.add_(attn_scores)
			else:
				y = results
			x.append(y)
		x = combine_heads(x)
		x = self.output_perform(x)
		if to_weights:
			return x, avg_attn_scores / self.num_heads
		else:
			return x

def attention_bias_ignore_padding(src_tokens, padding_idx):
	"""Calculate the padding mask based on which embedding are zero
	Args:
		src_tokens: [batch_size, length]
	Returns:
		bias: [batch_size, length]
	"""
	return src_tokens.eq(padding_idx).unsqueeze(1)

def encoder_attention_bias(bias):
	batch_size, _, length = bias.size()
	return bias.expand(batch_size, length, length).float() * -1e9


class TransformerEncoder(nn.Module):
	"""Transformer encoder."""
	def __init__(self, embed_dim=256, max_positions=1024, pos="learned",
				 num_layers=4, num_heads=8,
				 filter_size=256, hidden_size=256,
				 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, cuda=True):
		super(TransformerEncoder, self).__init__()
		assert pos == "learned" or pos == "timing" or pos == "nopos"

		self.cuda = cuda

		self.dropout = dropout
		self.attention_dropout = attention_dropout
		self.relu_dropout = relu_dropout
		self.pos = pos

		padding_idx = 0
		if self.pos == "learned":
			self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx,
													   left_pad=False)
		if self.pos == "timing":
			self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx,
																 left_pad=False)

		self.layers = num_layers

		self.self_attention_blocks = nn.ModuleList()
		self.ffn_blocks = nn.ModuleList()
		self.norm1_blocks = nn.ModuleList()
		self.norm2_blocks = nn.ModuleList()
		for i in range(num_layers):
			self.self_attention_blocks.append(MultiheadAttention(hidden_size,
																 hidden_size,
																 hidden_size,
																 num_heads))
			self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
			self.norm1_blocks.append(LayerNormalization(hidden_size))
			self.norm2_blocks.append(LayerNormalization(hidden_size))
		self.out_norm = LayerNormalization(hidden_size)

	def forward(self, encoder_input):
		# embed tokens plus positions
		batch_size, src_time_steps, embed_dim = encoder_input.size()
		src_lengths = [src_time_steps] * batch_size
		src_tokens = encoder_input[:, :, 0]
		padding_idx = 0
		input_to_padding = attention_bias_ignore_padding(src_tokens, padding_idx)
		encoder_self_attention_bias = encoder_attention_bias(input_to_padding)
		if self.pos != "nopos":
			if self.cuda:
				encoder_input += self.embed_positions(src_tokens.type(torch.cuda.LongTensor))
			else:
				encoder_input += self.embed_positions(src_tokens.type(torch.LongTensor))

		x = F.dropout(encoder_input, p=self.dropout, training=self.training)
		for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
													 self.ffn_blocks,
													 self.norm1_blocks,
													 self.norm2_blocks):
			y = self_attention(norm1(x), None, encoder_self_attention_bias)
			x = residual(x, y, self.dropout, self.training)
			y = ffn(norm2(x))
			x = residual(x, y, self.dropout, self.training)
		x = self.out_norm(x)
		return x

	def max_positions(self):
		"""Maximum input length supported by the encoder."""
		if self.pos == "learned":
			return self.embed_positions.max_positions()
		else:
			return 1024
		 

class FunTimesCNN(nn.Module):

	def __init__(self, MLP_before_arch, MLP_after_arch, CNN_arch, use_sinc_net):
		super(FunTimes, self).__init__()
		if MLP_before_arch != None:
			self.embed_dim_projection = MLP_for_me(MLP_before_arch)
		else:
			self.embed_dim_projection = None

		if use_sinc_net:
			self.CNN_net = SincNet(CNN_arch)
		else:
			self.CNN_net = ConvNet(CNN_arch)
		self.result_projection = MLP_for_me(MLP_after_arch)
		
	def forward(self, x):
		if self.embed_dim_projection:
			x = self.embed_dim_projection(x)
		return self.result_projection(self.CNN_net(x))

class FunTimesLSTM(nn.Module):

	def __init__(self, MLP_before_arch, MLP_after_arch, lstm_embed_dim, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, lstm_dropout_in, lstm_dropout_out, raw=False):
		super(FunTimesLSTM, self).__init__()

		if MLP_before_arch != None:
			self.embed_dim_projection = MLP_for_me(MLP_before_arch)
		else:
			self.embed_dim_projection = None
		self.LSTM = LSTM(lstm_embed_dim, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, lstm_dropout_in, lstm_dropout_out)
		self.result_projection = MLP_for_me(MLP_after_arch)
		self.raw = raw
		
	def forward(self, x):
		if self.raw:
			x = x.unsqueeze(-1)
		if self.embed_dim_projection:
			x = self.embed_dim_projection(x)
		x = self.LSTM(x)
		x = self.result_projection(x)
		return x.squeeze(-1)


class FunTimesTransformer(nn.Module):

	def __init__(self, MLP_before_arch, MLP_after_arch, tr_embed_dim, tr_max_positions, tr_pos, tr_num_layers,
		tr_num_heads, tr_filter_size, tr_hidden_size, tr_dropout, 
		tr_attention_dropout, tr_relu_dropout, cuda):

		super(FunTimesTransformer, self).__init__()

		if MLP_before_arch != None:
			self.embed_dim_projection = MLP_for_me(MLP_before_arch)
		else:
			self.embed_dim_projection = None

		self.transformer = TransformerEncoder(
			tr_embed_dim, tr_max_positions, tr_pos, tr_num_layers,
			tr_num_heads, tr_filter_size, tr_hidden_size, tr_dropout, 
			tr_attention_dropout, tr_relu_dropout, cuda)

		self.result_projection = MLP_for_me(MLP_after_arch)
		
	def forward(self, x):
		if self.embed_dim_projection:
			x = self.embed_dim_projection(x)
		x = self.transformer(x)
		x = self.result_projection(x)
		return x.squeeze(-1)


class YeetZ_MLP(nn.Module):
	def __init__(self):
		super(YeetZ_MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(60, 10),
			nn.ReLU(),
			nn.Linear(10, 1),
			nn.Softplus()
		)
		
	def forward(self, x):
		x = self.layers(x)
		return x.squeeze(-1)

class EZConv(nn.Module):
	def __init__(self):
		super(EZConv, self).__init__()
		self.conv = nn.Conv1d(8, 80, 25)
		self.conv2 = nn.Conv1d(80, 60, 5)
		self.conv3 = nn.Conv1d(60, 60, 5)
		self.act = nn.LeakyReLU(0.2)
		self.mlp = YeetZ_MLP()

	def forward(self, x):
		x = x.transpose(1, 2)
		x = self.conv(x)
		x = F.max_pool1d(x, 3)
		x = self.act(x)

		x = self.conv2(x)
		x = F.max_pool1d(x, 3)
		x = self.act(x)

		x = self.conv3(x)
		x = F.max_pool1d(x, 3)
		x = self.act(x)

		x = x.transpose(1, 2)
		x = self.mlp(x)
		return x