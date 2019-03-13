import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
#import scipy.io.wavfile
import torch

def ReadList(list_file):
 f=open(list_file,"r")
 lines=f.readlines()
 list_sig=[]
 for x in lines:
    list_sig.append(x.rstrip())
 f.close()
 return list_sig


def read_conf():
 
 parser=OptionParser()
 parser.add_option("--cfg") # Mandatory
 parser.add_option("--model")
 (options,args)=parser.parse_args()
 cfg_file=options.cfg
 Config = ConfigParser.ConfigParser()
 Config.read(cfg_file)

 #[data]
 options.train_src_dir=Config.get('data', 'train_src_dir')
 options.train_tgt_dir=Config.get('data', 'train_tgt_dir')
 options.dev_src_dir=Config.get('data', 'dev_src_dir')
 options.dev_tgt_dir=Config.get('data', 'dev_tgt_dir')
 options.test_src_dir=Config.get('data', 'test_src_dir')
 options.test_tgt_dir=Config.get('data', 'test_tgt_dir')
 options.output_folder=Config.get('data', 'output_folder')
 options.save_dir=Config.get('data', 'save_dir')
 options.restore_file=Config.get('data', 'restore_file')

 #[windowing]
 options.fs=Config.get('windowing', 'fs')

 #[cnn]
 options.wlen=Config.get('cnn', 'wlen')
 options.cnn_N_filt=Config.get('cnn', 'cnn_N_filt')
 options.cnn_len_filt=Config.get('cnn', 'cnn_len_filt')
 options.cnn_max_pool_len=Config.get('cnn', 'cnn_max_pool_len')
 options.cnn_use_laynorm_inp=Config.get('cnn', 'cnn_use_laynorm_inp')
 options.cnn_use_batchnorm_inp=Config.get('cnn', 'cnn_use_batchnorm_inp')
 options.cnn_use_laynorm=Config.get('cnn', 'cnn_use_laynorm')
 options.cnn_use_batchnorm=Config.get('cnn', 'cnn_use_batchnorm')
 options.cnn_act=Config.get('cnn', 'cnn_act')
 options.cnn_drop=Config.get('cnn', 'cnn_drop')

 #[transformer]
 options.tr_embed_dim=Config.get('transformer', 'tr_embed_dim')
 options.tr_max_positions=Config.get('transformer', 'tr_max_positions')
 options.tr_pos=Config.get('transformer', 'tr_pos')
 options.tr_num_layers=Config.get('transformer', 'tr_num_layers')
 options.tr_num_heads=Config.get('transformer', 'tr_num_heads')
 options.tr_filter_size=Config.get('transformer', 'tr_filter_size')
 options.tr_hidden_size=Config.get('transformer', 'tr_hidden_size')
 options.tr_dropout=Config.get('transformer', 'tr_dropout')
 options.tr_attention_dropout=Config.get('transformer', 'tr_attention_dropout')
 options.tr_relu_dropout=Config.get('transformer', 'tr_relu_dropout')

 #[lstm]
 options.lstm_embed_dim=Config.get('lstm', 'lstm_embed_dim')
 options.lstm_hidden_size=Config.get('lstm', 'lstm_hidden_size')
 options.lstm_num_layers=Config.get('lstm', 'lstm_num_layers')
 options.lstm_bidirectional=Config.get('lstm', 'lstm_bidirectional')
 options.lstm_dropout_in=Config.get('lstm', 'lstm_dropout_in')
 options.lstm_dropout_out=Config.get('lstm', 'lstm_dropout_out')
 
 #[dnn_before]
 options.fc1_lay_use=Config.get('dnn_before', 'fc1_lay_use')
 options.fc1_lay=Config.get('dnn_before', 'fc1_lay')
 options.fc1_drop=Config.get('dnn_before', 'fc1_drop')
 options.fc1_use_laynorm_inp=Config.get('dnn_before', 'fc1_use_laynorm_inp')
 options.fc1_use_batchnorm_inp=Config.get('dnn_before', 'fc1_use_batchnorm_inp')
 options.fc1_use_batchnorm=Config.get('dnn_before', 'fc1_use_batchnorm')
 options.fc1_use_laynorm=Config.get('dnn_before', 'fc1_use_laynorm')
 options.fc1_act=Config.get('dnn_before', 'fc1_act')

 #[dnn_after]
 options.fc2_lay=Config.get('dnn_after', 'fc2_lay')
 options.fc2_drop=Config.get('dnn_after', 'fc2_drop')
 options.fc2_use_laynorm_inp=Config.get('dnn_after', 'fc2_use_laynorm_inp')
 options.fc2_use_batchnorm_inp=Config.get('dnn_after', 'fc2_use_batchnorm_inp')
 options.fc2_use_batchnorm=Config.get('dnn_after', 'fc2_use_batchnorm')
 options.fc2_use_laynorm=Config.get('dnn_after', 'fc2_use_laynorm')
 options.fc2_act=Config.get('dnn_after', 'fc2_act')


 #[optimization]
 options.optimizer=Config.get('optimization', 'optimizer')
 options.weight_decay=Config.get('optimization', 'weight_decay')
 options.lr=Config.get('optimization', 'lr')
 options.batch_size=Config.get('optimization', 'batch_size')
 options.N_epochs=Config.get('optimization', 'N_epochs')
 options.seed=Config.get('optimization', 'seed')
 options.cuda=Config.get('optimization', 'cuda')
 options.patience=Config.get('optimization', 'patience')
 
 return options


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 
         
         
def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
 # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
 sig_batch=np.zeros([batch_size,wlen])
 lab_batch=np.zeros(batch_size)
  
 snt_id_arr=np.random.randint(N_snt, size=batch_size)
 
 rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

 for i in range(batch_size):
     
  # select a random sentence from the list  (joint distribution)
  [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
  signal=signal.astype(float)/32768

  # accesing to a random chunk
  snt_len=signal.shape[0]
  snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
  snt_end=snt_beg+wlen
  
  sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
  lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
  
 inp=torch.from_numpy(sig_batch).float().cuda().contiguous()  # Current Frame
 lab=torch.from_numpy(lab_batch).float().cuda().contiguous()
  
 return inp,lab  