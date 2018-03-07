import numpy as np
import logging
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Embedding, Dropout, Linear
from nea.my_layers import Attention, MeanOverTime, Conv1DWithMasking
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
logger = logging.getLogger(__name__)

class REGRESSION(nn.Module):
	def __init__(self, args, emb_index, bidirec, initial_mean_value, overal_maxlen=0):
		super(REGRESSION, self).__init__()
		self.dropout_W = 0.5		# default=0.5
		self.dropout_U = 0.1		# default=0.1
		self.args = args
		cnn_border_mode='same'
		if initial_mean_value.ndim == 0:
			initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
		num_outputs = len(initial_mean_value)
		if args.recurrent_unit == 'lstm':
			from torch.nn import LSTM as RNN
		elif args.recurrent_unit == 'gru':
			from torch.nn import GRU as RNN
		elif args.recurrent_unit == 'simple':
			from torch.nn import RNN as RNN

		self.embed = Embedding(args.vocab_size, args.emb_dim)
		outputdim = args.emb_dim
		if args.cnn_dim > 0:
			self.conv = Conv1DWithMasking(outputdim,args.cnn_dim, args.cnn_window_size, padding=(args.cnn_window_size - 1)//2)
			outputdim = args.cnn_dim
		if args.rnn_dim > 0:
			self.rnn = RNN(outputdim,args.rnn_dim,num_layers=1, bias=True, dropout=self.dropout_W, batch_first=True, bidirectional=bidirec)
			outputdim = args.rnn_dim
			if bidirec==1:
				outputdim =  args.rnn_dim * 2
		if args.dropout_prob > 0:
			self.dropout = Dropout(args.dropout_prob)
		if args.aggregation == 'mot':
			self.mot = MeanOverTime()
		elif args.aggregation.startswith('att'):
			self.att = Attention(op=args.aggregation, activation='tanh', init_stdev=0.01)

		self.linear  = Linear(outputdim,num_outputs)
		# if not args.skip_init_bias:
		# 	self.linear.bias.data = (torch.log(initial_mean_value) - torch.log(1 - initial_mean_value)).float()
		self.emb_index = emb_index
		if args.emb_path:
			from .w2vEmbReader import W2VEmbReader as EmbReader
			logger.info('Initializing lookup table')
			emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
			self.embed[emb_index].weight.data = emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].get_weights()) 
		logger.info('  Done')

	def forward(self,x,lens,mask=None):
		# for i in x:
		# 	print(x.shape)
		lens, perm_idx = lens.sort(0, descending=True)
		x = x[perm_idx]
		x=self.embed(x)
		if self.args.cnn_dim > 0:
			x = self.conv(x, mask=mask)
		if self.args.rnn_dim > 0:
			print(lens.numpy())
			x = pack_padded_sequence(x, lens.numpy(), batch_first=True)
			# h0 = Variable(torch.zeros(batch_size, self.args.rnn_dim))
			# c0 = Variable(torch.zeros(batch_size, self.args.rnn_dim))
			x,_ = self.rnn(x)  # (h0, c0)
			print(x)
			# current = temp[0]
			x, _ = pad_packed_sequence(x, batch_first=True)
		if self.args.dropout_prob > 0:
			x = self.dropout(x)
		if self.args.aggregation == 'mot':
			x= self.mot (x, mask=mask)
		elif self.args.aggregation.startswith('att'):
			x= self.att(x, mask=mask)
		x = self.linear(x)
		x = F.sigmoid(x)
		return x

def create_model(args, overal_maxlen, vocab, initial_mean_value):
	
	import keras.backend as K
	
	if args.model_type == 'cls':
		raise NotImplementedError
	elif args.model_type == 'reg':
		logger.info('Building a REGRESSION model')
		return REGRESSION(args, emb_index =0, bidirec=0, overal_maxlen=overal_maxlen, initial_mean_value=initial_mean_value)
	elif args.model_type == 'regp':
		logger.info('Building a REGRESSION model with POOLING')
		return REGRESSION(args, emb_index =0, bidirec=0, overal_maxlen=overal_maxlen, initial_mean_value=initial_mean_value)
	elif args.model_type == 'breg':
		logger.info('Building a BIDIRECTIONAL REGRESSION model')
		return REGRESSION(args, emb_index =1, bidirec=1, overal_maxlen=overal_maxlen, initial_mean_value=initial_mean_value)
	elif args.model_type == 'bregp':
		logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
		return REGRESSION(args, emb_index =1, bidirec=1, overal_maxlen=overal_maxlen, initial_mean_value=initial_mean_value)
	
