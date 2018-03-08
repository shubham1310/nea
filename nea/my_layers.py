import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import pdb
from .utils import tensordot

class Attention(torch.nn.Module):
	def __init__(self, input_size, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
		super(Attention, self).__init__(**kwargs)
		assert op in {'attsum', 'attmean'}
		assert activation in {None, 'tanh'}
		self.op = op
		self.activation = activation
		self.init_stdev = init_stdev
		self.input_size = input_size
		self.att_V = nn.Parameter(torch.randn(input_size, 1) * init_stdev)
		self.att_W = nn.Parameter(torch.randn(input_size, input_size) * init_stdev)

	def forward(self,x, mask=None):
		# pdb.set_trace()
		y = tensordot(x, self.att_W)
		if not self.activation:
			weights = tensordot(y,self.att_V)
		elif self.activation == 'tanh':
			weights = tensordot(F.tanh(y), self.att_V)
		w = F.softmax(weights, dim=2)
		w = w.expand(*w.size()[:-1], self.input_size)
		s = (x * w).sum(1) 
		if self.op == 'attsum':
			return s
		elif self.op == 'attmean':
			return torch.div(s, Variable(mask.squeeze(1).sum(1).unsqueeze(1).expand(*s.size()).float()))

class MeanOverTime(torch.nn.Module):
	def __init__(self , **kwargs):
		super(MeanOverTime, self).__init__(**kwargs)

	def forward(self,x, mask=None):
		if not(mask is None):
			mask = mask.type(torch.DoubleTensor)
			# pdb.set_trace()
			s = x.sum(1) 
			return torch.div(s, Variable(mask.squeeze(1).sum(1).unsqueeze(1).expand(*s.size()).float()))
		else:
			return x.mean(1)

class Conv1DWithMasking(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super(Conv1DWithMasking, self).__init__()
		self.conv =  torch.nn.Conv1d(*args)

	def forward(self, x, mask=None):
		# pdb.set_trace()
		x = x.permute([0, 2, 1])
		x= self.conv(x)
		x = x.permute([0, 2, 1])
		if not(mask is None):
			x = torch.mul(x, Variable(mask.squeeze(1).unsqueeze(2).expand(*x.size()).float()))
		return x
