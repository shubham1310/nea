import torch
import numpy as np
import torch.nn.functional as F
import sys

class Attention(torch.nn.Module):
	def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
		super(Attention, self).__init__(**kwargs)
		assert op in {'attsum', 'attmean'}
		assert activation in {None, 'tanh'}
		self.op = op
		self.activation = activation
		self.init_stdev = init_stdev
		self.att_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype('float32')
		self.att_W =(np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype('float32')
		self.trainable_weights = [self.att_v, self.att_W]

	def forward(self,x, mask=None):
		y = torch.mm(x, self.att_W)
		if not self.activation:
			weights = torch.mm(self.att_v, y, axes=[0, 2])
		elif self.activation == 'tanh':
			weights = torch.mm(self.att_v, F.tanh(y), axes=[0, 2])
		weights = F.softmax(weights)
		out = x * weights.unsqueeze(1).repeat(1, 1, x.shape[2])
		if self.op == 'attsum':
			out = out.sum(axis=1)
		elif self.op == 'attmean':
			out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
		return out.type(torch.DoubleTensor)

class MeanOverTime(torch.nn.Module):
	def __init__(self, mask_zero=True, **kwargs):
		self.mask_zero = mask_zero
		super(MeanOverTime, self).__init__(**kwargs)

	def forward(self,x, mask=None):
		if self.mask_zero:
			mask = mask.type(torch.DoubleTensor)
			return (x.sum(1) / mask.sum(1).unsqueeze(1)).type(torch.DoubleTensor)
		else:
			return x.mean(1)

class Conv1DWithMasking(torch.nn.Module):
	def __init__(self, **kwargs):
		self.conv =  torch.nn.Conv1d(kwargs)
		super(Conv1DWithMasking, self).__init__(**kwargs)

	def forward(self, x, mask=None):
		if not(mask ==None):
			x = torch.mul(x, mask)
		return self.conv(x)
