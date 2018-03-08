#!/usr/bin/env python
import os
import argparse
import logging
import numpy as np
import scipy
from time import time
import sys
import pdb
import nea.utils as U
import pickle as pk

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae) (default=mse)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
args = parser.parse_args()
print (args)

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir)
U.print_args(args)

assert args.model_type in {'reg', 'regp', 'breg', 'bregp'}
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.loss in {'mse', 'mae'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.aggregation in {'mot', 'attsum', 'attmean'}

if args.seed > 0:
	np.random.seed(args.seed)

from nea.asap_evaluator import Evaluator
from nea.asap_reader import *
# import nea.asap_reader as dataset

# data_x is a list of lists
vocab, vocab_size, overal_maxlen,_ = get_stats((args.train_path, args.dev_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path)

traindataset= dataloader((args.train_path, args.train_path), args.prompt_id, args.vocab_size, args.maxlen, vocab_path=args.vocab_path)
devdataset= dataloader((args.train_path, args.dev_path), args.prompt_id, args.vocab_size, args.maxlen, vocab_path=args.vocab_path)
testdataset = dataloader((args.train_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, vocab_path=args.vocab_path)

traindata = torch.utils.data.DataLoader(traindataset,batch_size=args.batch_size,shuffle=True, num_workers=1)
devdata = torch.utils.data.DataLoader(devdataset,batch_size=args.batch_size,shuffle=True, num_workers=1)
testdata = torch.utils.data.DataLoader(testdataset,batch_size=args.batch_size,shuffle=True, num_workers=1)

# Dump vocab
with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
	pk.dump(vocab, vocab_file)

# We need the dev and test sets in the original scale for evaluation
# dev_y_org = dev_y.astype(dataset.get_ref_dtype())
# test_y_org = test_y.astype(dataset.get_ref_dtype())


def mean0(ls):
    if isinstance(ls[0], list):
        islist = True
        mean = [0.0 for i in range(len(ls[0]))]
    else:
        islist = False
        mean = 0.0
    for i in range(len(ls)):
        if islist:
            for j in range(len(mean)):
                mean[j] += ls[i][j]
        else:
            mean += ls[i]
    if islist:
        for i in range(len(mean)):
            mean[i] /= len(ls)
    else:
        mean /= len(ls)
        mean = [mean]
    return mean


from nea.models import create_model
if args.loss == 'mse':
	lossty = nn.MSELoss()
	metric = 'mean_absolute_error'
else:
	lossty = nn.L1Loss()
	metric = 'mean_squared_error'

imv = mean0(traindataset.y)
model = create_model(args, overal_maxlen, vocab, np.array(imv))
print(model)
from nea.optimizers import get_optimizer
optimizer = get_optimizer(model.parameters(), args)


# evl = Evaluator(dataset, args.prompt_id, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org)

logger.info('--------------------------------------------------------------------------------------------------------------------------')
logger.info('Initial Evaluation:')
# evl.evaluate(model, -1, print_info=True)

total_train_time = 0
total_eval_time = 0
# print(overal_maxlen)

for ii in range(args.epochs):
	for i, data in enumerate(traindata):
		train_x, train_y, train_pmt, lens, paddingm = data

		train_y = np.array(train_y, dtype='float32')
		if args.prompt_id:
			train_pmt = np.array(train_pmt, dtype='int32')
		# Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
		train_y = Variable(torch.from_numpy(get_model_friendly_scores(train_y, train_pmt)))

		model.zero_grad()
		t0 = time()
		train_x = train_x.long()
		out,perm_ids=model(Variable(train_x),lens, mask=paddingm)
		# pdb.set_trace()
		lossm = lossty(out.squeeze(1),train_y[perm_ids].float())
		lossm.backward()

		optimizer.step()
		tr_time = time() - t0
		total_train_time += tr_time
		logger.info('Epoch %d, Iteration %d, train: %fs' % (ii, i, tr_time))
		logger.info('[Train] loss: %.4f' % lossm.data[0])

		
		# dev_y = np.array(dev_y, dtype='float32')
		# test_y = np.array(test_y, dtype='float32')
		# dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
		# test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
		# if args.prompt_id:
		# 	train_pmt = np.array(train_pmt, dtype='int32')
		# 	dev_pmt = np.array(dev_pmt, dtype='int32')
		# 	test_pmt = np.array(test_pmt, dtype='int32')
		# Evaluate
		# t0 = time()
		# evl.evaluate(model, ii)
		# evl_time = time() - t0
		# total_eval_time += evl_time
		
		# Print information
		# train_loss = train_history.history['loss'][0]
		# train_metric = train_history.history[metric][0]
		# logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time, evl_time))
		
		# evl.print_info()
	logger.info('Training:   %i seconds in total' % total_train_time)
	logger.info('Evaluation: %i seconds in total' % total_eval_time)
	# evl.print_final_info()
