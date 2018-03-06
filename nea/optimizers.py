import torch
import torch.optim as opt
def get_optimizer(params,args):

	clipvalue = 0
	clipnorm = 10

	if args.algorithm == 'rmsprop':
		optimizer = opt.RMSprop(params,lr=0.001, alpha=0.9, eps=1e-06)
	elif args.algorithm == 'sgd':
		optimizer = opt.SGD(params,lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
	elif args.algorithm == 'adagrad':
		optimizer = opt.Adagrad(params,lr=0.01)
	elif args.algorithm == 'adadelta':
		optimizer = opt.Adadelta(params,lr=1.0, rho=0.95, eps=1e-06)
	elif args.algorithm == 'adam':
		optimizer = opt.Adam(params,lr=0.001, betas=(0.9,0.999), eps=1e-08)
	elif args.algorithm == 'adamax':
		optimizer = opt.Adamax(params,lr=0.002,betas=(0.9,0.999), eps=1e-08)
	
	return optimizer
