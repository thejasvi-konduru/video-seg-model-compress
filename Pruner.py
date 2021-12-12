import collections
import numpy as np

import torch

class Pruner(object):
	"""Super class for pruner"""
	def __init__(self, config_fp, on_gpu):
		super(Pruner, self).__init__()
		self.config_fp = config_fp
		self.on_gpu = on_gpu
		# Masks for each layer
		self.mask_dict = collections.OrderedDict()
		# Pruning configurations for each layer
		self.layer_configs = self.parse_config_file(config_fp)

	def apply_masks(self, model):
		with torch.no_grad():
			for layer in self.mask_dict:
				model.state_dict()[layer] *= self.mask_dict[layer]

	def print_stats(self):
		for layer in self.mask_dict:
			mask = self.mask_dict[layer]
			mask_np = mask.cpu().numpy()
			sp   = 1.0 - np.count_nonzero(mask_np)/mask_np.size
			print(layer, "sparsity = {}".format(sp*100))