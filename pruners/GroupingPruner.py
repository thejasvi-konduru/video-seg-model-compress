import collections
import numpy as np
import json

import torch

from .Pruner import Pruner

class GroupingPrunerConfig():
	def __init__(self, num_groups):
		self.num_groups = num_groups

class GroupingPruner(Pruner):
	def __init__(self, config_fp, on_gpu=True):
		super(GroupingPruner, self).__init__(config_fp, on_gpu)

	def parse_config_file(self, config_fp):
		layer_configs = collections.OrderedDict()

		# Reading the configuration file
		with open(config_fp) as json_file:
			data = json.load(json_file)

			pruner_type = data["pruner_type"]

			# Parsing through each layer set
			for ls_config in data["configs"]:
				num_groups = ls_config["num_groups"]

				for layer in ls_config["layer_set"]:
					layer_configs[layer] = GroupingPrunerConfig(num_groups)

		return layer_configs

	def generate_masks(self, model, is_static=True, verbose=False):
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			exp_config = self.layer_configs[layer]

			if verbose:
				print("Generating mask for layer {}".format(layer))

			# Generating mask
			mask = GroupingPruner.construct_mask(tensor.cpu().numpy(), exp_config)

			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)

	@staticmethod
	def construct_mask(tensor, config):
		mask = np.zeros(tensor.shape, dtype=tensor.dtype)

		ofm_stride = tensor.shape[0] // config.num_groups
		ifm_stride = tensor.shape[1] // config.num_groups

		for gid in range(config.num_groups):
			mask[gid*ofm_stride:(gid+1)*ofm_stride, gid*ifm_stride:(gid+1)*ifm_stride] = 1

		return mask

	def test():
		num_groups = 4
		ofm,ifm = 4,4
		kh,kw  = 1,3
		
		pconfig = GroupingPrunerConfig(num_groups)

		tensor = np.zeros((ofm,ifm,kh,kw))
		mask = GroupingPruner.construct_mask(tensor, pconfig)
		#print(mask)
		print(mask.reshape(ofm, ifm*kh*kw).astype(int))












