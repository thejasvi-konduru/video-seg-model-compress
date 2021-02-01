from .Pruner import Pruner
from .BlockPruner import BlockPrunerConfig,BlockPruner

import torch
import collections
import json
import numpy as np
import itertools


class HbPrunerConfig:
	def __init__(self, block_configs):
		self.block_configs = block_configs

class HbPruner(Pruner):
	"""docstring for BlockPruner"""
	def __init__(self, config_fp, on_gpu=True):
		super(HbPruner, self).__init__(config_fp, on_gpu)

	def parse_config_file(self, config_fp):
		layer_configs = collections.OrderedDict()

		# Reading the configuration file
		with open(config_fp) as json_file:
			data = json.load(json_file)

			# Parsing through each layer set
			for ls_config in data["configs"]:
				layer_set = ls_config["layer_set"]
				json_block_configs = ls_config["levels"]

				for layer in layer_set:
					block_configs = []
					for json_block_config in json_block_configs:
						block_configs.append(BlockPruner.generate_block_pruner_config(json_block_config))

					layer_configs[layer] = HbPrunerConfig(block_configs)
					
		return layer_configs

	def generate_masks(self, model, is_static=False, verbose=False):
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			pconfig = self.layer_configs[layer]
			tensor = tensor.cpu().numpy()

			if verbose:
				print("Generating mask for layer {}".format(layer))

			# Generating mask
			mask = HbPruner.generate_mask(tensor, pconfig, is_static)

			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)

	@staticmethod
	def generate_mask(tensor, pconfig, is_static=False):
		final_mask = np.zeros(tensor.shape, dtype=tensor.dtype)
		for block_config in pconfig.block_configs:

			if is_static:
				mask = BlockPruner.generate_mask_by_construction(tensor, block_config)
			else:
				mask = BlockPruner.generate_mask_by_pruning(tensor, block_config)
			
			# Removing selected portion 
			tensor = tensor - mask*tensor	

			# Updating final mask
			final_mask = final_mask + mask

		return final_mask

	def test():
		# Checking "prune_matrix_as_element" function
		rows = 8
		cols = 8
		collapse_tensor = True

		# Level one
		sparsity = 0.5
		block_height = 2
		block_width = 2
		sub_rows = block_height
		sub_cols = -1
		level_1 = BlockPrunerConfig(sparsity, block_height, block_width, sub_rows, sub_cols, collapse_tensor)

		print("Level 1")
		print("Matrix dimensions : {} {}".format(rows,cols))
		print("Sub matrix dimensions : {} {}".format(sub_rows, sub_cols))
		print("Sparsity   : {}".format(sparsity*100))
		print("Block size : ({},{})".format(block_height, block_width))

		# Level two
		sparsity = 0.875
		block_height = 1
		block_width = 1
		sub_rows = block_height
		sub_cols = -1
		level_2 = BlockPrunerConfig(sparsity, block_height, block_width, sub_rows, sub_cols, collapse_tensor)
		
		print("Level 2")
		print("Matrix dimensions : {} {}".format(rows,cols))
		print("Sub matrix dimensions : {} {}".format(sub_rows, sub_cols))
		print("Sparsity   : {}".format(sparsity*100))
		print("Block size : ({},{})".format(block_height, block_width))

		pconfig = HbPrunerConfig([level_1, level_2])

		# Generating random matrix
		mat = np.arange(rows*cols) + 1
		np.random.shuffle(mat)

		# Reshapaing to a tensor
		tensor = mat.reshape(rows, cols//2, 2)
		tensor = mat.reshape(rows, cols)

		mask = HbPruner.generate_mask(tensor, pconfig, is_static=True)

		print(tensor.reshape(rows,cols))
		print()
		print((tensor*mask).reshape(rows,cols))