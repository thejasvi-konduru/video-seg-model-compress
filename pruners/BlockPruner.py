import torch
import collections
import json
import numpy as np
import itertools

# from .Pruner import Pruner
# from .utils import write_array_to_file


def write_array_to_file(array, fh):
	for element in array:
		fh.write(str(element) + " ")
	fh.write("\n")

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


class BlockPrunerConfig:
	def __init__(self, sparsity, block_height, block_width, sub_rows, sub_cols, collapse_tensor):
		self.sparsity = sparsity
		self.block_height = block_height
		self.block_width = block_width

		self.sub_rows = sub_rows
		self.sub_cols = sub_cols

		self.collapse_tensor = collapse_tensor

	def __str__(self):
		return "{} {} {}".format(self.block_height, self.block_width, self.sparsity)


class BlockMatrix():
	def __init__(self, rows, cols, bh, bw, values, indices, rowBlockPtr):
		self.rows = rows
		self.cols = cols
		self.bh = bh
		self.bw = bw

		self.values = values
		self.indices = indices
		self.rowBlockPtr = rowBlockPtr

	def print(self):
		print("rows", self.rows)
		print("cols", self.cols)
		print("bh", self.bh)
		print("bw", self.bw)

		print("indices", self.values)
		print("indices", self.indices)
		print("rowBlockPtr", self.rowBlockPtr)

class BlockPruner(Pruner):
	"""docstring for BlockPruner"""
	def __init__(self, config_fp, on_gpu=True):
		super(BlockPruner, self).__init__(config_fp, on_gpu)

	def parse_config_file(self, config_fp):
		layer_configs = collections.OrderedDict()

		# Reading the configuration file
		with open(config_fp) as json_file:
			data = json.load(json_file)

			# Parsing through each layer set
			for ls_config in data["configs"]:
				layer_set = ls_config["layer_set"]
				for layer in layer_set:
					layer_configs[layer] = BlockPruner.generate_block_pruner_config(ls_config)
		print(layer_configs)
		return layer_configs

	def generate_masks(self, model, is_static=False, verbose=False):
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			pconfig = self.layer_configs[layer]
			
			# Generating mask
			if is_static:
				if verbose:
					print("Generating mask for layer {} using static approach".format(layer))
				mask = BlockPruner.generate_mask_by_construction(tensor.cpu().numpy(), pconfig)
			else:
				if verbose:
					print("Generating mask for layer {} using pruning approach".format(layer))
				mask = BlockPruner.generate_mask_by_pruning(tensor.cpu().numpy(), pconfig)
			
			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)


	@staticmethod
	def generate_block_pruner_config(block_config_dict):

		sparsity = block_config_dict["sparsity"]
		block_height = block_config_dict["block_height"]
		block_width  = block_config_dict["block_width"]

		sub_rows = block_config_dict["sub_rows"]
		sub_cols = block_config_dict["sub_cols"]

		collapse_tensor = block_config_dict["collapse_tensor"]

		return BlockPrunerConfig(sparsity, block_height, block_width, sub_rows, sub_cols, collapse_tensor)


	@staticmethod
	def generate_mask_by_pruning(tensor, pconfig, rev_mask=False):
		mask = BlockPruner.prune_tensor_as_block(tensor, pconfig.sparsity, 
						pconfig.block_height, pconfig.block_width,
						pconfig.sub_rows, pconfig.sub_cols, pconfig.collapse_tensor, rev_mask)
		return mask

	@staticmethod
	def prune_tensor_as_block(tensor, sparsity, block_height, block_width, sub_rows = -1, sub_cols = -1, collapse_tensor=True, rev_mask=False, dump_fpath=None):
		assert sparsity >= 0 and sparsity <= 1, "Sparsity should be within [0,1]"

		# Collapsing dimensions of the tensor
		mat = tensor.reshape(tensor.shape[0], tensor.size//tensor.shape[0])
		rows = mat.shape[0]
		cols = mat.shape[1]

		# Fixing rows dimension
		block_height = rows if block_height == -1 else block_height
		sub_rows = rows if sub_rows == -1 else sub_rows

		# Fixing columns dimension
		unit_size = tensor.size // (tensor.shape[0]*tensor.shape[1])
		if block_width == -1:
			block_width = cols
		else:
			if not collapse_tensor:				
				block_width *= unit_size

		if sub_cols == -1:
			sub_cols = cols
		else:
			if not collapse_tensor:
				sub_cols *= unit_size

		# Mask (1->Keep, 0->Don't keep)
		mask = np.zeros((rows,cols), dtype=mat.dtype)

		# Base case
		if (rows,cols) == (sub_rows,sub_cols):
			nrb = (rows+block_height-1)//block_height
			ncb = (cols+block_width-1)//block_width

			# Construct meta matrix
			if (block_height,block_width) == (1,1):
				meta_matrix = mat
			else:
				meta_matrix = np.zeros((nrb,ncb), dtype=mat.dtype)
				for rb in range(nrb):
					for cb in range(ncb):
						r_base   = rb*block_height
						r_offset = min(block_height, rows-rb*block_height)
						c_base   = cb*block_width
						c_offset = min(block_width, cols-cb*block_width)

						ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
						meta_matrix[rb,cb] = np.sum(np.abs(mat[ind_range]))
			
			# Prune
			if sparsity > 0: 
				# Sort the elements and return a mask
				thresh_ind = max(0, int(sparsity * meta_matrix.size) - 1)
				thresh_val = np.sort(np.abs(meta_matrix).flatten())[thresh_ind]

				if (block_height,block_width) == (1,1):
					# Setting the mask
					mask[np.abs(meta_matrix) > thresh_val] = 1;
				else:
					for rb,cb in itertools.product(range(nrb),range(ncb)):
						if np.abs(meta_matrix[rb,cb]) > thresh_val:
							r_base   = rb*block_height
							r_offset = min(block_height, rows-rb*block_height)
							c_base   = cb*block_width
							c_offset = min(block_width, cols-cb*block_width)

							ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
							mask[ind_range] = 1
			else:
				mask.fill(1)
		else:
			nrb = (rows+sub_rows-1)//sub_rows
			ncb = (cols+sub_cols-1)//sub_cols

			# Running the function on each unit.
			for rb in range(nrb):
				for cb in range(ncb):
					r_base   = rb*sub_rows
					r_offset = min(sub_rows, rows-rb*sub_rows)
					c_base   = cb*sub_cols
					c_offset = min(sub_cols, cols-cb*sub_cols)

					ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
					mat_sub = mat[ind_range]
					mask_sub = BlockPruner.prune_tensor_as_block(mat_sub, sparsity, block_height, block_width,
									 sub_rows, sub_cols, collapse_tensor=True, rev_mask=False)

					mask[ind_range] = mask_sub
		

		# Reverse mask if needed
		if rev_mask:
			mask = (mask + 1)%2

		if dump_fpath is not None:
			block_mat = BlockPruner.generate_block_matrix(mat*mask, block_height, block_width);
			BlockPruner.write_block_matrix_to_file(block_mat, dump_fpath)

		# Return the mask in the same shape of tensor
		mask = mask.reshape(tensor.shape)

		return mask


	@staticmethod
	def generate_mask_by_construction(tensor, pconfig, rev_mask=False):
		mask = BlockPruner.construct_tensor_as_block(tensor, pconfig.sparsity, 
						pconfig.block_height, pconfig.block_width,
						pconfig.sub_rows, pconfig.sub_cols, pconfig.collapse_tensor, rev_mask)
		return mask

	@staticmethod
	def construct_tensor_as_block(tensor, sparsity, block_height, block_width, sub_rows = -1, sub_cols = -1, collapse_tensor=True, rev_mask=False, dump_fpath=None):
		assert sparsity >= 0 and sparsity <= 1, "Sparsity should be within [0,1]"

		# Collapsing dimensions of the tensor
		mat = tensor.reshape(tensor.shape[0], tensor.size//tensor.shape[0])
		rows = mat.shape[0]
		cols = mat.shape[1]

		# Fixing rows dimension
		block_height = rows if block_height == -1 else block_height
		sub_rows = rows if sub_rows == -1 else sub_rows

		# Fixing columns dimension
		unit_size = tensor.size // (tensor.shape[0]*tensor.shape[1])
		if block_width == -1:
			block_width = cols
		else:
			if not collapse_tensor:
				block_width *= unit_size

		if sub_cols == -1:
			sub_cols = cols
		else:
			if not collapse_tensor:
				sub_cols *= unit_size

		# Mask (1->Keep, 0->Don't keep)
		mask = np.zeros((rows,cols), dtype=mat.dtype)

		# Base case
		if (rows,cols) == (sub_rows,sub_cols):
			nrb = (rows+block_height-1)//block_height
			ncb = (cols+block_width-1)//block_width

			if sparsity > 0: 
				# Generate
				meta_matrix = np.zeros((nrb,ncb), dtype=mat.dtype)
				nnzb = int((1.0 - sparsity) * meta_matrix.size)
				flat_bids = np.random.choice(meta_matrix.size, nnzb, replace=False)

				# Setting the non zero blocks
				meta_matrix.reshape(meta_matrix.size)[flat_bids] = 1
				meta_matrix = meta_matrix.reshape(nrb,ncb)

				if (block_height, block_width) == (1,1):
					mask[meta_matrix == 1] = 1
				else:
					for rb,cb in itertools.product(range(nrb),range(ncb)):
						if meta_matrix[rb,cb] == 1:
							r_base   = rb*block_height
							r_offset = min(block_height, rows-rb*block_height)
							c_base   = cb*block_width
							c_offset = min(block_width, cols-cb*block_width)

							ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
							mask[ind_range] = 1
			else:
				mask.fill(1)
		else:
			nrb = (rows+sub_rows-1)//sub_rows
			ncb = (cols+sub_cols-1)//sub_cols

			# Running the function on each unit.
			for rb in range(nrb):
				for cb in range(ncb):
					r_base   = rb*sub_rows
					r_offset = min(sub_rows, rows-rb*sub_rows)
					c_base   = cb*sub_cols
					c_offset = min(sub_cols, cols-cb*sub_cols)

					ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
					mat_sub = mat[ind_range]
					mask_sub = BlockPruner.construct_tensor_as_block(mat_sub, sparsity, block_height, block_width,
									 sub_rows, sub_cols, collapse_tensor=True, rev_mask=False)

					mask[ind_range] = mask_sub
		

		# Reverse mask if needed
		if rev_mask:
			mask = (mask + 1)%2

		if dump_fpath is not None:
			block_mat = BlockPruner.generate_block_matrix(mat*mask, block_height, block_width);
			BlockPruner.write_block_matrix_to_file(block_mat, dump_fpath)

		# Return the mask in the same shape of tensor
		mask = mask.reshape(tensor.shape)

		return mask

	@staticmethod
	def generate_block_matrix(mat, block_height, block_width):
		assert(len(mat.shape) == 2)
		rows,cols = mat.shape

		if block_height == 1 and block_width == 1:
			nnz = np.count_nonzero(mat)
			values = np.zeros(nnz, dtype=mat.dtype)
			indices = np.zeros(nnz, dtype=int)
			rowBlockPtr = np.zeros(rows+1, dtype=int)

			rindices,cindices = np.nonzero(mat)

			for id,(r,c) in enumerate(zip(rindices,cindices)):
				values[id]  = mat[r,c]
				indices[id] = c
				rowBlockPtr[r] += 1

		else:
			nrb = (rows+block_height-1)//block_height
			ncb = (cols+block_width-1)//block_width

			# Meta matrix
			meta_matrix = np.zeros((nrb,ncb), dtype=mat.dtype)

			for rb in range(nrb):
				for cb in range(ncb):
					r_base   = rb*block_height
					r_offset = min(block_height, rows-rb*block_height)
					c_base   = cb*block_width
					c_offset = min(block_width, cols-cb*block_width)

					ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
					meta_matrix[rb,cb] = np.sum(np.abs(mat[ind_range]))

			# Details
			nnzb = np.count_nonzero(meta_matrix)
			nnz = nnzb * block_height * block_width
			values = np.zeros(nnz, dtype=mat.dtype)
			indices = np.zeros(nnzb, dtype=int)
			rowBlockPtr = np.zeros(nrb+1, dtype=int)

			block_id = 0
			for rb in range(nrb):
				for cb in range(ncb):
					if meta_matrix[rb,cb] != 0:
						indices[block_id] = cb
						rowBlockPtr[rb] += 1

						r_base   = rb*block_height
						r_offset = min(block_height, rows-rb*block_height)
						c_base   = cb*block_width
						c_offset = min(block_width, cols-cb*block_width)

						ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
						
						spos = block_id * (block_height * block_width)
						epos = spos + (block_height * block_width)
						values[spos:epos] = mat[ind_range].flatten("F")

						# Moving to next block id
						block_id += 1


		# Convert counts to pointer
		rowBlockPtr[1:] = np.cumsum(rowBlockPtr[:-1])
		rowBlockPtr[0] = 0

		block_mat =  BlockMatrix(rows, cols, block_height, block_width, values, indices, rowBlockPtr)

		return block_mat

	@staticmethod
	def write_block_matrix_to_file(block_mat, filepath="block_data.txt"):
		# Number of non zero blocks
		nnzb = block_mat.rowBlockPtr[-1]

		fh = open(filepath, "w")

		fh.write(str(block_mat.rows) + "\n")
		fh.write(str(block_mat.cols) + "\n")
		fh.write(str(block_mat.bh) + "\n")
		fh.write(str(block_mat.bw) + "\n")
		fh.write(str(nnzb) + "\n")

		write_array_to_file(block_mat.values, fh);
		write_array_to_file(block_mat.indices, fh);
		write_array_to_file(block_mat.rowBlockPtr, fh);

		fh.close()



	def test():
		# Checking "prune_matrix_as_element" function
		rows = 8
		cols = 8
		sparsity = 0.5
		block_height = 2
		block_width = 2
		sub_rows = 4
		sub_cols = 4
		collapse_tensor = True
		
		print("Matrix dimensions : {} {}".format(rows,cols))
		print("Sub matrix dimensions : {} {}".format(sub_rows, sub_cols))
		print("Sparsity   : {}".format(sparsity*100))
		print("Block size : ({},{})".format(block_height, block_width))

		pconfig = BlockPrunerConfig(sparsity, block_height, block_width, sub_rows, sub_cols, collapse_tensor)

		# Generating random matrix
		arr = np.arange(rows*cols) + 1
		np.random.shuffle(arr)
		mat = arr.reshape(rows,cols)

		
		tensor = mat.reshape(rows, cols)
		mask = BlockPruner.generate_mask_by_pruning(tensor, pconfig)

		print(tensor.reshape(rows,cols))
		print()
		print((tensor*mask).reshape(rows,cols))

		block_mat = BlockPruner.generate_block_matrix(mat*mask, block_height, block_width)
		BlockPruner.write_block_matrix_to_file(block_mat, filepath="block_test.txt")
	
# config_path = "block_config.json"

# BlockPruner.test()
# pruner = BlockPruner(config_path)
# pruner.test()