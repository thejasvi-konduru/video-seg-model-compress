import collections
import numpy as np
import json

import torch

from .Pruner import Pruner
from .utils import write_array_to_file

class BlockletType(object):
	def __init__(self, bh, bw):
		self.bh = bh
		self.bw = bw
	def __str__(self):
		return "{}x{}".format(self.bh, self.bw)

class BlockletMatrix():
	def __init__(self, rows, cols, grb, gcb, bh, bw, values, indices):
		self.rows = rows
		self.cols = cols
		self.grb = grb
		self.gcb = gcb

		self.type = BlockletType(bh, bw)
		self.values = values
		self.indices = indices

class MultiBlockletMatrix():
	""" Multi Blocklet matrix data structure """
	def __init__(self, rows, cols, bl_mats, grb, gcb):
		self.rows = rows
		self.cols = cols
		
		# List of blocklet matrices
		self.bl_mats = bl_mats

		self.grb = grb
		self.gcb = gcb

	def print(self):
		for bl_id,bl_mat in enumerate(self.bl_mats):
			print("BL : ", bl_id)
			bl_mat.print()

class RMBMatrix():
	def __init__(self, rows, cols, bh, bw, mbl_mats, indices, rowBlockPtr):
		self.rows = rows
		self.cols = cols
		self.bh = bh
		self.bw = bw
		self.mbl_mats = mbl_mats

		self.indices = indices
		self.rowBlockPtr = rowBlockPtr

	def print(self):
		print("rows", self.rows)
		print("cols", self.cols)
		print("bh", self.bh)
		print("bw", self.bw)

		print("indices", self.indices)
		print("rowBlockPtr", self.rowBlockPtr)

		for mbl_id,mbl_mat in enumerate(self.mbl_mats):
			print("MBL : ", mbl_id)
			mbl_mat.print()

class RmbPrunerConfig(object):
	"""docstring for RMBPruning"""
	def __init__(self, bh, bw, spo, bl_types, bl_counts):
			self.bh = bh 
			self.bw = bw
			self.spo = spo
			self.bl_types = bl_types
			self.bl_counts = bl_counts

class RmbPruner(Pruner):
	def __init__(self, config_fp, on_gpu=True):
		super(RmbPruner, self).__init__(config_fp, on_gpu)

	def parse_config_file(self, config_fp):
		layer_configs = collections.OrderedDict()

		# Reading the configuration file
		with open(config_fp) as json_file:
			data = json.load(json_file)

			# Parsing through each layer set
			for ls_config in data["configs"]:
				global_bh = ls_config["global_bh"]
				global_bw = ls_config["global_bw"]
				global_sp = ls_config["global_sp"]

				# Reading blocklet information
				bl_types  = []
				bl_counts = []
				for bl_config in ls_config["blocklets"]:
					bl_type = BlockletType(bl_config["bh"], bl_config["bw"])
					bl_count = bl_config["count"]

					bl_types.append(bl_type)
					bl_counts.append(bl_count)

				layer_set = ls_config["layer_set"]
				for layer in layer_set:
					layer_configs[layer] = RmbPrunerConfig(global_bh, global_bw, global_sp, bl_types, bl_counts)

		return layer_configs

	def generate_masks(self, model, verbose=False):
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			rmb_pconfig = self.layer_configs[layer]

			if verbose:
				print("Generating mask for layer {}".format(layer))

			# Generating mask
			mask = RmbPruner.prune_tensor_as_rmb(tensor.cpu().numpy(), rmb_pconfig)

			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)

	@staticmethod
	def prune_tensor_as_rmb(tensor, config, dump_fpath=None):
		# Operate on the clone matrix
		mat = tensor.reshape(tensor.shape[0], -1).copy()
		mask = np.zeros(mat.shape, dtype=mat.dtype)

		rows = mat.shape[0]
		cols = mat.shape[1]
		bh = config.bh
		bw = config.bw

		assert rows%bh == 0, "Block height should divide rows"
		assert cols%bw == 0, "Block width should divide columns"

		nrb = rows // bh
		ncb = cols // bw

		### Taking care of outer sparsity
		meta_matrix_mask = np.ones((nrb,ncb), dtype=mat.dtype)
		if config.spo > 0:
			meta_matrix = np.zeros((nrb,ncb))
			if bh !=1 and bw != 1:
				for rb in range(nrb):
					for cb in range(ncb):
						loc_mat = mat[rb*bh:(rb+1)*bh , cb*bw:(cb+1)*bw]
						meta_matrix[rb,cb] = np.sum(np.abs(loc_mat))
			else:
				meta_matrix = np.abs(mat)

			for rb in range(nrb):
				# Get threshold value
				thresh_ind = int(config.spo*meta_matrix.shape[1])-1
				
				if thresh_ind >= 0:
					thresh_val = np.sort(np.abs(meta_matrix[rb].flatten()))[thresh_ind]

					# Keeping blocks which are beyond threshold
					meta_matrix_mask[rb][meta_matrix[rb] <= thresh_val] = 0
				
			"""
			# Get threshold value
			thresh_ind = int(config.spo*(meta_matrix.size-1))
			thresh_val = np.sort(np.abs(meta_matrix.flatten()))[thresh_ind]

			# Keeping blocks which are beyond threshold
			meta_matrix_mask[meta_matrix <= thresh_val] = 0
			"""

		### Taking care of inner sparsity
		bl_mats = []
		for rb in range(nrb):
			if rb % max(1,int(0.1*nrb)) == 0:
				print("Completed", "{} %".format(100.0*(rb/nrb)))
			for cb in range(ncb):
				if meta_matrix_mask[rb,cb] == 0:
					mask[rb*bh:(rb+1)*bh , cb*bw:(cb+1)*bw] = 0
					continue

				# Process if the block is not empty
				loc_mat = mat[rb*bh:(rb+1)*bh , cb*bw:(cb+1)*bw]

				# Loop through all blocklet types
				for bl_id,bl_type in enumerate(config.bl_types):
					bl_bh = config.bl_types[bl_id].bh
					bl_bw = config.bl_types[bl_id].bw

					for _ in range(config.bl_counts[bl_id]):
						# Pick a blocklet matrix
						bl_nrb = bh // bl_bh
						bl_ncb = bw // bl_bw

						# Initializing values and indices
						values = np.zeros((bh, bl_bw))
						indices = np.zeros(bl_nrb)

						val_buffer = np.zeros(bl_ncb)
						for bl_rb in range(bl_nrb):
							rb_mat = loc_mat[bl_rb*bl_bh:(bl_rb+1)*bl_bh]
							
							# Get representative element
							for bl_cb in range(bl_ncb):
								val_buffer[bl_cb] = np.sum(np.abs(rb_mat[:,bl_cb*bl_bw:(bl_cb+1)*bl_bw]))

							# Get the block with maximum value
							ch_bl_cb = np.argmax(val_buffer)

							# Setting values and indices
							values[bl_rb*bl_bh:(bl_rb+1)*bl_bh] = rb_mat[:,ch_bl_cb*bl_bw:(ch_bl_cb+1)*bl_bw]
							indices[bl_rb] = ch_bl_cb

							# Zeroing out values
							rb_mat[: , ch_bl_cb*bl_bw:(ch_bl_cb+1)*bl_bw] = 0

							# Setting the mask
							startRow =  rb*bh + (bl_rb*bl_bh)
							endRow =  startRow + bl_bh
							startCol = cb*bw + (ch_bl_cb*bl_bw)
							endCol = startCol + bl_bw

							mask[startRow:endRow, startCol:endCol] = 1.0


						# Create a blocklet matrix
						bl_mat = BlockletMatrix(bh,bw, rb,cb, bl_bh,bl_bw, values, indices)
						bl_mats.append(bl_mat)

		# Print pruned matrix
		#print((mat * mask).astype(int))

		if dump_fpath is not None:
			rmb_mat = RmbPruner.generate_rmb_mat_from_bl_mats(rows, cols, config.bh, config.bw, bl_mats)
			RmbPruner.write_rmb_matrix_to_file(rmb_mat, dump_fpath)

		# Reshpaing mask
		mask = mask.reshape(tensor.shape)

		return mask


	@staticmethod
	def generate_rmb_mat_from_bl_mats(rows, cols, bh, bw, bl_mats):
		nrb = rows // bh
		ncb = cols // bw

		# Order blocklets in row major fashion
		bl2mblIds = [bl_mat.grb*ncb + bl_mat.gcb for bl_mat in bl_mats]
		bl_mats =  [ bl_mats[id] for id in np.argsort(bl2mblIds)] # Ordering blocklets in row major
		bl2mblIds = np.sort(bl2mblIds) # mbl ids of new order

		# Get counts and pointer
		mblIds,bl_counts = np.unique(bl2mblIds, return_counts=True)
		bl_ptr = np.zeros(mblIds.size+1, dtype=int)
		bl_ptr[1:] = np.cumsum(bl_counts)
		bl_ptr[0] = 0

		# Construct mbl matrices
		mbl_mats = []
		indices = np.zeros(mblIds.size, dtype=int)
		rowBlockPtr = np.zeros(nrb+1, dtype=int)

		for i,mbl_id in enumerate(mblIds):
			grb = mbl_id // ncb
			gcb = mbl_id %  ncb
			mbl_mats.append(MultiBlockletMatrix(rows, cols, bl_mats[bl_ptr[i]:bl_ptr[i+1]], grb, gcb))

			indices[i] = gcb
			rowBlockPtr[grb] += 1

		# Convert counts to pointer
		rowBlockPtr[1:] = np.cumsum(rowBlockPtr[:-1])
		rowBlockPtr[0] = 0

		# Constructing RMB matrix
		rmb_mat = RMBMatrix(rows, cols, bh, bw, mbl_mats, indices, rowBlockPtr)

		return rmb_mat

	@staticmethod
	def write_rmb_matrix_to_file(rmb_mat, filepath = "rmb_data.txt"):
		# Number of row blocks and column blocks
		nrb = rmb_mat.rows // rmb_mat.bh
		ncb = rmb_mat.cols // rmb_mat.bw

		# Calculate nnz and nnzb
		nnzb = len(rmb_mat.mbl_mats)
		num_blets = 0
		nnz = 0
		num_indices = 0
		for mbl_mat in rmb_mat.mbl_mats:
			num_blets += len(mbl_mat.bl_mats)
			for bl_mat in mbl_mat.bl_mats:
				nnz += bl_mat.values.size
				num_indices += bl_mat.indices.size


		### Populating row_patterns, col_patterns
		row_patterns = np.zeros(num_blets, dtype=int)
		col_patterns = np.zeros(num_blets, dtype=int)

		# row_patterns, col_patterns
		cur_bl_id = 0
		for mbl_mat in rmb_mat.mbl_mats:
			for bl_mat in mbl_mat.bl_mats:
				row_patterns[cur_bl_id] = int(round(np.log2(bl_mat.rows//bl_mat.type.bh)))
				col_patterns[cur_bl_id] = int(round(np.log2(bl_mat.cols//bl_mat.type.bw)))
				cur_bl_id += 1


		### Populating valPtr, indPtr, bletPtr;
		valPtr  = np.zeros(nnzb+1, dtype=int)
		indPtr  = np.zeros(nnzb+1, dtype=int)
		bletPtr = np.zeros(nnzb+1, dtype=int)

		### valPtr,indPtr and bletPtr
		for mbl_id,mbl_mat in enumerate(rmb_mat.mbl_mats):
			mbl_nnz = 0
			mbl_num_indices = 0
			for bl_mat in mbl_mat.bl_mats:
				mbl_nnz += bl_mat.values.size
				mbl_num_indices += bl_mat.indices.size

			valPtr[mbl_id] = mbl_nnz
			indPtr[mbl_id] = mbl_num_indices
			bletPtr[mbl_id] = len(mbl_mat.bl_mats)

		# converting counts to ptr
		valPtr[1:] = np.cumsum(valPtr[:-1])
		valPtr[0] = 0
		indPtr[1:] = np.cumsum(indPtr[:-1])
		indPtr[0] = 0
		bletPtr[1:] = np.cumsum(bletPtr[:-1])
		bletPtr[0] = 0


		### Populating values and l_indices
		values = np.zeros(nnz)
		l_indices = np.zeros(num_indices, dtype=int)

		for mbl_id,mbl_mat in enumerate(rmb_mat.mbl_mats):
			valInd = valPtr[mbl_id]
			indInd = indPtr[mbl_id]
			for bl_mat in mbl_mat.bl_mats:
				blet_nnz = bl_mat.values.size
				blet_num_indices = bl_mat.indices.size
				values[valInd: valInd+blet_nnz] = bl_mat.values.flatten("F")
				l_indices[indInd: indInd+blet_num_indices] = bl_mat.indices.flatten("F")

				valInd += blet_nnz
				indInd += blet_num_indices

		fh = open(filepath, "w")

		fh.write(str(rmb_mat.rows) + "\n")
		fh.write(str(rmb_mat.cols) + "\n")
		fh.write(str(rmb_mat.bh) + "\n")
		fh.write(str(rmb_mat.bw) + "\n")
		fh.write(str(nnz) + "\n")
		fh.write(str(nnzb) + "\n")
		fh.write(str(num_blets) + "\n")
		fh.write(str(num_indices) + "\n")

		write_array_to_file(values, fh);
		write_array_to_file(rmb_mat.indices, fh);
		write_array_to_file(rmb_mat.rowBlockPtr, fh);
		write_array_to_file(row_patterns, fh);
		write_array_to_file(col_patterns, fh);
		write_array_to_file(l_indices, fh);
		write_array_to_file(valPtr, fh);
		write_array_to_file(indPtr, fh);
		write_array_to_file(bletPtr, fh);

		fh.close()



	def test():
		# Checking "prune_matrix_as_element" function
		rows = 8
		cols = 8
		global_sp = 0.5
		global_bh = 4
		global_bw = 4

		bl_types = [BlockletType(2,2), BlockletType(1,1)]
		bl_counts = [1,1]

		print("Matrix dimensions : {} {}".format(rows,cols))
		print("Global block dimensions : {} {}".format(global_bh, global_bw))
		print("Global sparsity : {}".format(global_sp))

		# Hbs pruner
		pconfig = RmbPrunerConfig(global_bh, global_bw, global_sp, bl_types, bl_counts)

		mat = np.arange(rows*cols)
		np.random.shuffle(mat)
		mat = mat.reshape(rows,cols)

		# Generating mask
		mask = RmbPruner.prune_tensor_as_rmb(mat, pconfig, "rmb_data.txt")

		print(mat)
		print(mat*mask)