import collections
import numpy as np
import json
import random

import torch

from .Pruner import Pruner
from .utils import write_array_to_file, get_meta_matrix

class BlockletType(object):
	def __init__(self, bh, bw):
		self.bh = bh
		self.bw = bw
	def __str__(self):
		return "{}x{}".format(self.bh, self.bw)

class CyDiaBlockletMatrix():
	def __init__(self, rows, cols, grb, gcb, bh, bw, values, offset):
		self.rows = rows
		self.cols = cols
		self.grb = grb
		self.gcb = gcb

		self.type = BlockletType(bh, bw)
		self.values = values
		self.offset = offset

	def print(self):
		print("rows", self.rows)
		print("cols", self.cols)
		print("bh", self.bh)
		print("bw", self.bw)

		print("values", self.values)
		print("offset", self.offset)

class MultiCyDiaBlockletMatrix():
	""" Multi Blocklet matrix data structure """
	def __init__(self, rows, cols, cdbl_mats, grb, gcb):
		self.rows = rows
		self.cols = cols
		
		# List of blocklet matrices
		self.cdbl_mats = cdbl_mats

		self.grb = grb
		self.gcb = gcb

	def print(self):
		for cdbl_id,cdbl_mat in enumerate(self.cdbl_mats):
			print("CDBL : ", cdbl_id)
			cdbl_mat.print()

class RMCDBMatrix():
	def __init__(self, rows, cols, bh, bw, mcdbl_mats, indices, rowBlockPtr):
		self.rows = rows
		self.cols = cols
		self.bh = bh
		self.bw = bw
		self.mcdbl_mats = mcdbl_mats

		self.indices = indices
		self.rowBlockPtr = rowBlockPtr

	def print(self):
		print("rows", self.rows)
		print("cols", self.cols)
		print("bh", self.bh)
		print("bw", self.bw)

		print("indices", self.indices)
		print("rowBlockPtr", self.rowBlockPtr)

		for mcdbl_id,mcdbl_mat in enumerate(self.mcdbl_mats):
			print("MBL : ", mcdbl_id)
			mcdbl_mat.print()

class RmcdbPrunerConfig(object):
	"""docstring for RMCDBPruning"""
	def __init__(self, bh, bw, spo, bl_types, bl_counts, collapse_tensor):
			self.bh = bh 
			self.bw = bw
			self.spo = spo
			self.bl_types = bl_types
			self.bl_counts = bl_counts
			self.collapse_tensor = collapse_tensor

class RmcdbPruner(Pruner):
	def __init__(self, config_fp, on_gpu=True):
		super(RmcdbPruner, self).__init__(config_fp, on_gpu)

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
				collapse_tensor = ls_config["collapse_tensor"]

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
					layer_configs[layer] = RmcdbPrunerConfig(global_bh, global_bw, global_sp, bl_types, bl_counts, collapse_tensor)

		return layer_configs

	def generate_masks(self, model, is_static=False, verbose=False):
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			rmcdb_pconfig = self.layer_configs[layer]

			# Generating mask
			if is_static:
				if verbose:
					print("Generating mask for layer {} using static approach".format(layer))
				mask = RmcdbPruner.construct_rmcdb_matrix(tensor.cpu().numpy(), rmcdb_pconfig)
			else:
				if verbose:
					print("Generating mask for layer {} using pruning approach".format(layer))
				mask = RmcdbPruner.prune_tensor_as_rmcdb(tensor.cpu().numpy(), rmcdb_pconfig)

			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)


	@staticmethod
	def construct_rmcdb_matrix(tensor, config):
		rows = tensor.shape[0]
		cols = tensor.size // tensor.shape[0]
		bh = config.bh # Block height
		bw = config.bw # Block width

		assert rows%bh == 0, "Block height should divide rows"
		assert cols%bw == 0, "Block width should divide columns"

		nrb = rows // bh # Number of row blocks
		ncb = cols // bw # Number of column blocks

		# Mask of the structure
		mask = np.zeros((rows,cols), dtype=tensor.dtype)

		# Randomly pick outside blocks
		meta_matrix_mask = np.ones((nrb,ncb), dtype=tensor.dtype)
		if config.spo > 0:
			### Equal sparsity in rows ###
			# Number of zero blocks in a row block
			nzb_in_rb = int(config.spo*meta_matrix_mask.shape[1])
			zero_cb_ids = np.random.choice(ncb, nzb_in_rb, replace=False)
			meta_matrix_mask[rb,zero_cb_ids] = 0

		### Taking care of inner sparsity
		cdbl_mats = []
		for rb in range(nrb):
			if rb % max(1,int(0.1*nrb)) == 0:
				print("Completed", "{} %".format(100.0*(rb/nrb)))
			for cb in range(ncb):
				if meta_matrix_mask[rb,cb] == 0:
					mask[rb*bh:(rb+1)*bh , cb*bw:(cb+1)*bw] = 0
					continue

				# Loop through all blocklet types
				for mcdbl_id,bl_type in enumerate(config.bl_types):
					bl_bh = bl_type.bh
					bl_bw = bl_type.bw
					bl_count = config.bl_counts[mcdbl_id]

					assert bh%bl_bh == 0, "Block height should divide rows in a blocklet"
					assert bw%bl_bw == 0, "Block width should divide columns in a blocklet"

					# Number of row and column blocks in a blocklet
					bl_nrb = bh // bl_bh
					bl_ncb = bw // bl_bw

					# Choosen diagonals 
					row_indices  = np.arange(bl_nrb)
					ch_dias = np.random.choice(bl_ncb, bl_count, replace=False)

					for ch_dia in ch_dias:
						cur_col_indices = (row_indices + ch_dia)%bl_ncb
						for bl_rb, bl_cb in zip(row_indices, cur_col_indices):

							# Setting the mask
							startRow =  rb*bh + (bl_rb*bl_bh)
							endRow =  startRow + bl_bh
							startCol = cb*bw + (bl_cb*bl_bw)
							endCol = startCol + bl_bw

							mask[startRow:endRow, startCol:endCol] = 1
		
		# Reshpaing mask
		mask = mask.reshape(tensor.shape)

		return mask

	@staticmethod
	def prune_tensor_as_rmcdb(tensor, config, dump_fpath=None):
		# Operate on the clone matrix
		mat = tensor.reshape(tensor.shape[0], -1).copy()
		mask = np.zeros(mat.shape, dtype=mat.dtype)

		rows = mat.shape[0]
		cols = mat.shape[1] 
		bh = config.bh # Block height
		bw = config.bw # Block width

		assert rows%bh == 0, "Block height should divide rows"
		assert cols%bw == 0, "Block width should divide columns"

		nrb = rows // bh # Number of row blocks
		ncb = cols // bw # Number of column blocks

		### Taking care of outer sparsity
		meta_matrix_mask = np.ones((nrb,ncb), dtype=mat.dtype)
		if config.spo > 0:
			meta_matrix = get_meta_matrix(mat, bh, bw)

			### Equal sparsity in rows ###
			for rb in range(nrb):
				# Get threshold value
				thresh_ind = int(config.spo*meta_matrix.shape[1])-1
				
				if thresh_ind >= 0:
					# Retaining blocks which are above a threshold
					thresh_val = np.sort(np.abs(meta_matrix[rb].flatten()))[thresh_ind]
					meta_matrix_mask[rb][meta_matrix[rb] <= thresh_val] = 0
				
		### Taking care of inner sparsity
		cdbl_mats = []
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
				for mcdbl_id,bl_type in enumerate(config.bl_types):
					bl_bh = bl_type.bh
					bl_bw = bl_type.bw
					bl_count = config.bl_counts[mcdbl_id]

					assert bh%bl_bh == 0, "Block height should divide rows in a blocklet"
					assert bw%bl_bw == 0, "Block width should divide columns in a blocklet"

					# Number of row and column blocks in a blocklet
					bl_nrb = bh // bl_bh
					bl_ncb = bw // bl_bw

					# Construct meta matrix
					meta_loc_mat = get_meta_matrix(loc_mat, bl_bh, bl_bw)

					# Calculate scores for diagonals
					dia_scores = np.zeros(bl_ncb)
					row_indices  = np.arange(bl_nrb)
					base_col_indices = row_indices%bl_ncb
					for bl_cb in range(bl_ncb):
						cur_col_indices = (base_col_indices + bl_cb)%bl_ncb
						dia_scores[bl_cb] = np.sum(meta_loc_mat[row_indices,cur_col_indices])

					# Choosen diagonals 
					ch_dias = np.argsort(dia_scores)[::-1][:bl_count]

					for ch_dia in ch_dias:
						values = np.zeros((bh,bl_bw), dtype=mat.dtype)
						cur_col_indices = (row_indices + ch_dia)%bl_ncb
						for bl_rb, bl_cb in zip(row_indices, cur_col_indices):
							# Populating values
							values[bl_rb*bl_bh:(bl_rb+1)*bl_bh] = loc_mat[bl_rb*bl_bh:(bl_rb+1)*bl_bh, bl_cb*bl_bw:(bl_cb+1)*bl_bw]

							# Zeroing out values
							loc_mat[bl_rb*bl_bh:(bl_rb+1)*bl_bh][bl_cb*bl_bw:(bl_cb+1)*bl_bw] = 0

							# Setting the mask
							startRow =  rb*bh + (bl_rb*bl_bh)
							endRow =  startRow + bl_bh
							startCol = cb*bw + (bl_cb*bl_bw)
							endCol = startCol + bl_bw

							mask[startRow:endRow, startCol:endCol] = 1

							cdbl_mat = CyDiaBlockletMatrix(bh,bw, rb,cb, bl_bh,bl_bw, values, ch_dia)
							cdbl_mats.append(cdbl_mat)

		# Print pruned matrix
		#print((mat * mask).astype(int))

		if dump_fpath is not None:
			rmcdb_mat = RmcdbPruner.generate_rmcdb_mat_from_cdbl_mats(rows, cols, config.bh, config.bw, cdbl_mats)
			RmcdbPruner.write_rmcdb_matrix_to_file(rmcdb_mat, dump_fpath)

		# Reshpaing mask
		mask = mask.reshape(tensor.shape)

		return mask


	@staticmethod
	def generate_rmcdb_mat_from_cdbl_mats(rows, cols, bh, bw, cdbl_mats):
		nrb = rows // bh
		ncb = cols // bw

		# Order blocklets in row major fashion
		bl2mblIds = [cdbl_mat.grb*ncb + cdbl_mat.gcb for cdbl_mat in cdbl_mats]
		cdbl_mats =  [ cdbl_mats[id] for id in np.argsort(bl2mblIds)] # Ordering blocklets in row major
		bl2mblIds = np.sort(bl2mblIds) # mbl ids of new order

		# Get counts and pointer
		mblIds,bl_counts = np.unique(bl2mblIds, return_counts=True)
		bl_ptr = np.zeros(mblIds.size+1, dtype=int)
		bl_ptr[1:] = np.cumsum(bl_counts)
		bl_ptr[0] = 0

		# Construct mbl matrices
		mcdbl_mats = []
		indices = np.zeros(mblIds.size, dtype=int)
		rowBlockPtr = np.zeros(nrb+1, dtype=int)

		for i,mcdbl_id in enumerate(mblIds):
			grb = mcdbl_id // ncb
			gcb = mcdbl_id %  ncb
			mcdbl_mats.append(MultiCyDiaBlockletMatrix(rows, cols, cdbl_mats[bl_ptr[i]:bl_ptr[i+1]], grb, gcb))

			indices[i] = gcb
			rowBlockPtr[grb] += 1

		# Convert counts to pointer
		rowBlockPtr[1:] = np.cumsum(rowBlockPtr[:-1])
		rowBlockPtr[0] = 0

		# Constructing RMB matrix
		rmcdb_mat = RMCDBMatrix(rows, cols, bh, bw, mcdbl_mats, indices, rowBlockPtr)

		return rmcdb_mat

	@staticmethod
	def write_rmcdb_matrix_to_file(rmcdb_mat, filepath = "rmcdb_data.txt"):
		# Number of row blocks and column blocks
		nrb = rmcdb_mat.rows // rmcdb_mat.bh
		ncb = rmcdb_mat.cols // rmcdb_mat.bw

		# Calculate nnz and nnzb
		nnzb = len(rmcdb_mat.mcdbl_mats)
		num_blets = 0
		nnz = 0
		for mcdbl_mat in rmcdb_mat.mcdbl_mats:
			num_blets += len(mcdbl_mat.cdbl_mats)
			for cdbl_mat in mcdbl_mat.cdbl_mats:
				nnz += cdbl_mat.values.size

		### Populating row_patterns, col_patterns
		row_patterns = np.zeros(num_blets, dtype=int)
		col_patterns = np.zeros(num_blets, dtype=int)

		# row_patterns, col_patterns
		cur_cdbl_id = 0
		for mcdbl_mat in rmcdb_mat.mcdbl_mats:
			for cdbl_mat in mcdbl_mat.cdbl_mats:
				row_patterns[cur_cdbl_id] = cdbl_mat.type.bh
				col_patterns[cur_cdbl_id] = cdbl_mat.type.bw
				cur_cdbl_id += 1


		### Populating valPtr, indPtr, bletPtr;
		valPtr  = np.zeros(nnzb+1, dtype=int)
		bletPtr = np.zeros(nnzb+1, dtype=int)

		### valPtr and bletPtr
		for mcdbl_id,mcdbl_mat in enumerate(rmcdb_mat.mcdbl_mats):
			mbl_nnz = 0
			for cdbl_mat in mcdbl_mat.cdbl_mats:
				mbl_nnz += cdbl_mat.values.size

			valPtr[mcdbl_id] = mbl_nnz
			bletPtr[mcdbl_id] = len(mcdbl_mat.cdbl_mats)

		# converting counts to ptr
		valPtr[1:] = np.cumsum(valPtr[:-1])
		valPtr[0] = 0
		bletPtr[1:] = np.cumsum(bletPtr[:-1])
		bletPtr[0] = 0


		### Populating values and offsets
		values = np.zeros(nnz)
		offsets = np.zeros(num_blets, dtype=int)

		for mcdbl_id,mcdbl_mat in enumerate(rmcdb_mat.mcdbl_mats):
			valInd = valPtr[mcdbl_id]
			offInd = bletPtr[mcdbl_id]
			for cdbl_mat in mcdbl_mat.cdbl_mats:
				blet_nnz = cdbl_mat.values.size
				values[valInd: valInd+blet_nnz] = cdbl_mat.values.flatten("F")
				offsets[offInd] = cdbl_mat.offset

				valInd += blet_nnz
				offInd += 1

		fh = open(filepath, "w")

		fh.write(str(rmcdb_mat.rows) + "\n")
		fh.write(str(rmcdb_mat.cols) + "\n")
		fh.write(str(rmcdb_mat.bh) + "\n")
		fh.write(str(rmcdb_mat.bw) + "\n")
		fh.write(str(nnz) + "\n")
		fh.write(str(nnzb) + "\n")
		fh.write(str(num_blets) + "\n")

		write_array_to_file(values, fh);
		write_array_to_file(rmcdb_mat.indices, fh);
		write_array_to_file(rmcdb_mat.rowBlockPtr, fh);
		write_array_to_file(row_patterns, fh);
		write_array_to_file(col_patterns, fh);
		write_array_to_file(offsets, fh);
		write_array_to_file(valPtr, fh);
		write_array_to_file(bletPtr, fh);

		fh.close()



	def test():
		# Checking "prune_matrix_as_element" function
		rows = 8
		cols = 8
		global_sp = 0
		global_bh = 4
		global_bw = 4

		bl_types = [BlockletType(1,1)]
		bl_counts = [1]

		print("Matrix dimensions : {} {}".format(rows,cols))
		print("Global block dimensions : {} {}".format(global_bh, global_bw))
		print("Global sparsity : {}".format(global_sp))

		# Hbs pruner
		pconfig = RmcdbPrunerConfig(global_bh, global_bw, global_sp, bl_types, bl_counts)

		mat = np.arange(rows*cols) + 1
		np.random.shuffle(mat)
		mat = mat.reshape(rows,cols)

		# Generating mask
		mask = RmcdbPruner.prune_tensor_as_rmcdb(mat, pconfig, "rmcdb_data.txt")

		print(mat)
		print(mat*mask)
