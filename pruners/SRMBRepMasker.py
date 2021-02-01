import collections
import numpy as np
import json

import torch

from .Pruner import Pruner

class SRMBRepMaskerConfig():
	def __init__(self, obh, obw, cbh, cbw, ibh, ibw, osp, opat, isp, ipat, is_repetitive, collapse_tensor, cross_prob, is_symmetric):
		self.obh = obh # Outer block height
		self.obw = obw # Outer block width
		self.cbh = cbh # Core block height
		self.cbw = cbw # Core block width
		self.ibh = ibh # Inner block height
		self.ibw = ibw # Inner block width
		self.osp = osp # Outer sparsity
		self.opat = opat # Outer sparsity pattern
		self.isp = isp # Inner sparsity
		self.ipat = ipat # Inner sparsity pattern
		self.is_repetitive = is_repetitive # Is pattern of outer block repeated ?
		self.collapse_tensor = collapse_tensor # Collapse tensor to matrix or consider first two dimensions?

		self.cross_prob = cross_prob
		self.is_symmetric = is_symmetric

class SRMBRepMasker(Pruner):
	def __init__(self, config_fp, on_gpu=True):
		super(SRMBRepMasker, self).__init__(config_fp, on_gpu)

	def parse_config_file(self, config_fp):
		# Configs for each layer
		layer_configs = collections.OrderedDict()

		# Reading the configuration file
		with open(config_fp) as json_file:
			data = json.load(json_file)

			# Parsing through layer sets.
			for ls_config in data["configs"]:
				# Parsing through layers in a layer set.
				for layer in ls_config["layer_set"]:
					layer_configs[layer] = SRMBRepMaskerConfig(ls_config["obh"], ls_config["obw"], ls_config["cbh"], ls_config["cbw"],  \
														ls_config["ibh"], ls_config["ibw"], \
														ls_config["osp"], ls_config["opat"], \
														ls_config["isp"], ls_config["ipat"], \
														ls_config["is_repetitive"], ls_config["collapse_tensor"], \
														ls_config["cross_prob"], ls_config["is_symmetric"])

		return layer_configs

	def generate_masks(self, model, is_static=True, verbose=False):
		use_same_mask = False

		mask = None
		for layer in self.layer_configs:
			tensor = model.state_dict()[layer]
			exp_config = self.layer_configs[layer]
			

			# Generating mask
			if mask is None or not use_same_mask: 
				mask = SRMBRepMasker.construct_mask(tensor.cpu().numpy(), exp_config)

				if verbose:
					print("Generated mask for layer {}".format(layer))

			if use_same_mask and len(tensor.shape) == 2 and len(mask.shape) != 2:
				# Converting the mask to 2D
				mask = np.sum(mask, axis=(2,3))
				mask[mask > 0] = 1

			if self.on_gpu:
				self.mask_dict[layer] = torch.from_numpy(mask).cuda()
			else:
				self.mask_dict[layer] = torch.from_numpy(mask)

	@staticmethod
	def get_ramanujan_pattern(rows, cols, d, cross_prob=0.5, is_symmetric=False, debug=False):
		# d -> degree of left nodes or nnz per row.
		# As cols = dl*pow(2,lifts)
		assert(cols % d == 0) 
		assert(cols//d & (cols//d - 1) == 0) 
		# As rows = dr*pow(2,lifts)
		assert(rows//(cols//d) > 0)

		if is_symmetric:
			assert(rows == cols), ("When symmetric, #rows = #cols")

		# Final mask
		mask = np.zeros((rows,cols), dtype=int)

		# Initialization
		cur_rows = rows//(cols//d)
		cur_cols = d
		mask[:cur_rows, :cur_cols] = 1

		if debug:
			print(mask)
			print()

		while cur_cols < cols:
			# Cloning the pattern
			mask[cur_rows:2*cur_rows, cur_cols:2*cur_cols] = mask[:cur_rows, :cur_cols]

			# Lifting the pattern
			for l in range(cur_rows):
				s = l if is_symmetric else 0
				for r in range(s, cur_cols):
					if mask[l,r] == 1:
						cross = np.random.binomial(1, cross_prob)

						if cross == 1:
							if debug:
								print("Crossing {},{} and {},{}".format(l,r,l+cur_rows,r+cur_cols))
								print("Before crossing")
								print(mask)

							# Top right
							mask[l,r] = 0
							mask[l+cur_rows, r+cur_cols] = 0
							mask[l,r+cur_cols] = 1
							mask[l+cur_rows,r] = 1

							if is_symmetric:
								mask[r,l] = 0
								mask[r+cur_cols, l+cur_rows] = 0
								mask[r+cur_cols,l] = 1
								mask[r,l+cur_rows] = 1

							if debug:
								print("After crossing")
								print(mask)
								print()

			# Moving to next size
			cur_rows = 2*cur_rows
			cur_cols = 2*cur_cols


		if debug:
			print(mask)

		return mask


	@staticmethod
	def generate_sparsity_pattern(M, N, sparsity, pattern, cross_prob=0.5, is_symmetric=False):
		nnz = M*int((1.0 - sparsity) * N) 
		nnz_per_row = nnz//M

		# Mask for chosen pattern
		mask = np.zeros((M, N))

		# Avoiding long path
		if sparsity == 0:
			mask[:] = 1
			return mask

		if pattern == "RANDOM":
			flat_ids = np.random.choice(M*N, nnz, replace=False)
			mask.reshape(M*N)[flat_ids] = 1
		elif pattern == "UROW":
			assert(nnz%M == 0)
			for i in range(M):
				colInds = np.random.choice(N, nnz_per_row, replace=False)
				mask[i,colInds] = 1
		elif pattern == "RAMANUJAN":
			mask = SRMBRepMasker.get_ramanujan_pattern(M, N, nnz_per_row, cross_prob, is_symmetric)
		elif pattern == "TRANS":
			assert(nnz%M == 0)
			assert (M==N),"Matrix should be square"

			if nnz_per_row <= int(0.25 * N):
				print("Truly random")
				x_coords = np.arange(M)
				for k in range(nnz_per_row):
					success = False
					while not success:
						y_coords = np.random.permutation(M)
						if np.sum(mask[x_coords, y_coords]) == 0:
							mask[x_coords,y_coords] = 1
							success = True
			else:
				# All elements are present
				mask += 1
				
				v_degrees = np.ones(N, dtype=int)*M
				v_pool = np.arange(N)
				v_pool_size = N
				num_disconn = N - nnz_per_row

				for u in range(M):
					v_choice_flags = np.zeros(N)
					for i in range(num_disconn):
						# Choose a v with highest degree and that which was not chosen
						v_pool_clone = v_pool[:v_pool_size]
						v_pool_degrees = v_degrees[v_pool_clone]
						max_deg = np.max(v_pool_degrees)
						max_locs = np.where(v_pool_degrees == max_deg)[0]

						success = False
						while not success:
							max_ind = np.random.randint(max_locs.size)
							ind = max_locs[max_ind]
							v = v_pool_clone[ind]

							if v_choice_flags[v] == 0:
								# Remove the edge
								mask[u,v] = 0
								v_choice_flags[v] = 1
								v_degrees[v] -= 1

								if v_degrees[v] == nnz_per_row:
									# Remove vertex v from the pool
									last_v = v_pool[v_pool_size-1]
									v_pool[v_pool_size-1] = v_pool[ind]
									v_pool[ind] = last_v
									v_pool_size -= 1

								success = True
					
					#print(mask[:u+1].astype(int))
					#print(v_degrees)
					#print(v_pool[:v_pool_size])
			

			"""
			num_edges = M*N
			num_edges_to_keep = M*nnz_per_row
			num_edges_to_remove = num_edges - num_edges_to_keep
			edge_choices = np.arange(num_edges)

			print(num_edges, num_edges_to_keep, num_edges_to_remove)

			for i in range(num_edges_to_remove):
				print("Removing", i)
				success = False
				cur_guess_id = 0
				while not success:						
					# Choose a random edge
					edge_ind = np.random.randint(num_edges-i-cur_guess_id)
					edge_choice = edge_choices[edge_ind]
					
					v1 = edge_choice // N
					v2 = edge_choice % N

					if np.sum(mask[v1,:]) > nnz_per_row and np.sum(mask[:,v2]) > nnz_per_row:
						mask[v1,v2] = 0
						success = True

						# Swap edge
						last_edge = edge_choices[num_edges-i-1]
						edge_choices[num_edges-i-1] = edge_choice
						edge_choices[edge_ind] = last_edge
					else:
						last_edge = edge_choices[num_edges-i-cur_guess_id-1]
						edge_choices[num_edges-i-cur_guess_id-1] = edge_choice
						edge_choices[edge_ind] = last_edge

					cur_guess_id += 1

				print("Removed", i)
			"""
		elif pattern == "CDIA":
			assert(nnz%M == 0)
			base_colInds = np.random.choice(N, nnz_per_row, replace=False)
			for i in range(M):
				colInds = (i + base_colInds)%N
				mask[i,colInds] = 1
		elif pattern == "CDIASTRIDE":
			assert(nnz%M == 0)
			stride = N//nnz_per_row
			base_colInds = np.arange(0, N, stride)
			for i in range(M):
				colInds = (i + base_colInds)%N
				mask[i,colInds] = 1
		elif pattern == "COLUMN":
			assert(nnz%M == 0)
			base_colInds = np.random.choice(N, nnz_per_row, replace=False)
			mask[:,base_colInds] = 1
		elif pattern == "CBAND":
			assert(nnz%M == 0)
			k = nnz_per_row//2
			base_colInds = (np.arange(-k,k) + N)%N
			for i in range(M):
				colInds = (i + base_colInds)%N
				mask[i,colInds] = 1
		elif pattern == "CCDIA":
			assert(nnz%M == 0)
			base_colInds = np.arange(nnz_per_row) # Continuos cyclical diagonal
			for i in range(M):
				colInds = (i + base_colInds)%N
				mask[i,colInds] = 1
		elif pattern == "CCOLUMN":
			assert(nnz%M == 0)
			mask[:,:nnz_per_row] = 1
		elif pattern == "GROUP":
			num_groups = N//nnz_per_row
			stride_h = M//num_groups
			stride_w = nnz_per_row

			for g in range(num_groups):
				mask[g*stride_h:(g+1)*stride_h, g*stride_w:(g+1)*stride_w] = 1
		else:
			print("Unsupported {}".format(pattern))
			exit(-1)

		return mask


	@staticmethod
	def construct_mask(tensor, config):
		# Dimensions
		rows = tensor.shape[0]
		cols = tensor.shape[1]
		kernel_size = tensor.size // (rows*cols)
		if config.collapse_tensor:
			cols *= kernel_size
			kernel_size = 1

		# Modified stuff
		obh = rows if config.obh == -1 else config.obh
		obw = cols if config.obw == -1 else config.obw
		cbh = obh  if config.cbh == -1 else config.cbh
		cbw = obw  if config.cbw == -1 else config.cbw
		ibh,ibw = config.ibh, config.ibw

		if config.is_repetitive:
			OBmat = SRMBRepMasker.generate_sparsity_pattern(rows//obh, cols//obw, config.osp, config.opat, config.cross_prob, config.is_symmetric)
			CBmat = np.ones((obh//cbh, obw//cbw), dtype=tensor.dtype)
			Pmat  = SRMBRepMasker.generate_sparsity_pattern(cbh//ibh, cbw//ibw, config.isp, config.ipat, config.cross_prob, config.is_symmetric)
			IBmat = np.ones((ibh, ibw*kernel_size), dtype=tensor.dtype)

			mask_mat = np.kron(np.kron(OBmat, np.kron(CBmat, Pmat)), IBmat)
			mask = mask_mat.reshape(tensor.shape)
			mask = mask.astype(tensor.dtype)
		else:
			OBmat = SRMBRepMasker.generate_sparsity_pattern(rows//obh, cols//obw, config.osp, config.opat, config.cross_prob, config.is_symmetric)
			OCPmat = np.zeros((rows//ibh, cols//ibw), dtype=tensor.dtype)
			CBmat = np.ones((obh//cbh, obw//cbw), dtype=tensor.dtype)
			IBmat = np.ones((ibh, ibw*kernel_size), dtype=tensor.dtype)

			nrb, ncb = rows//obh, cols//obw
			core_nrb, core_ncb = cbh//ibh, cbw//ibw
			smbl_nrb, smbl_ncb = obh//ibh, obw//ibw
			for rb in range(nrb):
				for cb in range(ncb):
					if OBmat[rb,cb] == 1:
						Pmat = SRMBRepMasker.generate_sparsity_pattern(core_nrb, core_ncb, config.isp, config.ipat, config.cross_prob, config.is_symmetric)
						RepPmat = np.kron(CBmat, Pmat)
						# Setting in global mask
						OCPmat[rb*smbl_nrb:(rb+1)*smbl_nrb, cb*smbl_ncb:(cb+1)*smbl_ncb] += RepPmat

			mask_mat = np.kron(OCPmat, IBmat)
			mask = mask_mat.reshape(tensor.shape)

		return mask

	def test():
		import sys
		np.set_printoptions(threshold=sys.maxsize)

		ofm,ifm = 6,6
		obh,obw = 6,6
		cbh,cbw = obh,obw
		ibh,ibw = 1,1
		osp = 0
		opat = "UROW"
		isp = 0.625
		ipat = "TRANS"
		is_repetitive = True
		collapse_tensor = False

		config =  SRMBRepMaskerConfig(obh, obw, cbh, cbw, ibh, ibw, osp, opat, isp, ipat, is_repetitive, collapse_tensor)

		kh,kw = 1,1
		tensor = np.zeros((ofm,ifm,kh,kw))
		mask = SRMBRepMasker.construct_mask(tensor, config)

		print(mask.reshape(ofm, ifm*kh*kw).astype(int))
		print(np.sum(mask, axis=0).reshape(-1))
		print(np.sum(mask, axis=1).reshape(-1))
		print(np.sum(mask))
