import numpy as np

def write_array_to_file(array, fh):
	for element in array:
		fh.write(str(element) + " ")
	fh.write("\n")


def get_meta_matrix(mat, block_height, block_width):
	assert (len(mat.shape) == 2)

	if block_height == 1 and block_width == 1:
		meta_matrix = np.copy(mat)
	else:
		rows, cols = mat.shape[0], mat.shape[1]
		nrb = rows // block_height
		ncb = cols // block_width

		meta_matrix = np.zeros((nrb,ncb), dtype=mat.dtype)

		for rb in range(nrb):
			for cb in range(ncb):
				r_base   = rb*block_height
				r_offset = min(block_height, rows-rb*block_height)
				c_base   = cb*block_width
				c_offset = min(block_width, cols-cb*block_width)

				ind_range = np.ix_(range(r_base, r_base+r_offset), range(c_base, c_base+c_offset))
				meta_matrix[rb,cb] = np.sum(np.abs(mat[ind_range]))

	return meta_matrix

"""
def write_rmb_to_file(rows, cols, bh, bw, bl_mats, filename="rmb_data.txt"):
	# Number of row blocks and column blocks
	nrb = rows//bh
	ncb = cols//bw

	### Reordering blocklet matrices in row major order
	order = np.argsort([bl_mat.grb*ncb + bl_mat.gcb for bl_mat in bl_mats])
	bl_mats_new = [bl_mats[entry] for entry in order]
	bl_mats = bl_mats_new # Over riding

	# Meta matrix with blocklet counts
	meta_mat = np.zeros((nrb,ncb), dtype=int)
	for bl_mat in bl_mats:
		meta_mat[bl_mat.grb, bl_mat.gcb] += 1

	### Calculating nnz and nnzb
	num_blets = len(bl_mats)
	nnzb = np.count_nonzero(meta_mat)
	nnz = 0
	num_indices = 0
	for bl_mat in bl_mats:
		nnz += bl_mat.values.size
		num_indices += bl_mat.indices.size

	### Populating indices and rowBlockPtr
	indices = np.zeros(nnzb, dtype=int)
	rowBlockPtr = np.zeros(nrb+1, dtype=int)

	# indices
	cur_bid = 0
	for rb in range(nrb):
		for cb in range(ncb):
			if meta_mat[rb,cb] != 0:
				indices[cur_bid] = cb 
				rowBlockPtr[rb] += 1
				cur_bid += 1

	# row block ptr
	rowBlockPtr[1:] = np.cumsum(rowBlockPtr[:-1])
	rowBlockPtr[0] = 0;
	
	### Populating row_patterns, col_patterns
	row_patterns = np.zeros(num_blets, dtype=int)
	col_patterns = np.zeros(num_blets, dtype=int)

	# row_patterns, col_patterns
	for bl_id,bl_mat in enumerate(bl_mats):
		row_patterns[bl_id] = int(round(np.log2(bl_mat.rows//bl_mat.bh)))
		col_patterns[bl_id] = int(round(np.log2(bl_mat.cols//bl_mat.bw)))
		 
	
	### Populating valPtr, indPtr, bletPtr;
	valPtr  = np.zeros(nnzb+1, dtype=int)
	indPtr  = np.zeros(nnzb+1, dtype=int)
	bletPtr = np.zeros(nnzb+1, dtype=int)

	# bletPtr
	bletPtr[1:] = np.cumsum(meta_mat[np.nonzero(meta_mat)])
	bletPtr[0] = 0

	for mbl_id in range(nnzb):
		start_bl_id = bletPtr[mbl_id]
		end_bl_id = bletPtr[mbl_id+1]
		valPtr[mbl_id] = np.sum([bl_mats[bl_id].values.size for bl_id in range(bletPtr[mbl_id], bletPtr[mbl_id+1])])
		indPtr[mbl_id] = np.sum([bl_mats[bl_id].indices.size for bl_id in range(bletPtr[mbl_id], bletPtr[mbl_id+1])])

	# valPtr,indPtr
	valPtr[1:] = np.cumsum(valPtr[:-1])
	valPtr[0]  = 0
	indPtr[1:] = np.cumsum(indPtr[:-1])
	indPtr[0]  = 0

	### Populating values and l_indices
	values = np.zeros(nnz)
	l_indices = np.zeros(num_indices, dtype=int)

	for mbl_id in range(nnzb):
		valInd = valPtr[mbl_id]
		indInd = indPtr[mbl_id]
		for bl_id in range(bletPtr[mbl_id],bletPtr[mbl_id+1]):
			blet_nnz = bl_mats[bl_id].values.size
			blet_indices = bl_mats[bl_id].indices.size
			values[valInd: valInd+blet_nnz] = bl_mats[bl_id].values.flatten("F")
			l_indices[indInd: indInd+blet_indices] = bl_mats[bl_id].indices.flatten("F")

			valInd += blet_nnz
			indInd += blet_indices


	fh = open(filename, "w")

	fh.write(str(rows) + "\n")
	fh.write(str(cols) + "\n")
	fh.write(str(bh) + "\n")
	fh.write(str(bw) + "\n")
	fh.write(str(nnz) + "\n")
	fh.write(str(nnzb) + "\n")
	fh.write(str(num_blets) + "\n")
	fh.write(str(num_indices) + "\n")

	write_array(fh, values);
	write_array(fh, indices);
	write_array(fh, rowBlockPtr);
	write_array(fh, row_patterns);
	write_array(fh, col_patterns);
	write_array(fh, l_indices);
	write_array(fh, valPtr);
	write_array(fh, indPtr);
	write_array(fh, bletPtr);

	fh.close()
"""