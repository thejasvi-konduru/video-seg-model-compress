import matplotlib.pyplot as plt
import numpy as np

import pruners.SRMBRepMasker

def get_adjacency_matrix(inc_mat):
	adj_mat = np.zeros((inc_mat.shape[0]+inc_mat.shape[1], inc_mat.shape[0]+inc_mat.shape[1]), dtype=int)
	adj_mat[:inc_mat.shape[0], inc_mat.shape[0]:] = inc_mat
	adj_mat[inc_mat.shape[0]:, :inc_mat.shape[0]] = np.transpose(inc_mat)

	return adj_mat
	

if __name__ == "__main__":
	# Graph configs
	g1_nlv, g1_nrv, g1_d = 6,6, 3
	g2_nlv, g2_nrv, g2_d = 6,6, 3

	# Calculate Eigen values of  input graphs
	print("Generating first Ramanujan bipartite graph G")
	g1_inc_mat = pruners.SRMBRepMasker.SRMBRepMasker.get_ramanujan_pattern(g1_nlv, g1_nrv, g1_d)
	g1_adj_mat = get_adjacency_matrix(g1_inc_mat)

	print("Generating second Ramanujan bipartite graph H")
	g2_inc_mat = pruners.SRMBRepMasker.SRMBRepMasker.get_ramanujan_pattern(g2_nlv, g2_nrv, g2_d)
	g2_adj_mat = get_adjacency_matrix(g2_inc_mat)

	print("Bipartite graph product")
	# Bipartite graph product graph
	gbp_inc_mat  = np.kron(g1_inc_mat, g2_inc_mat)
	gbp_other_inc_mat = np.kron(g1_inc_mat, np.transpose(g2_inc_mat))
	gbp_adj_mat = get_adjacency_matrix(gbp_inc_mat)

	# Tensor graph product graph
	print("Tensor product")
	gtp_adj_mat  = np.kron(g1_adj_mat, g2_adj_mat)


	# Rearranged Tensor product
	gtp_rarngd_adj_mat = np.zeros_like(gtp_adj_mat)


	base_r, off_r = 0,g1_nlv*g2_nlv
	base_c, off_c = g1_nlv*g2_nlv, g1_nrv*g2_nrv
	gtp_rarngd_adj_mat[base_r:base_r+off_r, base_c:base_c+off_c] = gbp_inc_mat
	base_r = g1_nlv*g2_nlv + g1_nrv*g2_nrv
	base_c = base_r + g1_nlv*g2_nrv
	off_r = g1_nlv*g2_nrv
	off_c = g1_nrv*g2_nlv
	gtp_rarngd_adj_mat[base_r:base_r+off_r, base_c:base_c+off_c] = gbp_other_inc_mat
	gtp_rarngd_adj_mat = gtp_rarngd_adj_mat + np.transpose(gtp_rarngd_adj_mat)

	# Visualize 
	fig = plt.figure()

	ax1 = fig.add_subplot(231)
	ax1.imshow(g1_inc_mat,  interpolation='none')

	ax2 = fig.add_subplot(232)
	ax2.imshow(g2_inc_mat, interpolation='none')

	ax3 = fig.add_subplot(233)
	ax3.imshow(gbp_inc_mat, interpolation='none')

	ax4 = fig.add_subplot(234)
	ax4.imshow(g1_inc_mat,  interpolation='none')

	ax5 = fig.add_subplot(235)
	ax5.imshow(np.transpose(g2_inc_mat), interpolation='none')

	ax6 = fig.add_subplot(236)
	ax6.imshow(gbp_other_inc_mat, interpolation='none')

	plt.show()

	exit(-1)


	g1_evals,_ = np.linalg.eigh(g1_adj_mat)
	g1_evals_sorted = np.sort(g1_evals)
	g2_evals,_ = np.linalg.eigh(g2_adj_mat)
	g2_evals_sorted = np.sort(g2_evals)
	gbp_evals,_ = np.linalg.eigh(gbp_adj_mat)
	gbp_evals_sorted = np.sort(gbp_evals)
	gtp_evals,_  = np.linalg.eigh(gtp_adj_mat)
	gtp_evals_sorted = np.sort(gtp_evals)

	# Print top 2 in each
	print("G1  : ", end="")
	print(g1_evals_sorted[-1], g1_evals_sorted[-2])
	print("G2  : ", end="")
	print(g2_evals_sorted[-1], g2_evals_sorted[-2])
	print("BGP : ", end="")
	print(gbp_evals_sorted[-1], gbp_evals_sorted[-2])
	print("GP  : ", end="")
	print(gtp_evals_sorted[-1], gtp_evals_sorted[-2])

	calc_gtp_evals_sorted = np.sort(np.kron(g1_evals_sorted, g2_evals_sorted))

	verbose = False
	error = 0
	for i in range(calc_gtp_evals_sorted.size):
		if verbose:
			print("{:5.2f} {:5.2f}".format(gtp_evals_sorted[i], calc_gtp_evals_sorted[i]))
		diff = gtp_evals_sorted[i] - calc_gtp_evals_sorted[i]
		error += diff*diff

	print(error)


	exit(-1)

	


	"""

	badj1 = pruners.SRMBRepMasker.SRMBRepMasker.get_ramanujan_pattern(rows, cols, d)
	badj2 = pruners.SRMBRepMasker.SRMBRepMasker.get_ramanujan_pattern(rows, cols, d)


	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.imshow(badj1,  interpolation='none')

	ax2 = fig.add_subplot(122)
	ax2.imshow(np.kron(badj1, badj1), interpolation='none')


	plt.show()
	"""