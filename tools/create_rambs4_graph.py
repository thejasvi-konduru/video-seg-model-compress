import sys
import numpy as np

import pruners.SRMBRepMasker


import os
import numpy as np
import math

def get_sparsity(tensor):
	sp =  1.0 - np.count_nonzero(tensor)/tensor.size
	return sp

def calculate_spectral_gap(inc_mat):
	adj_mat = np.zeros((inc_mat.shape[0]+inc_mat.shape[1], inc_mat.shape[0]+inc_mat.shape[1]), dtype=int)
	adj_mat[:inc_mat.shape[0], inc_mat.shape[0]:] = inc_mat
	adj_mat[inc_mat.shape[0]:, :inc_mat.shape[0]] = np.transpose(inc_mat)

	evals,evecs = np.linalg.eigh(adj_mat)
	evals_sorted = np.sort(evals)
	d = evals_sorted[-1]
	ram_bound = 2*math.sqrt(d-1)
	sgap = d - evals_sorted[-2]
	sgap_norm = sgap/(d-ram_bound)

	print(evals_sorted[-1], evals_sorted[-2], sgap, sgap_norm)

	return sgap_norm


gh,gw = 256,256
obh,obw = 256,256
jrh,jrw = 1,1
jbh,jbw = 8,8

gih,giw = obh//(jrh*jbh), obw//(jrw*jbw)
goh,gow = gh//obh, gw//obw
opat, ipat = "RAMANUJAN","RAMANUJAN"
osp, isp = 0, 0.9375

assert(goh*jrh*gih*jbh == gh)
assert(gow*jrw*giw*jbw == gw)

# Default flags
is_repetitive = True
collapse_tensor = False


config =  pruners.SRMBRepMasker.SRMBRepMaskerConfig(obh, obw, 
													obh//jrh, obw//jrw, 
													jbh, jbw, 
													osp, opat, 
													isp, ipat, 
													is_repetitive, collapse_tensor,
													cross_prob=0.5, is_symmetric=False)

tensor = np.zeros((gh,gw))
mask = pruners.SRMBRepMasker.SRMBRepMasker.construct_mask(tensor, config)
get_sparsity(mask)

sgap_norm = calculate_spectral_gap(mask)

print("Go : {}x{}".format(goh, gow))
print("Jr : {}x{}".format(jrh, jrw))
print("Gi : {}x{}".format(gih, giw))
print("Jb : {}x{}".format(jbh, jbw))

print("{:.2f} ".format(sgap_norm), end=",")


