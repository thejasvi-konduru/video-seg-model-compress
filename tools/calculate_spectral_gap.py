import os
import numpy as np
import math

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def get_adjacency_from_incidence(inc_mat):
    # Constructing adjacency matrix
    adj_mat = np.zeros((inc_mat.shape[0]+inc_mat.shape[1], inc_mat.shape[0]+inc_mat.shape[1]), dtype=int)
    adj_mat[:inc_mat.shape[0], inc_mat.shape[0]:] = inc_mat
    adj_mat[inc_mat.shape[0]:, :inc_mat.shape[0]] = np.transpose(inc_mat)

    return adj_mat

def extract_spectral_gap(exp_dir, verbose=False):
    import torch
    model = torch.load(os.path.join(exp_dir,"model_best.pth.tar"))

    sgaps = []
    sgap_norms = []
    sgap_ram_norms = []
    for pname in model["state_dict"]:
        x = model["state_dict"][pname].cpu().numpy()
        sp = 1.0 - np.count_nonzero(x)/x.size
        if len(x.shape) >= 4 and sp > 0:
            x_mat  = np.sum(x, axis=(2,3))
            x_mask = (x_mat != 0).astype(int)

            # Left and right degrees
            lv_degrees = np.sum(x_mask, axis=1)
            rv_degrees = np.sum(x_mask, axis=0)
            lv_degree = lv_degrees[0]
            rv_degree = rv_degrees[0]

            assert (lv_degrees == lv_degree).all(), "Graph is irregular from left"
            assert (rv_degrees == rv_degree).all(), "Graph is irregular from right"
            
            adj_mat = get_adjacency_from_incidence(x_mask)
            evals,evecs = np.linalg.eigh(adj_mat)
            evals_sorted = np.sort(evals)

            # Because it's a bipartite graph, 
            # spectral gap is difference of largest two positive values
            lambda_1 = evals_sorted[-1]
            lambda_2 = evals_sorted[-2]
            sgap = lambda_1 - lambda_2 # Spectral gap
            ram_bound = math.sqrt(lv_degree-1) + math.sqrt(rv_degree-1) #Bound
            sgap_ram_norm = sgap/(lambda_1-ram_bound)

            sgaps.append(sgap)
            sgap_ram_norms.append(sgap_ram_norm)

            if verbose:
                print(pname)
                print("Top-3 Eigen values ",evals_sorted[-3:])
                print(sgap, ram_bound, sgap_ram_norm)
                print()

    avg_sgap = sum(sgaps)/len(sgaps)
    avg_sgap_ram_norm = sum(sgap_ram_norms)/len(sgap_ram_norms)

    if verbose:
        print("Average sgap : ", avg_sgap_ram_norm)

    #return avg_sgap
    return avg_sgap_ram_norm

if __name__ == "__main__":
    import sys
    extract_spectral_gap(sys.argv[1], True)
