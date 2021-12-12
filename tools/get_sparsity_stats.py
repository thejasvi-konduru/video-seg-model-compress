import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("model", help="Path to the model file")
parser.add_argument("--collapse", action="store_true")

args = parser.parse_args()

# Loading model
cp = torch.load(args.model)["state_dict"]

for key in cp:
    tensor = cp[key]
    # Only Convolutional and FC layers
    if len(tensor.shape) >= 2:
        # Convert torch tensor to numpy tensor
        tensor = tensor.detach().cpu().numpy()
        # Overall sparsity
        sp = (1.0 - np.count_nonzero(tensor)/tensor.size)
        # Only allow sparse layers
        #if sp == 0:
        #    continue

        # CONVERTING VALUES IN TENSOR TO 0 or 1
        tensor[tensor != 0] = 1 
        tensor = tensor.astype(int)
        mat = tensor.reshape(tensor.shape[0], tensor.size//tensor.shape[0])

        if len(tensor.shape) == 4:
            filter_sp = (1.0 - np.count_nonzero(np.sum(tensor, axis=(1,2,3)))/tensor.shape[0])
            chan_sp = (1.0 - np.count_nonzero(np.sum(tensor, axis=(0,2,3)))/tensor.shape[1])

            row_sp = (1.0 - np.count_nonzero(np.sum(mat, axis=1))/mat.shape[0])
            col_sp = (1.0 - np.count_nonzero(np.sum(mat, axis=0))/mat.shape[1])
        elif len(tensor.shape) == 2:
            filter_sp = (1.0 - np.count_nonzero(np.sum(mat, axis=1))/mat.shape[0])
            chan_sp   = (1.0 - np.count_nonzero(np.sum(mat, axis=0))/mat.shape[1])

            row_sp = filter_sp
            col_sp = chan_sp
        else:
            exit(-1)

        print("{:38s} {:20s} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}".format(key, str(tensor.shape), sp*100, filter_sp*100,
                                                             chan_sp*100, row_sp*100, col_sp*100), end="")

        print()
        continue
        # Get sparsity at the block level
        block_size = [64,32]
        if mat.shape[0] >= 128:
            block_size = [128,32]

        split_mat = mat.reshape(mat.shape[0]//block_size[0], block_size[0], mat.shape[1]//block_size[1], block_size[1])
        block_mat = np.transpose(split_mat, (0,2,1,3))

        block_sp = (1.0 - np.count_nonzero(np.sum(block_mat, axis=(2,3)))/(block_mat.shape[0] * block_mat.shape[1]))
        print(" | {:3d}x{:3d} - {:5.2f}".format(block_size[0], block_size[1], block_sp*100))

        block_mask = np.clip(np.sum(block_mat, axis=(2,3)), 0,1).astype(int)
        if block_mask.shape[1] == 16:
            #pass
            print(block_mask)

        








