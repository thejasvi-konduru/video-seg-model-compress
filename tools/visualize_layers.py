import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch


def get_block_mask(matrix, bh, bw):
    matrix = matrix.reshape(matrix.shape[0]//bh, bh, matrix.shape[1]//bw, bw)
    matrix = matrix.transpose(0,2,1,3)
    block_mask = np.sum(matrix, axis=(2,3))
    block_mask[block_mask != 0] = 1
    return block_mask


parser = argparse.ArgumentParser()

parser.add_argument("model", help="Path to model file")
parser.add_argument("--ofm", type=int, default=-1, help="Number of ofm to show")
parser.add_argument("--ifm", type=int, default=-1, help="Number of columns to show")
parser.add_argument("--bh", type=int, default=1, help="Height of the block")
parser.add_argument("--bw", type=int, default=1, help="Width of the block")
parser.add_argument("--collapse", action="store_true")

args = parser.parse_args()

# Load checkpoint
model = torch.load(args.model)["state_dict"]

for key in model:
    # Conv and FC has tensor sizes >=2 for weights
    if len(model[key].shape) >= 2:
        # Convert to numpy and mask for convenience
        tensor = model[key].detach().cpu().numpy()
        tensor[tensor != 0] = 1
        # Handy to have matricized form
        matrix = tensor.reshape(tensor.shape[0], tensor.size//tensor.shape[0])

        # Do not show dense layers
        if np.count_nonzero(tensor) == tensor.size:
            print("Layer {} is dense.Hense no show".format(key))
            continue

        print("Showing sparse layer {} with shape {}".format(key, str(tensor.shape)))

        # Getting the sub matrix
        kernel_size = tensor.size//(tensor.shape[0]*tensor.shape[1])
        rows = matrix.shape[0] if args.ofm == -1 else args.ofm
        cols = matrix.shape[1] if args.ifm == -1 else args.ifm * kernel_size
        sub_mat = matrix[:rows, :cols]


        # Getting the block matrix
        bh = args.bh
        bw = args.bw if args.collapse else args.bw*kernel_size
        block_mat = get_block_mask(sub_mat, bh, bw)

        print(np.sum(block_mat, axis=0))
        print(np.sum(block_mat, axis=1))
        #continue
        plt.imshow(block_mat, interpolation='none')
        plt.show()
