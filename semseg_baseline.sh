#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --job-name=semseg_pruner
#SBATCH --output=log_outputs/semseg_baseline_drn_d_54%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

# source activate py_3.7
module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

dataset=cityscapes
arch=drn_d_54
epochs=500
lr=0.01
batch_size=4
crop_size='512X512'
# sparsity = 0.75
# pconfig_path=sparse_experiments/config.json
# base_model=base_models/resnet50-19c8e357.pth
ename=${dataset}_${arch}_${epochs}_${crop_size}

#exp_dir=sparse_experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive
# launch_dir=/home/${USER}/M2020/S2021/rmbsnn/classifier/
# pruners_lib_dir=/home/${USER}/M2020/S2021/rmbsnn/
scratch_dir=/ssd_scratch/cvit/furqan.shaik

# # Making sure base models are there
# if [ $USER !=  "dharmateja" ]; then
# 	rsync -a dharmateja@ada:/home/dharmateja/work/rmbsnn/classifier/base_models .
# fi

# Creating directory
# mkdir -p $exp_dir

# Move config to experiment directory
# cp ${pconfig_path} ${exp_dir}/config.json
# rm -rf ${scratch_dir}
# Creating scratch directory
mkdir -p ${scratch_dir}

# Set up cityscapes dataset if it does not exist
if [ ! -f "${scratch_dir}/cityscapes/cityscapes.tar" ]; then
	# Loading data from dataset to scratch
	rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -a furqan.shaik@ada:/share3/furqan.shaik/S2021/drn/datasets/cityscapes/* ${scratch_dir}/cityscapes/cityscapes/
	cd ${scratch_dir}/cityscapes/
	tar -xvf cityscapes.tar --strip 1 
	cd ${scratch_dir}/cityscapes/cityscapes/
	python3 prepare_data.py gtFine/

	bash create_lists.sh
	echo ls
	# Copying necessary scripts
	#cp ${launch_dir}/imagenet-scripts/* ${scratch_dir}/imagenet/

fi

# Running the script
#export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
cd ~/M2020/Summer_2021/rmb

python semseg_baseline.py \
		-d=${scratch_dir}/cityscapes/cityscapes train \
		-c 19 --epochs=${epochs} \
		--crop_size=${crop_size} --lr=${lr} \
		-b=${batch_size} \
		--exp_dir=experiments/${ename} | tee experiments/semseg_baseline_drn_d_54_512X512_ep500.txt
