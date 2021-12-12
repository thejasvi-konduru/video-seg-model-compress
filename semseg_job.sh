#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --job-name=semseg_pruner
#SBATCH --output=log_outputs/semseg_drn_d_54/semseg_out_drn_d_54%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

dataset=cityscapes
arch=drn_d_54
epochs=500
lr=0.01
batch_size=4
sp=0.75
input_size='512X512'
pconfig_path=sparse_experiments/srmbrep_cityscapes_drn_d_54/sparse_srmbrep_cityscapes_drn_d_54_srmbrep_512X512_-1x-1_-1x-1_2x2_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/config.json
# # base_model=base_models/resnet50-19c8e357.pth
# ename=${dataset}_${arch}_${epochs}

exp_dir=sparse_experiments/srmbrep_cityscapes_drn_d_54/sparse_srmbrep_cityscapes_drn_d_54_srmbrep_512X512_-1x-1_-1x-1_2x2_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive
# launch_dir=/home/${USER}/M2020/S2021/rmbsnn/classifier/
# pruners_lib_dir=/home/${USER}/M2020/S2021/rmbsnn/
scratch_dir=/ssd_scratch/cvit/furqan.shaik

# # Making sure base models are there
# if [ $USER !=  "dharmateja" ]; then
# 	rsync -a dharmateja@ada:/home/dharmateja/work/rmbsnn/classifier/base_models .
# fiZazaz

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
# python semantic_seg.py -d=${scratch_dir}/cityscapes/cityscapes train -c 19 --epochs=${epochs}  --lr=${lr} -b=${batch_size} --exp_dir=sparse_experiments/srmbrep_cityscapes_drn_d_22/sparse_srmbrep_cityscapes_drn_d_22_srmbrep_512X512_-1x-1_-1x-1_2x2_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive --resume sparse_experiments/srmbrep_cityscapes_drn_d_22/sparse_srmbrep_cityscapes_drn_d_22_srmbrep_512X512_-1x-1_-1x-1_2x2_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive_lr0.01/checkpoint_best.pth.tar | tee sparse_experiments/srmbrep_cityscapes_drn_d_22/sparse_srmbrep_cityscapes_drn_d_22_srmbrep_512X512_-1x-1_-1x-1_2x2_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/log_resume.txt

python semantic_seg.py train \
		 -d=${scratch_dir}/cityscapes/cityscapes \
         --dataset=${dataset} \
         --arch=${arch} \
         --exp_dir=${exp_dir} \
         --mc_pruning \
         --pr_config_path=${pconfig_path} \
         --pr-static \
         --lr=${lr} --sparsity=${sp} \
		 --input_size=${input_size} \
         --epochs=${epochs} -b=${batch_size} | tee ${exp_dir}/log_500.txt