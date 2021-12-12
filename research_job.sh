#!/bin/bash
#SBATCH -A furqan.shaik
#SBATCH -p long

#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=16-00:00:00
#SBATCH --mail-type=END
#SBATCH --job-name=imagenet_pruner
#SBATCH --output=log_outputs/imgnet_out%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

source activate py_3.7
module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

# pconfig_path=sparse_experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/config.json
# base_model=sparse_experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/model_best.pth.tar
# # ename=rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive

# exp_dir=/S2021/rmb/sparse_experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive
launch_dir=/home/${USER}/M2020/S2021/rmb/
pruners_lib_dir=/home/${USER}/M2020/S2021/rmb/
scratch_dir=/scratch/${USER}/

echo ${USER}
# # Making sure base models are there
# if [ $USER !=  "dharmateja" ]; then
# 	rsync -a dharmateja@ada:/home/dharmateja/work/rmbsnn/classifier/base_models .
# fi

# Creating directory

# Move config to experiment directory
# cp ${pconfig_path} ${exp_dir}/config.json

# Creating scratch directory
rm -rf /scratch/furqan.shaik
mkdir ${scratch_dir}
# mkdir /scratch/
# mkdir -p ${scratch_dir}

# Set up imagenet dataset if it does not exist
if [ ! -f "/scratch/furqan.shaik/Imagenet2012/Imagenet-orig.tar" ]; then
	# Loading data from dataset to scratch
rsync -a furqan.shaik@ada:/share1/dataset/Imagenet2012/  ${scratch_dir}/Imagenet2012

# Copying necessary scripts
cp ${launch_dir}/imagenet-scripts/* ${scratch_dir}/Imagenet2012

# Extrat main, train, and val tar files
cd ${scratch_dir}/Imagenet2012
tar -xvf Imagenet-orig.tar --strip 1
mkdir train
tar -C train -xvf ./Imagenet-orig/ILSVRC2012_img_train.tar
mkdir val
tar -C val -xvf ./Imagenet-orig/ILSVRC2012_img_val.tar
mv prep_train.py train/
mv valprep.sh val/

# Run scripts to get into torch format
cd train
python prep_train.py

cd ../val
sh valprep.sh

# Go back to base directory
cd ${launch_dir}
fi

# Running the script
export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'


python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
          --dataset imagenet \
          --arch mobilenet_v2 \
        #   --resume experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/checkpoint.pth.tar \
          --exp-dir new_experiments/experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive  \
          --mc-pruning \
          --pr-config-path new_experiments/experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
          --pr-static \
          --lr 0.01 \
          --epochs 100 \
          --batch-size 128 | tee new_experiments/experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch mobilenet_v2 \
#          --exp-dir experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_mobilenet_v2/model_best.pth.tar \
#          --pr-config-path experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 128 | tee experiments/imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/log.txt

#python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#         --dataset imagenet \
#         --arch resnet18 \
#         --resume experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/checkpoint.pth.tar \
#         --exp-dir experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive  \
#         --mc-pruning \
#         --pr-config-path experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#         --pr-static \
#         --lr 0.01 \
#         --epochs 100 \
#         --batch-size 128 | tee experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch resnet18 \
#          --exp-dir experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_resnet18/model_best.pth.tar \
#          --pr-config-path experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 128 | tee experiments/imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch mobilenet_v2 \
#          --exp-dir experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_mobilenet_v2/model_best.pth.tar \
#          --pr-config-path experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 128 | tee experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch resnet18 \
#          --exp-dir experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_resnet18/model_best.pth.tar \
#          --pr-config-path experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 256 | tee experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_75.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch mobilenet_v2 \
#          --exp-dir experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_mobilenet_v2/model_best.pth.tar \
#          --pr-config-path experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.001 \
#          --epochs 100 \
#          --batch-size 128 | tee experiments/rbgp_imagenet_mobilenet_v2/sparse_imagenet_mobilenet_v2_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/log.txt
 
#python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch resnet18 \
#          --exp-dir experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_93.75-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_resnet18/model_best.pth.tar \
#          --pr-config-path experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_93.75-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 256 | tee experiments/rbgp_imagenet_resnet18/sparse_imagenet_resnet18_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_93.75-RAMANUJAN_50.00_collapse_repetitive/log.txt

# python rmbsnn_main.py /scratch/furqan.shaik/Imagenet2012 \
#          --dataset imagenet \
#          --arch resnet50 \
#          --exp-dir experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_imagenet_resnet50/model_best.pth.tar \
#          --pr-config-path experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 100 \
#          --batch-size 128 | tee experiments/rbgp_imagenet_resnet50/sparse_imagenet_resnet50_srmbrep_-1x-1_-1x-1_1x1_0.00-RAMANUJAN_75.00-RAMANUJAN_50.00_collapse_repetitive/log.txt
