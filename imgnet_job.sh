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

# source activate open_mmlab
module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

exp_dir=/home/${USER}/M2020/Summer_2021/rmb/experiments/
launch_dir=/home/${USER}/M2020/Summer_2021/rmb/
pruners_lib_dir=/home/${USER}/M2020/Summer_2021/rmb/
scratch_dir=/ssd_scratch/cvit/${USER}/

echo ${USER}
# # Making sure base models are there
# if [ $USER !=  "dharmateja" ]; then
# 	rsync -a dharmateja@ada:/home/dharmateja/work/rmbsnn/classifier/base_models .
# fi

# Creating directory

# Move config to experiment directory
# cp ${pconfig_path} ${exp_dir}/config.json

# Creating scratch directory
rm -rf /ssd_scratch/cvit/furqan.shaik
mkdir ${scratch_dir}
# mkdir /scratch/
# mkdir -p ${scratch_dir}

# Set up imagenet dataset if it does not exist
if [ ! -f "${scratch_dir}/Imagenet2012/Imagenet-orig.tar" ]; then
	# Loading data from dataset to scratch
rsync -aP furqan.shaik@ada:/share1/dataset/Imagenet2012/  ${scratch_dir}/Imagenet2012

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


python imagenet_train.py /ssd_scratch/cvit/furqan.shaik/Imagenet2012 \
          --dataset imagenet \
          --arch resnet50 \
          --exp_dir ${exp_dir} \
          --lr 0.01 \
          --epochs 200 \
          --batch-size 128
