#!/bin/sh
#BSUB -J backbone
#BSUB -o backbone%J.out
#BSUB -e backbone%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 2:00
#BSUB -N
# end of BSUB options

cd /zhome/2c/b/146593/Desktop/ADLCV/project/A_DLCV_project/src

module load python3/3.11.3

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
source /zhome/2c/b/146593/Desktop/ADLCV/project/env1/bin/activate

python segm/train.py