#!/bin/bash
#SBATCH -J ICL_uncertainty
#SBATCH -A huaiyang_zhong
#SBATCH -p dgx_normal_q
#SBATCH -N1 --ntasks-per-node=4 --cpus-per-task=4 --gres=gpu:4
#SBATCH --time=6:00:00




export PYTHONPATH="${PYTHONPATH}:/home/hzhong/TurquoiseKitty/ICL/ICL_uncertainty/src"

batch_DIR=$PWD

OMP_NUM_THREADS=1 python -W "ignore" -m torch.distributed.launch \
    --nproc_per_node=4  --master_port 29513 /home/hzhong/TurquoiseKitty/ICL/ICL_uncertainty/official_exp/experiment_8_varyingSigma/official_8_varySig.py \
    --quinine_config_path "${batch_DIR}/official_8_varySig.yaml"

