#!/usr/bin/env bash

#SBATCH --job-name=TrainLarge
#SBATCH --mem=80gb
#SBATCH --time=120:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/TrainLarge_%j.out   # Where to save the log

# The following will actually be run.
sleep 10
ls -a
export PATH="/home/users/cjb131/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)" 
conda init
sleep 10
conda activate img2img-turbo
cd img2img-turbo
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output2/cyclegan_turbo/TrainLarge" \
    --dataset_folder "data2/cyclegan" \
    --train_img_prep "no_resize" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size=2 --gradient_accumulation_steps=2 \
    --report_to "wandb" --tracker_project_name "TrainLarge" \
    --enable_xformers_memory_efficient_attention --validation_steps 1000 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 \
    --lambda_additional_disc 0.0 --lambda_l2 1.0