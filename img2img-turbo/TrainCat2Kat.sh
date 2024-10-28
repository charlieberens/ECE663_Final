#!/usr/bin/env bash

#SBATCH --job-name=RunCatsTraining
#SBATCH --mem=20gb
#SBATCH --time=48:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/RunCatsTraining_%j.out   # Where to save the log

# The following will actually be run.
ls -a
eval "$(conda shell.bash hook)" 
conda init
sleep 10
conda activate img2img-turbo
cd img2img-turbo
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/cat2kat" \
    --dataset_folder "data/cat2kat" \
    --train_img_prep "no_resize" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size=2 --gradient_accumulation_steps=2 \
    --report_to "wandb" --tracker_project_name "cat2kat_1" \
    --enable_xformers_memory_efficient_attention --validation_steps 1000 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1