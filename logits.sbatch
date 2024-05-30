#!/bin/bash 

#SBATCH -q ebrainccs 
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=40:00:00
#SBATCH --mem=150GB
#SBATCH -o may29_job_logits_saving.%J.out
#SBATCH -e may29_job_logits_saving.%J.err
#SBATCH --job-name=logits_saving


# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

####LUKELO SBATCH#####
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"


torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final \
    --data_path data/instruct/filtered_json_file.json \
    --image_folder /scratch/ltl2113/LLaVA-Med/data/Allimages \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /scratch/ltl2113/LLaVA-Med/checkpoints/LLAVA_MED_SG2_with_TUNE_Entire_model_April_27th \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
    --run_name April27th_LLAVA_MED_SG2_with_TUNE_entire_model_ML_ADAPTER


####AYA:###
# Set environment variables
# export TRANSFORMERS_CACHE="/scratch/ae2195/huggingface_cache"
export LD_LIBRARY_PATH=/home/ae2195/miniconda3/envs/llava-med/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="/scratch/ae2195/huggingface_cache"

# Assume images.json is in the current working directory or provide the full path
#JSON_FILE_PATH="./stage2_train/images.json"

#Stage 2: Medical Visual Instruct Tuning
torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /scratch/ae2195/LLaVA-Med/checkpoints/llava-med-7b-pretrain \
    --data_path /scratch/ae2195/LLaVA-Med/data/instruct/llava_med_instruct_60k_inline_mention.json \
    --image_folder /scratch/ae2195/LLaVA-Med/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints_for_logits_compute/stage2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb

