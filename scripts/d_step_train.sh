export HF_ENDPOINT=https://hf-mirror.com
work_dir="/home/mkdzir/Code/aieffects"
base_dir="$work_dir/save_ckp/DiT_aug4"

# cd ..
if [ ! -d "$base_dir" ]; then
    mkdir -p "$base_dir"
fi

output="$base_dir/results"
final="$base_dir/final_model"
logs="$base_dir/logs"
datanorm=""

# step1, 仅图像和轨迹训练
# torchrun --nproc_per_node=4 diffusion_trainer.py \
# --cfg \
# --dataset "nodes" \
# --datanorm "$datanorm" \
# --output "$output" \
# --final "$final" \
# --logs "$logs" \
# --i_drop 1 \
# --p_drop 0. \
# --batchsize 2048 \
# --epoch 100 \

# step3 图像和提示词一起训练
# torchrun --nproc_per_node=4 diffusion_trainer.py \
# --checkpoint_path "$final" \
# --dataset "all" \
# --datanorm "$datanorm" \
# --output "$output" \
# --final "$final" \
# --logs "$logs" \

# step4 图像和提示词一起训练
torchrun --nproc_per_node=2 diffusion_trainer.py \
--denoise_model "DiT" \
--cond_type "prompt" \
--PE "learned" \
--ckp "" \
--cfg \
--dataset "des" \
--datanorm "$datanorm" \
--output "$output" \
--final "$final" \
--logsdir "$logs" \
--i_drop 1 \
--p_drop 0.4 \
--batchsize 512 \
--epoch 10000 \
--lr 2e-4 \
--filter "" \
--resume_from_checkpoint \