work_dir="/ossfs/workspace/aieffects"
cd "$work_dir"

output = "./save_ckp/data_max_min/results"
final = "./save_ckp/data_max_min/final_model"
logs = "./save_ckp/data_max_min/logs"

# step1, 仅图像和轨迹训练
torchrun --nproc_per_node=4 trainer.py \
--cfg \
--dataset "nodes" \
--i_drop 0.1 \
--p_drop 0 \
--output "$output" \
--final "$final" \
--logs "$logs" \

# step2 仅提示词和轨迹训练
torchrun --nproc_per_node=4 trainer.py \
--cfg \
--checkpoint_path "$final" \
--dataset "des" \
--i_drop 1 \
--p_drop 0.1 \
--output "$output" \
--final "$final" \
--logs "$logs" \

# step3 图像和提示词一起训练
torchrun --nproc_per_node=4 trainer.py \
--cfg \
--checkpoint_path $final \
--dataset "all" \
--i_drop 0.1 \
--p_drop 0.1 \
--output $output \
--final $final \
--logs $logs \

# step4 图像和提示词一起训练
torchrun --nproc_per_node=4 trainer.py \
--cfg \
--checkpoint_path $final \
--dataset "des" \
--i_drop 0.1 \
--p_drop 0.1 \
--output $output \
--final $final \
--logs $logs \