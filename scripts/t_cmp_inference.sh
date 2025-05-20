export HF_ENDPOINT=https://hf-mirror.com
work_dir="/home/mkdzir/Code/aieffects"
cd "$work_dir"

BASE_DIR="$work_dir/generate_mars"
# directories=($(ls -d "$BASE_DIR"/*/ 2>/dev/null | sort))
# total_directories=${#directories[@]}
# if [ "$total_directories" -ge 10 ]; then
#   dir_to_remove="${directories[0]}"
#   rm -rf "$dir_to_remove"
#   echo "Deleted directory: $dir_to_remove"
# fi

# 使用日期时间戳生成唯一目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NEW_DIR="$BASE_DIR/$TIMESTAMP"
# mkdir -p "$NEW_DIR"

model=""
model_type="DiT"
final="$work_dir/save_ckp/$model_type/final_model"
# final="$work_dir/checkpoint/DiT"
img1="$work_dir/test_pic/leaf.png"
prompt1="上下跳动"
prompt2="向外迸发"
prompt3="左右摇摆"
prompt4="快速伸缩跳动"
guide=0.5

python diffusion_inference.py \
--img "$img1" \
--output "$NEW_DIR-$model_type-3-$prompt3.json" \
--prompt "$prompt3" \
--ckp "$final" \
--guide $guide \
--model "$model" \
--model_type "$model_type" \
--seed 110


python diffusion_inference.py \
--img "$img1" \
--output "$NEW_DIR-$model_type-3-$prompt3.json" \
--prompt "$prompt3" \
--ckp "$final" \
--guide $guide \
--model "$model" \
--model_type "$model_type" \
--seed 10011


python diffusion_inference.py \
--img "$img1" \
--output "$NEW_DIR-$model_type-3-$prompt3.json" \
--prompt "$prompt3" \
--ckp "$final" \
--guide $guide \
--model "$model" \
--model_type "$model_type" \
--seed 10012

python diffusion_inference.py \
--img "$img1" \
--output "$NEW_DIR-$model_type-3-$prompt3.json" \
--prompt "$prompt3" \
--ckp "$final" \
--guide $guide \
--model "$model" \
--model_type "$model_type" \
--seed 10013

python diffusion_inference.py \
--img "$img1" \
--output "$NEW_DIR-$model_type-3-$prompt3.json" \
--prompt "$prompt3" \
--ckp "$final" \
--guide $guide \
--model "$model" \
--model_type "$model_type" \
--seed 10014

# python inference.py --img "/ossfs/workspace/aieffects/test_pic/10jifen.png" --output "out2.json" --prompt "小车在跳动"
