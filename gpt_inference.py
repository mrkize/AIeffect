import torch
from tqdm import gui
from models.GPT_traj import GPT_traj, MyConfig
from models.GPT_multi import multi_gpt
from models.effect_dataset import MotionTrajectoryDataset
import os
import json
from mars.mars_converter import generate_json_from_keyframe_data_v1
import argparse
import copy
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="/ossfs/workspace/aieffects/save_ckp/data_none/final_model", help='模型路径')
    parser.add_argument('--img', type=str, default="/ossfs/workspace/aieffects/test_pic/lcar.png", help='图像路径')
    parser.add_argument('--headact', action='store_true', help='下游分类头')
    parser.add_argument('--prompt', type=str, default="", help='提示词')
    parser.add_argument('--output', type=str, default="json_output.json", help='输出路径')
    parser.add_argument('--guide', type=float, default=2.0, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    
    args = parser.parse_args()
    return args

args = get_parser()

json_path = '/ossfs/workspace/dataset_v1/keyframes'
imgs_dir = '/ossfs/workspace/dataset_v1/images'
prompt_list = ['Effect_Prompt_1.csv', 'Effect_Prompt_2.csv']
prompt_csv_list = [f"/ossfs/workspace/dataset_v1/{csv_name}" for csv_name in prompt_list]
dataset = MotionTrajectoryDataset(json_path, imgs_dir, prompt_csv_list)
dataset.read_input_data("des")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
valid_idx = [0,1,3,4,5,8,9]

config = MyConfig()
config.valid_idx = valid_idx
config.traj_dim = len(valid_idx)
if "headact" in args.ckp:
  config.head_act = True
# config.out_cls = torch.round(dataset.Max[0][0] - dataset.Min[0][0]).to(torch.int).tolist()


if args.model == "multi":
  valid_idx = [0,1]
  config.traj_dim = len(valid_idx)
  model = multi_gpt.from_pretrained(args.ckp, config=config, ignore_mismatched_sizes=True).to(device)
else:
  model = GPT_traj.from_pretrained(args.ckp, config=config, ignore_mismatched_sizes=True).to(device)
model.eval()

img_name_batch = [args.img]
img_batch = [Image.open(img) for img in img_name_batch]
img_batch = [dataset[10]["imgs"]]
if "only_prompt" in args.ckp:
  img_batch = None
prompt_batch = [args.prompt]
# prompt_batch = [dataset[10]["prompts"]]
print(prompt_batch)

# past_traj_batch = torch.stack([dataset[0]["past_traj"][0].unsqueeze(0), dataset[0]["past_traj"][0].unsqueeze(0)], dim=0).to(device)
# past_traj_batch = torch.tensor([0,0,1,1,1,0,1],dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
# past_traj_batch = torch.tensor([[0,0],[1,1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
# past_traj_batch = torch.tensor([[0,0],[1,1]],dtype=torch.float32).unsqueeze(0).to(device)

# past_traj_batch = dataset[0]["past_traj"][0].unsqueeze(0).unsqueeze(0).to(device)

model.eval()
traj_batch = model.generate_traj(img_batch, prompt_batch, total_frames=60, guidance_scale=args.guide)
if traj_batch is None:
  raise Exception("traj_batch is None")
fill_zero = torch.zeros(1, 60, 1).to(device)
fill_ones = torch.ones(1, 60, 1).to(device)
# 将零填充到根据valid_idx = [0,1,3,4,5,8,9]丢掉的轨迹分量中, 被丢弃的是原来10个分量中的2，6，7
if args.model == "multi":
  traj_batch = torch.cat([traj_batch, fill_zero, fill_ones.repeat(1,1,3), fill_zero.repeat(1,1,3), fill_ones], dim=-1)
else:
  traj_batch = torch.cat([traj_batch[:,:,:2], fill_zero, traj_batch[:,:,2:5], fill_zero, fill_zero, traj_batch[:,:,5:]], dim=-1)
# 还原rotation角度
traj_batch = dataset[10]["past_traj"].unsqueeze(0).cuda()
traj_batch[:,:,6:9] *= 180
workspace = "/ossfs/workspace"
# 写入json文件
with open("utils/json_template.json", 'r') as f:
  data = json.load(f)
  # for lyr in data["layers"]:
  #   if lyr["type"] == "sprite":
  #     lyr["img"] = lyr["img"].replace("dataset_v1", "../dataset_v1")
  layer = data["layers"][0]
  data["layers"] = []
  for img, traj in zip(img_name_batch, traj_batch):
    layert = layer
    # 轨迹分量position: 0-2, size: 3-5, rotation: 6-8 opacity: 9
    position = traj[:,:3]
    # print(position)
    size = traj[:,3:6]
    rotation = traj[:,6:9]
    opacity = traj[:,9]
    layert['data']["position"] = position.tolist()
    layert['data']["size"] = size.tolist()
    layert['data']["rotation"] = rotation.tolist()
    layert['data']["opacity"] = opacity.tolist()
    layert["img"] = img
    data["layers"].append(copy.deepcopy(layert))
  mars_json = generate_json_from_keyframe_data_v1(data)
  json_str = json.dumps(mars_json, ensure_ascii=False)
  json_str = json_str.replace('\\"', '"').replace('\\\\', '\\').strip('"')
  with open(args.output, 'w') as jo:
    jo.write(json_str)