import torch
from tqdm import gui
from models.Diffusion_net import Net
from models.MyConfig import MyConfig
from models.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
import os
import json
from mars.mars_converter import generate_json_from_keyframe_data_v1
import argparse
import copy
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="/ossfs/workspace/aieffects/save_ckp/diffusion/final_model", help='模型路径')
    parser.add_argument('--img', type=str, default="/ossfs/workspace/aieffects/test_pic/lcar.png", help='图像路径')
    parser.add_argument('--headact', action='store_true', help='下游分类头')
    parser.add_argument('--prompt', type=str, default="", nargs='+', help='提示词')
    parser.add_argument('--output', type=str, default="json_output.json", help='输出路径')
    parser.add_argument('--guide', type=float, default=2.0, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    
    args = parser.parse_args()
    return args

args = get_parser()

# 写入json文件
with open("utils/json_template.json", 'r') as f:
  data = json.load(f)
  mars_json = generate_json_from_keyframe_data_v1(data)
  json_str = json.dumps(mars_json, ensure_ascii=False)
  json_str = json_str.replace('\\"', '"').replace('\\\\', '\\').strip('"')
  with open("utils/output.json", 'w') as jo:
    jo.write(json_str)