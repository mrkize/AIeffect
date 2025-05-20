import torch
from models.utils import set_seed,curve_smoothing
from models.DiT import DiT
from models.MyConfig import MyConfig
from models.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
import sys
import json
from mars.mars_converter import generate_json_from_keyframe_data_v1
import argparse
import copy
from PIL import Image
import matplotlib.pyplot as plt
import math

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="checkpoint/DiT", help='模型路径')
    parser.add_argument('--img', type=str, default="test_pic/lcar.png", help='图像路径')
    parser.add_argument('--headact', action='store_true', help='下游分类头')
    parser.add_argument('--prompt', type=str, default="伸缩跳动", nargs='+', help='提示词')
    parser.add_argument('--output', type=str, default="json_output2.json", help='输出路径')
    parser.add_argument('--guide', type=float, default=0.5, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    parser.add_argument('--model_type', type=str, default="", help='')

    args = parser.parse_args()
    return args



args = get_parser()

# set_seed(1001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
valid_idx = [0,1,3,4,5,8,9]


# config = MyConfig.from_pretrained(args.ckp)
# if "checkpoint" in args.ckp:
#     d_trainer = GaussianDiffusionTrainer.from_pretrained(args.ckp,
#                                                          config=config)
#     model = d_trainer.model
# else:
#     model = DiT.from_pretrained(args.ckp, config=config)




prompt_batch = args.prompt
B= len(prompt_batch)

# x_f = torch.linspace(3.3, -3.3, steps=60)
# out1 = torch.tensor([1, 1.22, 0, 1, 1, 1, 0, 0, 0,
#                      1]).unsqueeze(0).repeat(60, 1)
# out1[:, 0] = x_f[:60]
# out2 = torch.tensor([0, 0, 0, 8, 3, 1, 0, 0, 0, 1]).unsqueeze(0).repeat(60, 1)
# pred_mars = torch.stack([out2, out1], dim=0)
# img_name_batch = ["test_pic/road.png", "test_pic/lcar.png"]




x_f = torch.linspace(-2.67, 2.67, steps=60)
out1 = torch.tensor([1, 2.68, 0, 2, 2, 2, 0, 0, 0,
                     1]).unsqueeze(0).repeat(60, 1)
out1[:, 0] = x_f[:60]
n_points = 60       # 生成60个点
period = 30         # 周期为30
min_val = 0.8       # 最小值
max_val = 1.2       # 最大值
amplitude = (max_val - min_val) / 2    # 0.2
offset = (max_val + min_val) / 2       # 1.0
x = torch.linspace(0, 2 * math.pi * (n_points / period), n_points)
sin_values = offset + amplitude * torch.sin(x)

out2 = torch.tensor([-1.4, 2.68, 0, 2, 2, 2, 0, 0, 0, 1]).unsqueeze(0).repeat(60, 1)
out2[:, 3] = sin_values[:60]
out2[:, 4] = sin_values[:60]

out3 = torch.tensor([2.19, -2.12, 0, 2, 2, 2, 0, 0, 0, 1]).unsqueeze(0).repeat(60, 1)
out4 = torch.tensor([-1.66, -2.12, 0, 2, 2, 2, 0, 0, 0, 1]).unsqueeze(0).repeat(60, 1)
pred_mars = torch.stack([out1, out2, out3, out4], dim=0)
img_name_batch = ["test_pic/cloud.png", "test_pic/sun.png", "test_pic/tree.png", "test_pic/tree.png"]


# 写入json文件
with open("utils/json_template.json", 'r') as f:
    data = json.load(f)
    layer = copy.deepcopy(data["layers"][0])
    data["layers"] = []
    for img, traj in zip(img_name_batch, pred_mars):
        layert = copy.deepcopy(layer)
        # 轨迹分量position: 0-2, size: 3-5, rotation: 6-8 opacity: 9
        position = traj[:, :3]
        # print(position)
        size = traj[:, 3:6]
        rotation = traj[:, 6:9]
        opacity = traj[:, 9]
        layert['data']["position"] = position.tolist()
        layert['data']["size"] = size.tolist()
        layert['data']["rotation"] = rotation.tolist()
        layert['data']["opacity"] = opacity.tolist()
        # print(opacity.tolist())
        layert["img"] = img
        data["layers"].append(layert)

    mars_json = generate_json_from_keyframe_data_v1(data)
    json_name = args.model_type + "-".join([str(i) for i in args.prompt])
    json_str = json.dumps(mars_json, ensure_ascii=False)
    json_str = json_str.replace("A_NuD6S5JtDtkAAAAAAAAAAAAADlB4AQ", json_name)
    json_str = json_str.replace('\\"', '"').replace('\\\\', '\\').strip('"')
with open(args.output, 'w') as jo:
    jo.write(json_str)
with open("keyfram2.json", 'w') as jo:
    jo.write(json.dumps(data, ensure_ascii=False))
    # sys.exit("退出程序")
