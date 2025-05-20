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
from datetime import datetime
import matplotlib.pyplot as plt
import os

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="checkpoint/DiT", help='模型路径')
    parser.add_argument('--img', type=str, default="test_pic/lcar.png", help='图像路径')
    parser.add_argument('--headact', action='store_true', help='下游分类头')
    parser.add_argument('--prompt', type=str, default="伸缩跳动", nargs='+', help='提示词')
    parser.add_argument('--output', type=str, default="json_output.json", help='输出路径')
    parser.add_argument('--guide', type=float, default=0.5, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    parser.add_argument('--model_type', type=str, default="", help='')
    parser.add_argument('--seed', type=int, default=1001, help='')
    args = parser.parse_args()
    return args


def plot_mars(pred_mars, prompt_batch, save_dir='newquxiantu'):
    labels = ["Position x", "Position y", "Position z",
              "Size x", "Size y", "Size z",
              "Rotation x", "Rotation y", "Rotation z",
              "Opacity"]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")+"_"+str(args.seed)
    torch.save(pred_mars, os.path.join(save_dir, f"pred_mars_{timestamp}.pt"))

    for sample_id in range(pred_mars.shape[0]):
        plt.figure(figsize=(10, 6))
        for i in range(pred_mars[sample_id].size(0)):
            plt.plot(pred_mars[sample_id][i].cpu().numpy(), label=labels[i])

        plt.title('The sequences of various Transform')
        plt.xlabel('Frames')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()  # 自动紧凑布局，防止标题或标签被遮挡
        img_name = f"{prompt_batch[sample_id]}_{timestamp}.png"
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')
        plt.close()  # 避免内存泄漏和多图叠加


args = get_parser()
set_seed(args.seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
valid_idx = [0,1,3,4,5,8,9]


config = MyConfig.from_pretrained(args.ckp)
# if "checkpoint" in args.ckp:
#     d_trainer = GaussianDiffusionTrainer.from_pretrained(args.ckp,
#                                                          config=config)
#     model = d_trainer.model
# else:
#     model = DiT.from_pretrained(args.ckp, config=config)

model = DiT.from_pretrained(args.ckp, config=config)

sampler = GaussianDiffusionSampler(model=model,
                                config=config,
                                w = args.guide,
                                ).to(device)
sampler.eval()


prompt_batch = args.prompt
B= len(prompt_batch)
img_name_batch = [args.img]*B
img_batch = [Image.open(img) for img in img_name_batch]
img_batch = None
print(prompt_batch)


xt_noise = torch.randn(size=[B, 60, len(valid_idx)], device=device)
mars = sampler(xt_noise, img_batch, prompt_batch)
# torch.save(mars, "mars.pt")
mars = curve_smoothing(mars)
# torch.save(mars, "mars_smoothing.pt")
pred_mars = torch.zeros(B, 60, 10).to(device)
pred_mars[:,:,config.valid_idx] = mars

# 还原rotation角度
# pred_mars = MotionTrajectoryDataset.ret_relative_position_rot(pred_mars)


plot_mars(pred_mars.permute(0, 2, 1), prompt_batch)


# 写入json文件
# with open("utils/json_template.json", 'r') as f:
#     data = json.load(f)
#     layer = copy.deepcopy(data["layers"][0])
#     data["layers"] = []
#     for img, traj in zip(img_name_batch, pred_mars):
#         layert = copy.deepcopy(layer)
#         # 轨迹分量position: 0-2, size: 3-5, rotation: 6-8 opacity: 9
#         position = traj[:,:3]
#         # print(position)
#         size = traj[:,3:6]
#         rotation = traj[:,6:9]
#         opacity = traj[:,9]
#         layert['data']["position"] = position.tolist()
#         layert['data']["size"] = size.tolist()
#         layert['data']["rotation"] = rotation.tolist()
#         layert['data']["opacity"] = opacity.tolist()
#         layert["img"] = img
#         data["layers"].append(layert)
#     mars_json = generate_json_from_keyframe_data_v1(data)
#     json_name = args.model_type + "-".join([str(i) for i in args.prompt])
#     json_str = json.dumps(mars_json, ensure_ascii=False)
#     json_str = json_str.replace("A_NuD6S5JtDtkAAAAAAAAAAAAADlB4AQ", json_name)
#     json_str = json_str.replace('\\"', '"').replace('\\\\', '\\').strip('"')
# with open(args.output, 'w') as jo:
#     jo.write(json_str)
    # sys.exit("退出程序")
