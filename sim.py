import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
from models.MyConfig import MyConfig
from models.effect_dataset import MotionTrajectoryDataset
from models.DiT import DiT
from models.Diffusion_net import Net
from models.Diffusion import GaussianDiffusionTrainer
import argparse
import os
import torch.nn.functional as F


def rowwise_pearson(x, y, eps=1e-8):
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1)) + eps

    return numerator / denominator                             # (D,)

def dtw_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两条 1D 序列的 DTW 距离
    x, y: shape (T,)
    返回一个标量张量
    """
    n, m = x.size(0), y.size(0)
    dtw = x.new_full((n+1, m+1), float('inf'))
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = torch.abs(x[i-1] - y[j-1])
            prev_min = torch.min(torch.stack([
                dtw[i-1, j],    # 插入
                dtw[i, j-1],    # 删除
                dtw[i-1, j-1],  # 匹配
            ]))
            dtw[i, j] = cost + prev_min

    return dtw[n, m]

def compare_columns_dtw_pearson(
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> dict:
    """
    对形状 (T, D) 的两个张量 X, Y 按列计算相似度
    返回：
      {
        'pearson': torch.Tensor of shape (D,),
        'dtw':     torch.Tensor of shape (D,),
        'pearson_mean': scalar,
        'dtw_mean':     scalar,
      }
    """
    assert X.shape == Y.shape, "X, Y 必须同型 (T, D)"
    T, D = X.shape

    # 1) 批量皮尔逊
    pearson_vals = pearson_corr_matrix(X, Y)  # (D,)

    # 2) 按列 DTW
    dtw_vals = []
    for i in range(D):
        dtw_vals.append(dtw_distance(X[:, i], Y[:, i]))
    dtw_vals = torch.stack(dtw_vals)          # (D,)

    return {
        'pearson':       pearson_vals,
        'dtw':           dtw_vals,
        'pearson_mean':  pearson_vals.mean(),
        'dtw_mean':      dtw_vals.mean(),
    }

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--denoise_model', type=str, default="", help='模型类型')
    parser.add_argument('--dataset', type=str, default='des', help='使用有文本标注的数据训练进行二次训练')
    parser.add_argument('--ckp', type=str, default="", help='checkpoint目录')
    parser.add_argument('--output', type=str, default="./save_ckp/diffusionCT/results", help='模型保存路径')
    parser.add_argument('--final', type=str, default="./save_ckp/diffusionCT/final_model", help='最终模型')
    parser.add_argument('--logsdir', type=str, default="./save_ckp/diffusionCT/logs", help='日志')
    parser.add_argument('--cfg', action='store_true', help='无分类器引导训练')
    parser.add_argument('--headact', action='store_true', help='下有头的选择')
    parser.add_argument('--i_drop', type=float, default=0.0, help='图像丢弃概率')
    parser.add_argument('--p_drop', type=float, default=0.1, help='文本丢弃概率')
    parser.add_argument('--datanorm', type=str, default="", help='数据归一化')
    parser.add_argument('--dataaug', action='store_true', help='数据增强')
    parser.add_argument('--batchsize', type=int, default=512, help='batchsize')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='training lr')
    parser.add_argument('--filter', type=str, default="", help='筛选数据集')
    parser.add_argument('--PE', type=str, default="learned", help='位置编码类型')
    parser.add_argument('--cond_type', type=str, default="prompt", help='控制类型')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='加载检查点')

    args = parser.parse_args()
    return args
args = get_parser()
valid_idx = [0, 1, 3, 4, 5, 8, 9]
# ====== 使用示例 ======
# 假设 X, Y 是你的 60×10 的动画属性张量
json_path = '/home/mkdzir/Pytorch/Dataset_mars/keyframes'
imgs_dir = '/home/mkdzir/Pytorch/Dataset_mars/images'
prompt_list = ['Effect_Prompt_1.csv', 'Effect_Prompt_2.csv']
prompt_csv_list = [f"/home/mkdzir/Pytorch/Dataset_mars/{csv_name}" for csv_name in prompt_list]

dataset = MotionTrajectoryDataset(json_path, imgs_dir, prompt_csv_list, valid_idx)
dataset.read_input_data(args.dataset, norm=args.datanorm, filter=args.filter)
X = dataset.traj_label[0][:,valid_idx].cuda()
Y = torch.load("mars.pt").squeeze(0).cuda()
res = compare_columns_dtw_pearson(X, Y)
print("Per-attribute Pearson:", res['pearson'])
print("Mean Pearson:",       res['pearson_mean'])
print("Per-attribute DTW:",   res['dtw'])
print("Mean DTW:",           res['dtw_mean'])
