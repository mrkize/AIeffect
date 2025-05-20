import torch
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



def pearson_corr_matrix(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xm = X.mean(dim=0)
    ym = Y.mean(dim=0)
    Xc = X - xm
    Yc = Y - ym
    cov = (Xc * Yc).mean(dim=0)
    std_x = Xc.std(dim=0, unbiased=False)
    std_y = Yc.std(dim=0, unbiased=False)
    denom = std_x * std_y
    denom = torch.where(denom < eps, torch.full_like(denom, eps), denom)
    return cov / denom

def dtw_path(x: torch.Tensor, y: torch.Tensor):
    n, m = x.size(0), y.size(0)
    D = x.new_full((n+1, m+1), float('inf'))
    D[0,0] = 0.
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = torch.abs(x[i-1] - y[j-1])
            D[i,j] = cost + torch.min(torch.stack([D[i-1,j], D[i,j-1], D[i-1,j-1]]))
    path = []
    i, j = n, m
    while i>0 or j>0:
        path.append((i-1, j-1))
        if i==0:
            j -= 1
        elif j==0:
            i -= 1
        else:
            choices = {(i-1,j): D[i-1,j], (i,j-1): D[i,j-1], (i-1,j-1): D[i-1,j-1]}
            (i,j), _ = min(choices.items(), key=lambda kv: kv[1])
    path.reverse()
    return D, path

def warp_Y_to_X(X: torch.Tensor, Y: torch.Tensor):
    T, D = X.shape
    Yw = torch.zeros_like(Y)
    dtw_distances = []
    for d in range(D):
        x, y = X[:,d], Y[:,d]
        Dmat, path = dtw_path(x, y)
        dtw_distances.append(Dmat[-1,-1])
        mapping = {}
        for ix, iy in path:
            mapping.setdefault(ix, []).append(iy)
        for ix in range(T):
            if ix in mapping:
                Yw[ix,d] = y[mapping[ix]].mean()
            else:
                Yw[ix,d] = y[ix]
    return Yw, torch.stack(dtw_distances).mean()

def framewise_similarity(X: torch.Tensor, Yw: torch.Tensor, metric: str = 'cosine'):
    """
    按列（每个属性）计算两个 (T, D) 序列在每个维度上的相似度。
    返回 shape (D,) 的向量。
    """
    if metric == 'cosine':
        # 每列做 1D 向量的余弦相似度
        Xn = X / X.norm(dim=0, keepdim=True).clamp(min=1e-8)  # (T,D)
        Yn = Y / Y.norm(dim=0, keepdim=True).clamp(min=1e-8)
        sims = (Xn * Yn).sum(dim=0)  # (D,)
        return sims, sims.mean()
    
    elif metric == 'euclidean':
        dists = ((X - Y) ** 2).sum(dim=0).sqrt()  # (D,)
        return dists, dists.mean()

    else:
        raise ValueError(f"未知 metric: {metric}")

def full_compare(X: torch.Tensor, Y: torch.Tensor):
    """
    完整流程：
      - 对齐前：帧级余弦 / 属性级皮尔逊
      - 对齐后：DTW → 帧级相似度 → 列级皮尔逊
    """
    # —— 对齐前评估 ——
    pre_cos_per, pre_cos_mean = framewise_similarity(X, Y, 'cosine')
    pre_pearson = pearson_corr_matrix(X, Y)
    pre_pearson_mean = pre_pearson.mean()

    # —— DTW 对齐 ——
    Yw, dtw_mean = warp_Y_to_X(X, Y)

    # —— 对齐后帧级相似度 ——
    cos_per, cos_mean = framewise_similarity(X, Yw, 'cosine')
    euc_per, euc_mean = framewise_similarity(X, Yw, 'euclidean')

    # —— 对齐后属性级皮尔逊 ——
    pearson = pearson_corr_matrix(X, Yw)
    pearson_mean = pearson.mean()

    return {
        'pre_cos_mean':     pre_cos_mean,
        'pre_pearson_mean': pre_pearson_mean,
        'Yw':               Yw,
        'dtw_mean':         dtw_mean,
        'cos_per':          cos_per,
        'cos_mean':         cos_mean,
        'euc_per':          euc_per,
        'euc_mean':         euc_mean,
        'pearson':          pearson,
        'pearson_mean':     pearson_mean,
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
dataset.read_input_data(args.dataset, data="", norm=args.datanorm, filter=args.filter)
X = dataset.traj_label[1][:,valid_idx].cuda()
# Y = torch.load("/home/mkdzir/Code/aieffects/mars_smoothing.pt").squeeze(0).cuda()
Y = torch.load("/home/mkdzir/Code/aieffects/mars.pt").squeeze(0).cuda()

max_vals = torch.load("/home/mkdzir/Code/Dataset_mars/organized_data/dynamic/max.pt").cuda()[0][0][valid_idx]
min_vals = torch.load("/home/mkdzir/Code/Dataset_mars/organized_data/dynamic/min.pt").cuda()[0][0][valid_idx]

def minmax_normalize(X: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor, eps=1e-8):
    return (X - min_vals) / (max_vals - min_vals + eps)
X = minmax_normalize(X, min_vals, max_vals)
Y = minmax_normalize(Y, min_vals, max_vals)
# res = full_compare(X, Y)
# print("对齐前平均余弦相似度: ", res['pre_cos_mean'].item())
# print("对齐前平均皮尔逊相关: ", res['pre_pearson_mean'].item())
# print("平均 DTW 距离: ", res['dtw_mean'].item())
# print("对齐后平均余弦相似度: ", res['cos_mean'].item())
# print("对齐后平均欧氏距离: ", res['euc_mean'].item())
# print("对齐后每列皮尔逊相关系数: ", res['pearson'])
# print("对齐后平均皮尔逊相关: ", res['pearson_mean'].item())


total_variation_per_column = torch.abs(X - Y).sum(dim=0)        # shape: (D,)
total_variation_mean = total_variation_per_column.mean()        # 标量

# ==== 二阶差分 ====
# 二阶差分: x[t] - 2 * x[t+1] + x[t+2]
x_diff2 = X[:-2] - 2 * X[1:-1] + X[2:]                           # shape: (T-2, D)
y_diff2 = Y[:-2] - 2 * Y[1:-1] + Y[2:]
second_order_diff_per_column = torch.abs(x_diff2 - y_diff2).sum(dim=0)  # shape: (D,)
second_order_diff_mean = second_order_diff_per_column.mean()            # 标量

# ==== 输出结果 ====
print("每列总变差: ", total_variation_per_column)
print("平均总变差: ", total_variation_mean)
print("每列二阶差分差异: ", second_order_diff_per_column)
print("平均二阶差分差异: ", second_order_diff_mean)