import torch
from models.effect_dataset import MotionTrajectoryDataset
import csv
# 假设输入数据是一个形状为 (n, 60, 10) 的 tensor
# n 是轨迹数，60 是帧数，10 是每帧的属性数
def compute_change_and_labels(trajectories):
    if torch.any(torch.isnan(trajectories)) or torch.any(torch.isinf(trajectories)):
        print("数据包含 NaN 或 inf，进行数据清理")
    # 计算每个轨迹帧之间的变化程度（总和）
    n, frames, attrs = trajectories.shape
    frame_changes = torch.zeros(n, frames - 1)

    for i in range(n):
        for j in range(1, frames):
            # 计算第j帧和第j-1帧之间的变化（欧氏距离）
            frame_changes[i, j-1] = torch.norm(trajectories[i, j] - trajectories[i, j-1])

    # 计算每个轨迹的变化总和
    sum_changes = frame_changes.sum(dim=1)
    
    # 计算所有轨迹的总体平均变化（以变化总和计）
    overall_avg_change = sum_changes.mean()
    print(overall_avg_change)
    # 根据变化总和与总体平均变化比较来打标
    labels = (sum_changes >= overall_avg_change).int()  # 小于平均值标记0，否则标记1
    
    return labels

def process_sequences_with_categories(data):
    # 1. 计算最大值和最小值的差值
    max_values = torch.max(data, dim=1)[0]  # (n, 10) 每列最大值
    min_values = torch.min(data, dim=1)[0]  # (n, 10) 每列最小值
    S = max_values - min_values  # (n, 10) 每列差值

    # 2. 对S的不同部分求和，得到4个不同的Q
    Q1 = torch.sum(S[:, :3], dim=1)  # (n,) xyz轴位置的和
    Q2 = torch.sum(S[:, 3:6], dim=1)  # (n,) 旋转的和
    Q3 = torch.sum(S[:, 6:9], dim=1)  # (n,) 缩放的和
    Q4 = S[:, -1]  # (n,) 透明度差值

    # 3. 计算四个Q的平均值
    Q1_mean = torch.mean(Q1)
    Q2_mean = torch.mean(Q2)
    Q3_mean = torch.mean(Q3)
    Q4_mean = torch.mean(Q4)

    # 4. 用各自的平均值将Q分为两类（0或1）
    categories1 = (Q1 > Q1_mean).int()  # (n,) 类别0或1
    categories2 = (Q2 > Q2_mean).int()  # (n,) 类别0或1
    categories3 = (Q3 > Q3_mean).int()  # (n,) 类别0或1
    categories4 = (Q4 > Q4_mean).int()  # (n,) 类别0或1

    return categories1, categories2, categories3, categories4


def modify_csv_based_on_list(csv_file, output_file, modification_list):
    """
    根据修改列表修改CSV文件
    
    参数:
        csv_file: 输入的CSV文件路径
        output_file: 输出的CSV文件路径
        modification_list: 修改列表，0表示"缓慢地"，1表示"快速地"
    """
    with open(csv_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if len(row) >= 2:  # 确保有第二列
                if i < len(modification_list):  # 确保列表有对应元素
                    replacement = "缓慢地" if modification_list[i] == 0 else "快速地"
                    row[1] = row[1].replace('-', replacement)
                else:
                    print(f"警告: 列表长度不足，第 {i} 行未修改")
            writer.writerow(row)


def modify_csv_based_on_categories(csv_file, output_file, categories1, categories2, categories3, categories4):
    """
    根据四个分类指示器修改CSV文件中的内容
    
    参数:
        csv_file: 输入的CSV文件路径
        output_file: 输出的CSV文件路径
        categories1: 依据S[:, :3]的分类指示器，长度为n的序列
        categories2: 依据S[:, 3:6]的分类指示器，长度为n的序列
        categories3: 依据S[:, 6:9]的分类指示器，长度为n的序列
        categories4: 依据S[:, -1]的分类指示器，长度为n的序列
    """
    with open(csv_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if len(row) >= 2:  # 确保有第二列数据（即不是空行或者没有第二列）
                # 先获取第二列的内容
                second_col = row[1]
                
                # 根据categories1修改
                if categories1[i] == 1:
                    second_col += " 移动幅度大"
                else:
                    second_col += " 移动幅度小"
                
                # 根据categories2修改
                if categories2[i] == 1:
                    second_col += " 伸缩幅度大"
                else:
                    second_col += " 伸缩幅度小"
                
                # 根据categories3修改
                if categories3[i] == 1:
                    second_col += " 旋转幅度大"
                else:
                    second_col += " 旋转幅度小"
                
                # 根据categories4修改
                if categories4[i] == 1:
                    second_col += " 透明度变化幅度大"
                else:
                    second_col += " 透明度变化幅度小"
                
                # 更新第二列的内容
                row[1] = second_col
            
            # 写入修改后的行
            writer.writerow(row)


label_list = [
    " ",  # 0
    " 在逐渐浮现",      # 1
    " 在逐渐消失",      # 2
    " 出现然后消失", # 3
    " 在闪烁"       # 4
]

def classify_opacity_sequence(sequence):
    n = len(sequence)
    
    # 判断序列的单调性
    if torch.all(torch.diff(sequence) > 0):
        return 1  # 单调递增, 数字标记为1
    elif torch.all(torch.diff(sequence) < 0):
        return 2  # 单调递减, 数字标记为2
    
    # 检查是否先增大后减小（出现然后消失）
    max_index = torch.argmax(sequence)
    if torch.all(sequence[:max_index] < sequence[max_index]) and torch.all(sequence[max_index:] > sequence[-1]):
        return 3  # 先增后减, 数字标记为3
    
    # 检查是否先减小后增大（闪烁）
    min_index = torch.argmin(sequence)
    if torch.all(sequence[:min_index] > sequence[min_index]) and torch.all(sequence[min_index:] < sequence[-1]):
        return 4  # 先减后增, 数字标记为4

    return 0  # 其他情况，默认为0（无法分类）

def classify_opacity_matrix(matrix):
    n, m = matrix.shape
    labels = []
    
    for i in range(n):
        sequence = matrix[i, :]
        label = classify_opacity_sequence(sequence)
        labels.append(label)
    
    return labels

# 读取原 CSV 文件，并根据标签添加类别文本
def add_labels_to_csv(input_csv, output_csv, labels):
    # 读取原 CSV 文件
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # 确保有足够的行数
    if len(rows) != len(labels):
        raise ValueError("CSV文件的行数与标签数量不匹配")
    
    # 将对应的类别文本添加到第二列的末尾
    for i in range(len(rows)):
        label_text = label_list[labels[i]]  # 用列表索引获取文本标签
        if len(rows[i]) > 1:
            rows[i][1] += label_text  # 在第二列的末尾添加类别文本
        else:
            rows[i].append(label_text)  # 如果第二列没有元素，则添加到末尾

    # 写入更新后的 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)




json_path = '/home/mkdzir/Code/Dataset_mars//keyframes'
imgs_dir = '/home/mkdzir/Code/Dataset_mars/images'
datasets = MotionTrajectoryDataset(json_path, imgs_dir, [])
datasets.read_input_data("des", data="", norm = "max_min")
# print(datasets.traj_label.shape)

# ------------------------------------------------------------------------
# labels = compute_change_and_labels(datasets.traj_label)
# modify_csv_based_on_list("/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des_split.csv", "/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des_cls.csv", labels)

# c1, c2, c3, c4 = process_sequences_with_categories(datasets.traj_label)
# modify_csv_based_on_categories("/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des_cls.csv", "/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des.csv", c1, c2, c3, c4)






# ------------------------------------------------------------------------
labels = classify_opacity_matrix(datasets.traj_label[:,:,-1])

# text_labels = [label_list[label] for label in labels]

input_csv = "/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des.csv"
output_csv = "/home/mkdzir/Code/Dataset_mars/organized_data/img_prompt_des_aug5.csv"
add_labels_to_csv(input_csv, output_csv, labels)