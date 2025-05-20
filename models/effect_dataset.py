import torch
from torch.utils.data import Dataset
import os
import json
import pandas as pd
from PIL import Image
import csv
import numpy as np

def check_and_create_directory(path):
    if not isinstance(path, str):
        raise ValueError("输入不是有效的字符串路径。")
    if not os.path.isdir(path):
        if os.path.exists(path):
            raise ValueError("指定的路径存在，但它不是一个目录。")
        else:
            os.makedirs(path)
            print(f"目录 '{path}' 已创建。")


def remove_indices(lst, indices):
    indices = sorted(indices, reverse=True)
    for index in indices:
        if 0 <= index < len(lst):
            del lst[index]

class MotionTrajectoryDataset(Dataset):
    def __init__(
        self, 
        json_dir, 
        imgs_dir, 
        prompt_csv_list,
        valid_idx = None,
        encoder_name = "OFA-Sys/chinese-clip-vit-base-patch16", 
        save_dir = "/home/mkdzir/Code/Dataset_mars"
        ):
        """
        初始化函数。
    
        参数:
            - json_dir (str): keyframe（轨迹）文件路径
            - imgs_dir: 图层文件路径
            - prompt_csv_dir: 动效描述csv文件(多个文件的列表)
        """
        # 选择有用的特征分量（10维特征中存在恒零的值，可以选择去掉）
        self.valid_idx = valid_idx

        self.json_dir = json_dir
        self.imgs_dir = imgs_dir

        # CFG训练需要的参数以及填充值
        self.img_drop_rate = 0
        self.prompt_drop_rate = 0
        self.cfg_img = Image.new("RGB", (336, 336), (0, 0, 0))
        self.cfg_prompt = ""

        self.json_list = []

        self.save_dir = "/home/mkdzir/Code/Dataset_mars/organized_data"
        self.save_dynamic = "/home/mkdzir/Code/Dataset_mars/organized_data/dynamic"
        check_and_create_directory(self.save_dynamic)
        self.prompt_csv_list = prompt_csv_list
        self.traj_label_file = "traj_label.pt"
        self.mean_file = "mean.pt"
        self.std_file = "std.pt"
        # 保存可直接读取的轨迹tensor的路径，分为全部（all）、有标注的文本描述（des）、无标注文本描述（nodes）
        self.traj_tensor_filename = {
            "all": "traj_label.pt",
            "des": "traj_label_des.pt",
            "nodes": "traj_label_nodes.pt"
        }
        # 保存可直接读取的图像和标注文本（prompt）的路径，分为全部（all）、有标注的文本描述（whitdes）、无标注文本描述（nodes）
        self.img_prompt_filename = {
            "all": "img_prompt.csv",
            "des": "img_prompt_des.csv",
            "nodes": "img_prompt_nodes.csv"
        }
        self.traj_path = None
        self.i_p_path = None
        
        self.imgs_names, self.prompts, self.traj_label = [], [], torch.rand((1,1,1))
        self.static_idx  = {"all": [], "des": [], "nodes": []}
        pass


    def get_effect_prompt(self, effect_name):
        effect_name = effect_name.replace("_", "*")
        for csv_path in self.prompt_csv_list:
            df = pd.read_csv(csv_path)
            if 'mars_url' not in df.columns or '动画文字描述' not in df.columns:
                raise ValueError("CSV 文件中缺少必须的列：'mars_url' 或 '动画文字描述'")
            matching_rows = df[df['mars_url'] == effect_name]
            if matching_rows.empty:
                continue
            prompt_list = matching_rows['动画文字描述'].tolist()
            return prompt_list[0]
        return None
        

    def get_img_prompt(self, effect_name, sep_id):
        effect_name = effect_name.replace("_", "*")
        for filename in self.prompt_csv_list:
            df = pd.read_csv(filename)
            match = df[(df['mars_url'] == effect_name) & (df['sep_id'] == sep_id)]
            if not match.empty:
                return match.iloc[0]['动画图层描述']
        return None


    # 将json文件中的轨迹、图像以及保存在csv文件中的prompt处理为可以直接输入模型的数据并保存到三个有序的列表中，后续直接读取
    def preprocess_file2data(self):
        imgs_list = []
        prompts_list = []
        traj_label_list = []
        idx = 0
        for json_name in self.json_list:
            with open(os.path.join(self.json_dir, json_name), 'r') as f:
                data = json.load(f)
                effect_name = data["name"]
                """
                # 获取图层
                prompt = self.get_prompt(effect_name)
                if prompt == '跳过':
                    effect_name = None
                """
                for layer_idx, img_layer in enumerate(data["layers"]):
                    if img_layer["type"] != "sprite":
                        continue
                    # 从csv文件中获取图层的prompt并将换行替换为空格避免错误
                    prompt = self.get_img_prompt(effect_name, layer_idx)
                    if isinstance(prompt, str):
                        prompt = prompt.replace("\n", " ")
                    if prompt == '跳过':
                        prompt = None
                    img_layer_data = img_layer["data"]
                    position = torch.tensor(img_layer_data["position"])
                    size = torch.tensor(img_layer_data["size"])
                    rotation = torch.tensor(img_layer_data["rotation"])
                    opacity = torch.tensor(img_layer_data["opacity"]).unsqueeze(1)
                    traj_label = torch.cat((position, size, rotation, opacity), dim=1)
                    if traj_label.shape[0] < 60:
                        continue
                    idx += 1

                    imgs_list.append(img_layer["img"][18:])
                    prompts_list.append(prompt)
                    traj_label_list.append(traj_label[:60])
                    print(idx, img_layer["img"], prompt)
        
        assert len(imgs_list) == len(prompts_list) == len(traj_label_list), "数据长度不一致"
        _, _, _, _ = self.Statistics(traj_label_list, self.save_dir)
        all_traj_label = torch.stack(traj_label_list,dim=0)
        # traj_label_list = self.get_relative_position_rot(traj_label_list)
        # 保存数据统计值最大值，最小值，均值，方差
        torch.save(all_traj_label, os.path.join(self.save_dir, self.traj_label_file))

        with open(os.path.join(self.save_dir, self.img_prompt_filename["all"]), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for img,prompt in zip(imgs_list, prompts_list):
                writer.writerow((img,prompt))
        return imgs_list, prompts_list, traj_label_list



    """
    数值统计最大值，最小值，均值，方差
    """
    def Statistics(self, traj_label_list, data_dir):
        data_tensor = torch.stack(traj_label_list)
        reshaped_data = data_tensor.view(-1, data_tensor.size(-1)) 

        max_val = reshaped_data.max(dim=0).values
        min_val = reshaped_data.min(dim=0).values
        max_val = max_val.view(1, 1, -1)
        min_val = min_val.view(1, 1, -1)
        mean = reshaped_data.mean(dim=0).view(1, 1, -1)  
        std = reshaped_data.std(dim=0).view(1, 1, -1)    

        torch.save(max_val, os.path.join(data_dir, 'max.pt'))
        torch.save(min_val, os.path.join(data_dir, 'min.pt'))
        torch.save(mean, os.path.join(data_dir, 'mean.pt'))
        torch.save(std, os.path.join(data_dir, 'std.pt'))
        return max_val, min_val, mean, std


    # 将数据均值方差标准化
    def normalize_traj(self, traj_label_list, dir, mean=None, std=None):
        data_tensor = torch.stack(traj_label_list)
        # 如果均值和方差没有输入，加载均值和方差
        mean = torch.load(os.path.join(dir, 'mean.pt')) if mean is None else mean
        std = torch.load(os.path.join(dir, 'std.pt')) if std is None else std
        normalized_data = (data_tensor - mean) / std
        return normalized_data


    # 将标准化的单个数据还原
    def denormalize_traj(self, normalized_data, dir, mean=None, std=None):
        data_tensor = torch.stack(normalized_data)
        # 如果均值和方差没有输入，加载均值和方差
        mean = torch.load(os.path.join(dir, 'mean.pt')) if mean is None else mean
        std = torch.load(os.path.join(dir, 'std.lspt')) if std is None else std
        mean, std = mean.to(normalized_data.device), std.to(normalized_data.device)
        denormalized_data = data_tensor * std + mean
        return denormalized_data


    def max_min_scale(self, traj_label_list, dir, max_val=None, min_val=None):
        """
        最大最小值缩放
        """
        max_vals = traj_label_list.max(dim=0)[0].max(dim=0)[0]  # 形状(10,)
        min_vals = traj_label_list.min(dim=0)[0].min(dim=0)[0]  # 形状(10,)
        
        max_vals = max_vals.view(1, 1, -1)  # 形状(1,1,10)
        min_vals = min_vals.view(1, 1, -1)  # 形状(1,1,10)
        
        # 归一化公式：(x - min) / (max - min)
        normalized_data = (traj_label_list - min_vals) / (max_vals - min_vals + 1e-8)  # 加小量防止除以0
        
        return normalized_data


    def inverse_max_min_scale(self, scaled_data, dir, max_val=None, min_val=None):
        """
        最大最小值缩放值还原
        """
        max_val = torch.load(os.path.join(dir, 'max.pt')) if max_val is None else max_val
        min_val = torch.load(os.path.join(dir, 'min.pt')) if min_val is None else min_val
        max_val, min_val = max_val.to(scaled_data.device), min_val.to(scaled_data.device)
        # data = torch.stack(scaled_data)
        data = scaled_data * (max_val - min_val) + min_val
        return data


    @staticmethod
    def get_relative_position_rot(traj_label):
        """
        获取相对于第一帧的相对位置和旋转

        参数:
        traj_label (torch.Tensor): 一个n*60x10张量

        返回:
        List[torch.Tensor]: 经过处理的张量
        """
        # traj_label[:,:, 0:3] -= traj_label[:,0, 0:3].clone().unsqueeze(1)
        traj_label[:, :, 3:6] = traj_label[:, :, 3:6].clone() / 10
        # traj_label[:, :, 6:9] = traj_label[:, :, 6:9].clone() / 180
        return traj_label

    @staticmethod
    def ret_relative_position_rot(traj_label):
        traj_label[:, :, 3:6] = traj_label[:, :, 3:6].clone() * 10
        # traj_label[:, :, 0:3] -= traj_label[:, 0, 0:3].clone()
        # traj_label[:, :, 6:9] = traj_label[:, :, 6:9].clone() * 180
        return traj_label

    # 将所有的数据集分为有标注文本描述和无标注文本描述两部分并保存
    def split_data(self):
        traj_label = torch.load(os.path.join(self.save_dir, self.traj_tensor_filename['all']))
        # normalized_traj_list, mean, std = self.normalize_traj(traj_label_list)
        traj_list_des = []
        traj_list_nodes = []
        rows_des = []
        rows_nodes = []
        with open(os.path.join(self.save_dir, self.img_prompt_filename['all']), 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            # assert row_count == len(traj_label), f"数据长度不一致, traj: {len(traj_label_list)}, lines: {row_count}"
            for traj, row in zip(traj_label, csv_reader):
                if row[1] == "":
                    traj_list_nodes.append(traj)
                    rows_nodes.append(row)
                else:
                    traj_list_des.append(traj)
                    rows_des.append(row)
        traj_des = torch.stack(traj_list_des, dim=0)
        traj_nodes = torch.stack(traj_list_nodes, dim=0)
        torch.save(traj_des, os.path.join(self.save_dir, self.traj_tensor_filename['des']))
        torch.save(traj_nodes, os.path.join(self.save_dir, self.traj_tensor_filename['nodes']))
        with open(os.path.join(self.save_dir, self.img_prompt_filename["des"]), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in rows_des:
                writer.writerow(row)
        with open(os.path.join(self.save_dir, self.img_prompt_filename["nodes"]), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in rows_nodes:
                writer.writerow(row)


    def judge_static(self, split="all"):
        self.traj_path = self.traj_tensor_filename[split]
        traj_label_list = torch.load(os.path.join(self.save_dir, self.traj_path))
        b = len(traj_label_list)

        for i in range(b):
            sub_tensor = traj_label_list[i]
            part1 = torch.all(sub_tensor[:, 0:3] == sub_tensor[0, 0:3], dim=1)
            part2 = torch.all(sub_tensor[:, 3:6] == sub_tensor[0, 3:6], dim=1)
            part3 = torch.all(sub_tensor[:, 6:9] == sub_tensor[0, 6:9], dim=1)
            part4 = sub_tensor[:, 9] == sub_tensor[0, 9]

            if torch.all(part1) and torch.all(part2) and torch.all(part3) and torch.all(part4):
                self.static_idx[split].append(i)
        print(f"static traj count: {len(self.static_idx[split])}")
        remove_indices(traj_label_list,self.static_idx[split])
        print("Save to dynamic traj.")
        torch.save(traj_label_list, os.path.join(self.save_dynamic, self.traj_tensor_filename[split]))
        df = pd.read_csv(os.path.join(self.save_dir, self.img_prompt_filename[split]), encoding='utf-8')
        df.drop(self.static_idx[split], inplace=True)
        print("Save to dynamic img-prompt.")
        df.to_csv(os.path.join(self.save_dynamic, self.img_prompt_filename[split]), index=False)


    # 读取已经处理好的保存数据
    def read_input_data(self, split='all', data="dynamic", norm="", filter=""):
        # 选择两个数据文件的保存路径
        self.traj_path = self.traj_tensor_filename[split]
        self.i_p_path = self.img_prompt_filename[split]

        data_dir = self.save_dynamic if data=="dynamic" else self.save_dir
        traj_label = torch.load(os.path.join(data_dir, self.traj_path))
        # 对数据做归一化处理
        if norm=="max_min":
            self.traj_label = self.max_min_scale(traj_label, data_dir)
        elif norm=="mean_std":
            self.traj_label = self.normalize_traj(traj_label, data_dir)
        else:
            self.traj_label = traj_label
        # self.traj_label = self.get_relative_position_rot(self.traj_label)
        # count = 0
        # for idx, traj in enumerate(self.traj_label):
        #     if traj.shape[0] != 60:
        #         print(f"error data {idx}: {traj.shape[0]}")
        #         count += 1
        # print(f"error data count: {count}")

        with open(os.path.join(data_dir, self.i_p_path), mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.imgs_names.append(row[0])
                self.prompts.append(row[1])
        # if split == 'all':
        #     self.Statistics(traj_label_list, data_dir)
        if filter != "": 
            idx = [i for i, s in enumerate(self.prompts) if filter in s]
            self.traj_label = self.traj_label[idx]
            self.prompts = [self.prompts[i] for i in idx]
            self.imgs_names = [self.imgs_names[i] for i in idx]

        self.mean = torch.load(os.path.join(data_dir, "mean.pt"))
        self.Max = torch.load(os.path.join(data_dir, "max.pt"))
        self.Min = torch.load(os.path.join(data_dir, "min.pt"))


    # 将轨迹（B*t*10）沿着时间维度翻转作为数据增强
    def data_augmentation(self):
        self.traj_label = torch.cat([self.traj_label, self.traj_label.flip(1)],dim=0)
        self.imgs_names = self.imgs_names + self.imgs_names
        self.prompts = self.prompts + self.prompts


    # 自定义 collate 函数
    def my_collate_fn(self, batch):
        # print(len(batch))
        traj_batch = []
        imgs_batch = []
        prompts_batch = []
        labels_batch = []
        for item in batch:
            traj_batch.append(item['past_traj'])
            imgs_batch.append(item['imgs'])
            prompts_batch.append(item['prompts'])
            labels_batch.append(item['labels'])

        batch_traj = torch.stack(traj_batch)[:,:-1,:]
        batch_target = torch.stack(labels_batch)
        # 如果没有prompt输入则将整个提示词batch设为None
        # if prompts_batch[0] == 'None':
            # prompts_batch = None
        return {"past_traj": batch_traj,
            "imgs":None, 
            "prompts": prompts_batch, 
            "labels": batch_target,
            }

    def diffusion_collate(self, batch):
        traj_batch = []
        imgs_batch = []
        prompts_batch = []
        for item in batch:
            traj_batch.append(item['past_traj'])
            imgs_batch.append(item['imgs'])
            prompts_batch.append(item['prompts'])

        batch_traj = torch.stack(traj_batch)
        return {"past_traj": batch_traj,
            "imgs":None, 
            "prompts": prompts_batch, 
            "labels": batch_traj[:,0,:],
            }
    
    def my_collate_fn_for_directgen(self, batch):
        # print(len(batch))
        traj_batch = []
        imgs_batch = []
        prompts_batch = []
        labels_batch = []
        for item in batch:
            traj_batch.append(item["past_traj"])
            imgs_batch.append(item['imgs'])
            prompts_batch.append(item['prompts'])
            labels_batch.append(item['labels'])
        
        batch_traj = torch.stack(traj_batch)
        batch_target = torch.stack(labels_batch)
        x = torch.rand(1)
        if x >0.8:
            batch_traj = batch_traj[:,::2,:]
            batch_target = batch_target[:,1::2,:]
        elif x>0.6:
            batch_traj = batch_traj[:,::4,:]
            batch_target = batch_target[:,2::4,:]
        elif x>0.4:
            # 帧长度为60
            s = torch.randint(0,8,(1,))
            batch_traj = batch_traj[:,s:s+49:8,:]
            batch_target = batch_target[:,4+s:s+53:8,:]
        elif x>0.2:
            s = torch.randint(0,4,(1,))
            batch_traj = batch_traj[:,s:s+49:16,:]
            batch_target = batch_target[:,8+s:s+57:16,:]
        elif x>0.1:
            s = torch.randint(0,15,(1,))
            batch_traj = batch_traj[:,s::30,:]
            batch_target = batch_target[:,15+s::30,:]
        else:
            s = torch.randint(0,30,(1,))
            batch_traj = batch_traj[:,s,:]
            batch_target = batch_target[:,30+s,:]            

        return {"past_traj": batch_traj,
            "imgs":imgs_batch, 
            "prompts": prompts_batch, 
            "labels": batch_target,
            }


    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.imgs_names)
    
    
    def __getitem__(self, idx):
        """
        根据索引获取一个样本。
        
        参数:
            - idx (int): 样本索引。
        返回:
            - imgs (PIL.Image): 图像
            - prompts (str): 控制文本
            - past_traj (Tensor): 历史轨迹
            - labels (Tensor): 轨迹标签值
        """
        image_path = os.path.join(self.imgs_dir, self.imgs_names[idx])
        # image = Image.open(image_path)

        prompt_context = self.prompts[idx]
        trajectory = self.traj_label[idx]
        if self.valid_idx is not None:
            trajectory = trajectory[:,self.valid_idx]
        
        return {
            "past_traj": trajectory,
            "imgs":image_path,
            "prompts": prompt_context, 
            "labels": trajectory,
            }


if __name__ == "__main__":
    json_path = '/home/mkdzir/Code/Dataset_mars/keyframes'
    imgs_dir = '/home/mkdzir/Code/Dataset_mars/images'
    prompt_list = ['Effect_Prompt_1.csv', 'Effect_Prompt_2.csv']
    img_prompt_list = [
    'img_description_batch_1.csv',
    'img_description_batch_2.csv',
    'img_description_batch_3.csv',
    'img_description_batch_4.csv',
    'img_description_batch_5.csv'
]
    prompt_csv_list = [f"/ossfs/workspace/dataset_v1/img_des/{csv_name}" for csv_name in img_prompt_list]
    datasets = MotionTrajectoryDataset(json_path, imgs_dir, prompt_csv_list)
    # datasets.preprocess_file2data()
    # datasets.split_data()
    datasets.read_input_data("des", data="dynamic")
    # for data in ["all", "des", "nodes"]:
    #     datasets.judge_static(data)
    # 加载mean和std

    pos1 = abs(datasets.traj_label).mean(0).mean(0)
    print(pos1)
    mean = torch.load("/ossfs/workspace/dataset_v1/organized_data/dynamic/mean.pt")
    std = torch.load("/ossfs/workspace/dataset_v1/organized_data/dynamic/std.pt")
    Max = torch.load("/ossfs/workspace/dataset_v1/organized_data/dynamic/max.pt")
    Min = torch.load("/ossfs/workspace/dataset_v1/organized_data/dynamic/min.pt")
    from torch.utils.data import DataLoader
    # dataloader = DataLoader(datasets, batch_size=2, shuffle=True, collate_fn=datasets.my_collate_fn)
    # for batch in dataloader:
        # print(batch["imgs"], batch["prompts"], batch["past_traj"], batch["labels"].shape)
        # break