from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments
from models.effect_dataset import MotionTrajectoryDataset
from models.MyConfig import MyConfig
from models.GPT_mars import GPT_mars
import argparse
import os

class DTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.model.model.save_pretrained(output_dir)
        
os.environ["TOKENIZERS_PARALLELISM"] = "false"

valid_idx = [0, 1, 3, 4, 5, 8, 9]


# /ossfs/workspace/aieffects/final_model
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
    parser.add_argument('--filter', type=str, default="迸发", help='筛选数据集')
    parser.add_argument('--PE', type=str, default="learned", help='位置编码类型')
    parser.add_argument('--cond_type', type=str, default="prompt", help='控制类型')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='加载检查点')

    args = parser.parse_args()
    return args

def print_params(**kwargs):

    print("\n\n"+"-"*150)
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print("-"*150+"\n\n")

args = get_parser()
print_params(**vars(args))

# 加载数据集
json_path = '/home/mkdzir/Pytorch/Dataset_mars/keyframes'
imgs_dir = '/home/mkdzir/Pytorch/Dataset_mars/images'
prompt_list = ['Effect_Prompt_1.csv', 'Effect_Prompt_2.csv']
prompt_csv_list = [f"/home/mkdzir/Pytorch/Dataset_mars/{csv_name}" for csv_name in prompt_list]

dataset = MotionTrajectoryDataset(json_path, imgs_dir, prompt_csv_list, valid_idx)
dataset.read_input_data(args.dataset, data="dynamic", norm=args.datanorm, filter=args.filter)



# 根据参数配置config
config = MyConfig(n_embd = 256, dim_head = 32)
config.valid_idx = valid_idx
config.traj_dim = len(valid_idx)
config.img_drop_rate = args.i_drop
config.prompt_drop_rate = args.p_drop
config.use_cfg = args.cfg
config.head_act = args.headact
config.net_path = args.ckp
config.denoise_model = args.denoise_model
config.cond_type = args.cond_type
config.PE = args.PE
config.n_layers = 8
config.c_init()

# 划分训练集和验证集，例如90%训练，10%验证
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
print(f"Training datasets: {len(train_dataset)}, eval datasets: {len(eval_dataset)}")

# 定义训练参数
training_args = TrainingArguments(
    output_dir=args.output,                     # 模型和日志的输出目录
    num_train_epochs=args.epoch,                       # 训练轮数，根据需要调整
    per_device_train_batch_size=args.batchsize,            # 训练批大小
    per_device_eval_batch_size=args.batchsize,             # 验证批大小
    # gradient_accumulation_steps=4,
    warmup_steps=100,                           # 预热步数
    learning_rate=args.lr,                         # 学习率
    weight_decay=0.0001,                       # 权重衰减
    logging_dir=args.logsdir,                      # 日志存储目录
    logging_steps=100,                           # 日志记录步数
    eval_strategy='steps',                      # 评估策略
    eval_steps=1000,                             # 评估步数
    save_steps=1000,                             # 检查点保存步数
    save_total_limit=3,                         # 最多保存的检查点数量
    load_best_model_at_end=True,                # 是否在训练结束时加载最佳模型
    metric_for_best_model='loss',               # 用于评估最佳模型的指标
    fp16=True,
    dataloader_num_workers=60,
    max_grad_norm=1.0,
    overwrite_output_dir=False,
    # deepspeed = "./utils/deepspeed_config.json",
    ignore_data_skip=True,
    report_to="none",
)

# 定义模型
model = GPT_mars(config)
if args.ckp != "":
        model = model.from_pretrained(args.ckp, config=config)


# 定义trainer类
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=dataset.my_collate_fn,
)

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最终模型
trainer.save_model(args.final)