## AIGC动效生成

利用文本控制（图层初始位置信息）自动图层生成动效轨迹

## 仓库内容

**models**

- DiT.py: 噪声预测模型
- Diffusion.py: 扩散训练和推理模型
- effect_dataset.py: 数据集定义
- MyConfig.py: 配置文件

**scripts**

- d_step_train.sh:  扩散模型训练脚本
- cmp_inference.sh: 推理脚本

diffusion_trainer.py: 扩散模型训练代码

diffusion_inference.py: 扩散模型推理代码

pipline.py: 文本到动效生成的pipline

## 训练和推理

在effect_dataset.py内调用三个方法处理数据集

```python

datasets.preprocess_file2data() #读取图层轨迹，读取csv文件中的图层文本描述，并保存图层与文本描述的对应

datasets.split_data() #将数据集拆分为有文本描述和无文本描述两部分

datasets.read_input_data("des", data="dynamic") #读取数据集
```
```bash
bash scripts/d_step_train.sh #开始训练

bash scripts/cmp_inference.sh #推理脚本
```

### 完整pipline调用方法

pipline.py文件实现了整个多图层动效生成的pipline，其中图层路径，文本描述和初始状态需要进行指定

```python
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="/home/mkdzir/Pytorch/aieffects/save_ckp/DiT_state/final_model", help='模型路径')
    parser.add_argument('--img', type=str, default="/ossfs/workspace/aieffects/test_pic/lcar.png", help='图像路径')
    parser.add_argument('--prompt', type=str, default="", nargs='+', help='提示词')
    parser.add_argument('--start_state', type=str, default="", help='提示词')
    parser.add_argument('--output', type=str, default="pipline.json", help='json文件输出路径')
    parser.add_argument('--guide', type=float, default=0.5, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    parser.add_argument('--llm', type=str, default="qwen/Qwen2.5-3B-Instruct", help='')
```


直接使用`python pipline.py`可以调用整个pipline，模型使用的是`save_ckp/DiT_state/final_model`目录下的检查点，这是一个输入时包含初始位置的模型，在第168行进行调用：

```python
    mars = sampler(xt_noise, None, m_text_verb, state[:, config.valid_idx])
```





