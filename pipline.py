import torch
from PIL import Image
from models.DiT import DiT
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.MyConfig import MyConfig
from models.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
import torch.nn.functional as F
from mars.mars_converter import generate_json_from_keyframe_data_v1
import argparse
import copy
from PIL import Image
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mars_params():
    def __init__(self, layer_images, prompt, start_state=None):
        self.layer_images = layer_images
        self.prompt = prompt
        self.default_start_state = torch.tensor([1,1,0,1,1,1,0,0,0,1], dtype = torch.float32).to(device)
        if start_state is not None:
            self.start_state = start_state
        else:
            self.start_state = self.default_start_state.unsqueeze(0).repeat(len(layer_images),1).to(device)

def get_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ckp', type=str, default="/home/mkdzir/Pytorch/aieffects/save_ckp/DiT_state/final_model", help='模型路径')
    parser.add_argument('--img', type=str, default="/ossfs/workspace/aieffects/test_pic/lcar.png", help='图像路径')
    parser.add_argument('--prompt', type=str, default="", nargs='+', help='提示词')
    parser.add_argument('--start_state', type=str, default="", help='提示词')
    parser.add_argument('--output', type=str, default="pipline.json", help='json文件输出路径')
    parser.add_argument('--guide', type=float, default=0.5, help='引导权重')
    parser.add_argument('--model', type=str, default="", help='')
    parser.add_argument('--llm', type=str, default="qwen/Qwen2.5-3B-Instruct", help='')
    args = parser.parse_args()
    return args


def load_clip_from_dit(ckp):
    config = MyConfig.from_pretrained(ckp)
    DiT_model = DiT.from_pretrained(ckp, config=config).to(device)
    clip = DiT_model.clip.eval()
    processor = DiT_model.processor
    return DiT_model, config, clip, processor


# 计算图像与文本的相似度进行匹配
def calculate_similarities(clip, processor, texts, image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip(**inputs)
    similarity = outputs.logits_per_image.softmax(dim=1)  # p
    max_similarities, best_match_indices = similarity.max(dim=1)
    threshold = 0.9
    dynamic_idx = []
    m_texts = []
    for i, image_path in enumerate(image_paths):
        if max_similarities[i].item() > threshold:
            matched_text = texts[best_match_indices[i]]
            m_texts.append(matched_text)
            dynamic_idx.append(i)
            print(i, ":", image_path, "与", matched_text, "相似度为", max_similarities[i].item())
    return m_texts, dynamic_idx


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].splitlines()[0]
    response = response.replace("。",".").split(".")[0]
    print(response)
    return response


# 使用一维卷积对生成曲线进行平滑
def curve_smoothing(all_mars):
    b, f, d = all_mars.shape
    all_mars = all_mars.permute(0, 2, 1).reshape(-1,1,f)
    kernel_size = 3
    kernel = (torch.ones(1, 1, kernel_size) / kernel_size).to(device)
    padding = kernel_size // 2
    all_mars_padded = F.pad(all_mars, (padding, padding), mode='replicate')
    smoothed_all_mars = F.conv1d(all_mars_padded, kernel)
    return smoothed_all_mars.reshape(b,d,f).permute(0, 2, 1)


def segmented_curve_smoothing(all_mars):
    b, f, d = all_mars.shape
    all_mars = all_mars.permute(0, 2, 1).reshape(-1, 1, f)
    kernel_size = 3
    kernel = (torch.ones(1, 1, kernel_size) / kernel_size).to(device)
    padding = kernel_size // 2
    all_mars_padded = F.pad(all_mars, (padding, padding), mode='replicate')
    smoothed_all_mars = F.conv1d(all_mars_padded, kernel)
    smoothed_all_mars = smoothed_all_mars.reshape(b, d, f).permute(0, 2, 1)

    # 对前后各K帧进行valid卷积
    K = kernel_size  # 可根据需要调整K值
    if K > 0 and f >= K + kernel_size - 1:
        # 处理前K帧
        front_part = smoothed_all_mars[:, :K + kernel_size - 1, :]
        front_conv = front_part.permute(0, 2, 1).reshape(-1, 1, K + kernel_size - 1)
        front_smoothed = F.conv1d(front_conv, kernel, padding=0)
        front_smoothed = front_smoothed.reshape(b, d, K).permute(0, 2, 1)
        smoothed_all_mars[:, :K, :] = front_smoothed

        # 处理后K帧
        end_part = smoothed_all_mars[:, -(K + kernel_size - 1):, :]
        end_conv = end_part.permute(0, 2, 1).reshape(-1, 1, K + kernel_size - 1)
        end_smoothed = F.conv1d(end_conv, kernel, padding=0)
        end_smoothed = end_smoothed.reshape(b, d, K).permute(0, 2, 1)
        smoothed_all_mars[:, -K:, :] = end_smoothed

    return smoothed_all_mars


def generate_mars(args, mars_gen_info):
    template = f"""
    原始描述文本：一位穿着红色衣服的女士旁边一个红色的礼盒伸缩跳动，周围两枚红色的爱心上下移动.
    解析的动态描述：红色的礼盒伸缩跳动,两枚红色的爱心上下移动.
    原始描述文本：绿色光球和红包在伸缩跳动,金币在伸缩跳动.
    解析的动态描述：绿色光球伸缩跳动,红包伸缩跳动,金币在伸缩跳动.
    原始描述文本：{mars_gen_info.prompt}.
    解析的动态描述：
    """

    model, tokenizer = load_model_and_tokenizer(args.llm)
    model = model.to(device)
    response = generate_response(model, tokenizer, template)
    descriptions = response.replace('，', ',').split(',')
    DiT_model, config, clip, processor = load_clip_from_dit(args.ckp)
    m_texts, dynamic_idx = calculate_similarities(clip, processor,
                                                  descriptions,
                                                  mars_gen_info.layer_images)

    print(dynamic_idx)
    extract_verb = f"""
    原始描述文本：蓝色的标语向外迸发,黄色的小车伸缩跳动,伴随着彩带在飘动.
    解析的单独描述：向外迸发,伸缩跳动,飘动.
    原始描述文本：绿色光球伸缩跳动,红包伸缩跳动,金币在伸缩跳动.
    解析的单独描述：伸缩跳动,伸缩跳动,伸缩跳动.
    原始描述文本：{",".join(text for text in m_texts)}.
    解析的单独描述：
    """
    extracted_verb = generate_response(model, tokenizer, extract_verb)
    m_text_verb = extracted_verb.strip('.').replace('，', ',').split(',')

    sampler = GaussianDiffusionSampler(
        model=DiT_model,
        config=config,
        w=args.guide,
    ).to(device)
    sampler.eval()

    # 静态图层使用默认的初始位置填充，如果有初始位置使用扩展到60帧即可
    B = len(m_text_verb)
    state = mars_gen_info.start_state[dynamic_idx]
    xt_noise = torch.randn(size=[B, 60, len(config.valid_idx)], device=device)
    mars = sampler(xt_noise, None, m_text_verb, state[:, config.valid_idx])

    # 将预测的valid_idx回填到原始的10分量中
    pred_mars = torch.zeros(B, 60, 10).to(device)
    pred_mars[:, :, config.valid_idx] = mars.clone()

    # 将预测的动态效果与静态效果合并
    all_mars = mars_gen_info.start_state.unsqueeze(1).repeat(1, 60, 1)
    all_mars[dynamic_idx, :, :] = pred_mars.clone()
    all_mars = curve_smoothing(all_mars)

    # 写入文件
    with open("utils/json_template.json", 'r') as f:
        data = json.load(f)
        layer = copy.deepcopy(data["layers"][0])
        data["layers"] = []
        for img, traj in zip(mars_gen_info.layer_images, all_mars):
            layert = copy.deepcopy(layer)
            # 轨迹分量position: 0-2, size: 3-5, rotation: 6-8 opacity: 9
            position = traj[:, :3]
            size = traj[:, 3:6]
            rotation = traj[:, 6:9]
            opacity = traj[:, 9]
            layert['data']["position"] = position.tolist()
            layert['data']["size"] = size.tolist()
            layert['data']["rotation"] = rotation.tolist()
            layert['data']["opacity"] = opacity.tolist()
            layert["img"] = img
            data["layers"].append(layert)
        mars_json = generate_json_from_keyframe_data_v1(data)
        json_str = json.dumps(mars_json, ensure_ascii=False)
        json_str = json_str.replace("A_NuD6S5JtDtkAAAAAAAAAAAAADlB4AQ",
                                    mars_gen_info.prompt)
        json_str = json_str.replace('\\"', '"').replace('\\\\',
                                                        '\\').strip('"')
    # 将json写入本地文件
    with open(args.output, 'w') as jo:
        jo.write(json_str)
    return json_str


if __name__ == "__main__":
    args = get_parser()
    # layer_images： List 图层路径
    # prompt： str 提示词
    # start_state： List 初始状态，一个B*10的tensor，表示每个图层第一帧的状态
    # layer_images, prompt, start_state输入到mars_params中，然后使用generate_mars调用
    layer_images = [
        "test_pic/sun.png", "test_pic/cloud.png", "test_pic/tree.png"
    ]
    prompt = "太阳伸缩跳动，云朵向下飘落"
    start_state = torch.tensor([
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, -1, 0, 1, 1, 1, 0, 0, 0, 1]], dtype=torch.float32).to(device)


    mars_gen_info = mars_params(layer_images, prompt, start_state)

    json_str = generate_mars(args, mars_gen_info)
