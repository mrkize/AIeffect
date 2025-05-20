import math
import torch
from torch import nn
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, PreTrainedModel, PretrainedConfig
from PIL import Image
import os
from .utils import TimeEmbedding, CrossAttention, Swish
from .MyConfig import MyConfig


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # causal attention
        use_rope = False
        if config.PE == "rope":
            use_rope = True
        self.attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = None, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size,
            rope = use_rope)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # img-feature crossattention
        self.img_cross_attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = config.visual_dim, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size,
            rope = use_rope)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        # prompt-feature crossattention
        self.prompt_cross_attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = config.prompt_dim, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size,
            rope = use_rope)
        self.ln_4 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = Swish(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward


    def forward(self, x, img_embd, prompt_embd):
        x = x + self.attn(self.ln_1(x))
        if img_embd is not None:
            x = x + self.img_cross_attn(self.ln_2(x), img_embd)
        if prompt_embd is not None:
            x = x + self.prompt_cross_attn(self.ln_3(x), prompt_embd)
        x = x + self.mlpf(self.ln_4(x))
        return x


class Net(PreTrainedModel):
    config_class = MyConfig
    
    def __init__(self, config):
        super().__init__(config)
        assert config.traj_dim is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.clip_path = config.clip_path
        
        assert config.n_heads*config.dim_head == config.n_embd, f"hidden size {config.n_embd} != {config.n_heads} heads of {config.dim_head}"
        assert config.PE in ["rope", "learned"], f"PE must be rope or learned, but got {config.PE}"
        # 初始化CLIP视觉编码器和文本编码器
        self.processor = ChineseCLIPProcessor.from_pretrained(self.clip_path)
        self.clip = ChineseCLIPModel.from_pretrained(self.clip_path)
        for param in self.clip.parameters():
            param.requires_grad = False

        self.transformer = nn.ModuleDict(dict(
            curve_embedding = nn.Linear(config.traj_dim, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.pos_embedding = None
        if config.PE == "learned":
            self.pos_embedding =  nn.Embedding(config.block_size, config.n_embd)

        self.time_embedding = TimeEmbedding(config.n_embd)
        self.head_fc = nn.Linear(config.n_embd, config.traj_dim)

        
        # 与cfg训练有关的参数
        self.use_cfg = config.use_cfg
        self.img_drop_rate = config.img_drop_rate
        self.prompt_drop_rate = config.prompt_drop_rate
        self.cfg_img = Image.new("RGB", (336, 336), (0, 0, 0))
        self.cfg_prompt = ""


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # def save_pretrained(self, save_directory):
    #     os.makedirs(save_directory, exist_ok=True)
    #     original_state_dict = self.state_dict()
    #     state_dict_to_save = {k: v for k, v in original_state_dict.items() if not k.startswith('clip')}
    #     torch.save(state_dict_to_save, os.path.join(save_directory, 'pytorch_model.bin'))
    #     self.config.save_pretrained(save_directory)
    #     print(f'Saved model to {save_directory}')


    @torch.no_grad()
    def get_embd(self, imgs, prompts, device):
        self.clip.eval()
        if imgs is not None:
            inputs_img = self.processor(images=imgs, return_tensors="pt").to(device)
            img_embd = self.clip.vision_model(**inputs_img).last_hidden_state
        else:
            img_embd = None
        if prompts is not None:
            inputs_prompt = self.processor(text=prompts, padding=True, return_tensors="pt").to(device)
            prompt_embd = self.clip.text_model(**inputs_prompt).last_hidden_state
        else:
            prompt_embd = None
        return img_embd, prompt_embd

    @staticmethod
    def replace_with_empty_strings_efficient(string_list, drop_ratio):
        num_to_replace = int(len(string_list) * drop_ratio)
        indices = torch.randperm(len(string_list)).tolist()
        for i in indices[:num_to_replace]:
            string_list[i] = ""
        return string_list


    def drop_data(self, imgs, prompts):
        """
        CFG--随机丢弃图像和提示词
        """
        if self.img_drop_rate == 1:
            imgs = None
        elif self.img_drop_rate > 0:
            use_img = torch.rand(len(imgs)) > self.img_drop_rate
            imgs = [img if flag else self.cfg_img for img, flag in zip(imgs, use_img)]
        if self.prompt_drop_rate > 0:
            prompts = self.replace_with_empty_strings_efficient(prompts, self.prompt_drop_rate)
        return imgs, prompts


    def forward(self, x, t, imgs, prompts):
        b, f , dim = x.size()
        pos = torch.arange(0, f, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.curve_embedding(x)
        if self.pos_embedding is not None:
            pos_emb = self.pos_embedding(pos)
            x = x+pos_emb
        x = self.transformer.drop(x)
        t = self.time_embedding(t).unsqueeze(1).expand(x.shape)
        x = x + t

        if self.training and self.use_cfg:
            imgs, prompts = self.drop_data(imgs, prompts)
               
        img_embd, prompt_embd = self.get_embd(imgs, prompts, x.device)
        for block in self.transformer.h:
          x = block(x, img_embd, prompt_embd)
        x = self.head_fc(x)
        return x

if __name__ == '__main__':
    from utils import TimeEmbedding, CrossAttention, Swish
    from MyConfig import MyConfig
    ckp = "/ossfs/workspace/aieffects/save_ckp/diffusion/tmp"
    config = MyConfig()
    # model = Net.from_pretrained(ckp, config=config, ignore_mismatched_sizes=True)
    model = Net(config).cuda()
    x = torch.randn(1, 60, 10).cuda()
    t = torch.randint(0, 100, (1,)).cuda()
    y = model(x, t, None, None)
    # model = Net.from_pretrained('test',config=config)
    print(y.shape)