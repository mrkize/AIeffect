from torch import nn
import torch
import math
from .MyConfig import MyConfig
from .utils import TimeEmbedding, CrossAttention, Swish, curve_smoothing
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, PreTrainedModel, AutoTokenizer
from PIL import Image
import os


"""
DiT 和Diffusion_net基本相同，唯一的差别是没有使用CrossAttention来融合其他模态的信息，而是使用
自适应归一化（AdaLN）来融合信息
"""


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.emb_size=config.n_embd
        self.nhead=config.n_heads

        # conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        )

        # layer norm
        self.norm1=nn.LayerNorm(config.n_embd)
        self.norm2=nn.LayerNorm(config.n_embd)

        # multi-head self-attention
        self.attn = CrossAttention(
            query_dim = config.n_embd,
            context_dim = None,
            heads = config.n_heads,
            dim_head = config.dim_head,
            dropout=config.attn_pdrop,
            block_size=config.block_size,
            rope = config.PE=="rope",)

        # feed-forward
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = Swish(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlpf(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Dit_head(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, config):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(config.n_embd, config.traj_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.cond_dim, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x




class DiT(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        assert config.traj_dim is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.clip_path = config.clip_path
        self.cond_type = config.cond_type

        assert config.n_heads * config.dim_head == config.n_embd, f"hidden size {config.n_embd} != {config.n_heads} heads of {config.dim_head}"

        self.transformer = nn.ModuleDict(
            dict(
                curve_embedding=nn.Linear(config.traj_dim, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(
                    [DiTBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            ))
        self.time_embedding = TimeEmbedding(config.n_embd)
        self.head = Dit_head(config)

        # 与cfg训练有关的参数
        self.use_cfg = config.use_cfg
        self.img_drop_rate = config.img_drop_rate
        self.prompt_drop_rate = config.prompt_drop_rate
        self.cfg_img = Image.new("RGB", (336, 336), (0, 0, 0))
        self.cfg_prompt = ""
        self.register_buffer(
            'default_start_state',
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                 1.0])[config.valid_idx])
        self.start_state_proj = nn.Linear(config.traj_dim,
                                          int(config.n_embd / 2))
        # self.transformer.apply(self._init_weights)
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.zero_out()

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6, ))

        # 在初始化其他参数之后初始化CLIP视觉编码器和文本编码器
        self.processor = ChineseCLIPProcessor.from_pretrained(self.clip_path)
        self.clip = ChineseCLIPModel.from_pretrained(self.clip_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.clip_path)
        for param in self.clip.parameters():
            param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def zero_out(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer.h:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    @torch.no_grad()
    def get_embd(self, imgs, prompts, device):
        self.clip.eval()
        img_embd = None
        prompt_embd = None
        if prompts is not None:
            inputs_prompt = self.processor(text=prompts,
                                           padding=True,
                                           return_tensors="pt").to(device)
            prompt_embd = self.clip.text_model(**inputs_prompt)[0][:, 0, :]
        if imgs is not None:
            inputs_img = self.processor(images=imgs,
                                        return_tensors="pt").to(device)
            img_embd = self.clip.vision_model(**inputs_img)[1]
        return img_embd, prompt_embd

    @staticmethod
    def replace_with_empty_strings_efficient(string_list, drop_ratio):
        num_to_replace = int(len(string_list) * drop_ratio)
        indices = torch.randperm(len(string_list)).tolist()
        for i in indices[:num_to_replace]:
            string_list[i] = ""
        return string_list

    @staticmethod
    def replace_with_cfg_efficient(img_list, cfg, drop_ratio):
        num_to_replace = int(len(img_list) * drop_ratio)
        indices = torch.randperm(len(img_list)).tolist()
        for i in indices[:num_to_replace]:
            img_list[i] = cfg
        return img_list

    def drop_data(self, imgs, prompts):
        if self.config.cond_type == "img":
            prompts = None
            if self.img_drop_rate > 0 and imgs is not None:
                imgs = self.replace_with_cfg_efficient(imgs, self.cfg_img,
                                                       self.img_drop_rate)
        elif self.config.cond_type == "prompt":
            imgs = None
            if self.prompt_drop_rate > 0 and prompts is not None:
                prompts = self.replace_with_empty_strings_efficient(
                    prompts, self.prompt_drop_rate)
        else:
            imgs = None
            prompts = self.replace_with_empty_strings_efficient(
                prompts, self.prompt_drop_rate)
            # start_state[replacement_mask] = self.default_start_state.expand_as(start_state)[replacement_mask]
        return imgs, prompts

    @staticmethod
    def sinusoidal_pos_embedding(seq_length, d_model, base):
        # Create the position indices and dimension indices
        position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(
            base,
            2 * (torch.arange(d_model, dtype=torch.float32) // 2) / d_model)
        # Compute the positional encoding matrix
        pos_embed = torch.zeros((seq_length, d_model))
        pos_embed[:, 0::2] = torch.sin(position / div_term[0::2])
        pos_embed[:, 1::2] = torch.cos(position / div_term[1::2])
        return pos_embed

    def forward(self, x, t, imgs, prompts, start_state=None):
        b, f, dim = x.size()
        pos = torch.arange(0, f, dtype=torch.long,
                           device=x.device).unsqueeze(0)
        x = self.transformer.curve_embedding(x)
        pos_emb = self.transformer.wpe(pos)
        x = x + pos_emb
        x = self.transformer.drop(x)
        t = self.time_embedding(t).unsqueeze(1).expand(x.shape)
        x = x + t

        if self.training and self.use_cfg:
            imgs, prompts = self.drop_data(imgs, prompts)
        img_embd, prompt_embd = self.get_embd(imgs, prompts, x.device)
        if self.cond_type == "prompt":
            c = prompt_embd
        elif self.cond_type == "img":
            c = img_embd
        else:
            assert start_state is not None and prompt_embd is not None
            cond_pos_embd = self.start_state_proj(start_state)
            c = torch.cat([prompt_embd, cond_pos_embd], dim=-1)
        for block in self.transformer.h:
            x = block(x, c)
        x = self.head(x, c)
        return curve_smoothing(x)


    def diffusion_inference(self,
                            x,
                            t,
                            img_embd,
                            prompt_embd,
                            start_state=None):
        b, f, dim = x.size()
        pos = torch.arange(0, f, dtype=torch.long,
                           device=x.device).unsqueeze(0)
        x = self.transformer.curve_embedding(x)
        pos_emb = self.transformer.wpe(pos)
        x = x + pos_emb
        t = self.time_embedding(t).unsqueeze(1).expand(x.shape)
        x = x + t

        if self.cond_type == "prompt":
            c = prompt_embd
        elif self.cond_type == "img":
            c = img_embd
        else:
            assert start_state is not None and prompt_embd is not None
            cond_pos_embd = self.start_state_proj(start_state)
            c = torch.cat([prompt_embd, cond_pos_embd], dim=-1)
        # print("condition shape:",c.shape)
        for block in self.transformer.h:
            x = block(x, c)
        x = self.head(x, c)
        return x


if __name__=='__main__':
    T=1000
    config = MyConfig()
    dit = DiT(config).cuda()
    x = torch.rand(1,60,10).cuda()
    t = torch.randint(0,T,(1,)).cuda()
    y = torch.randint(0,10,(1,))
    outputs=dit(x,t,None,[""])
    print(outputs.shape)
