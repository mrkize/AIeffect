import math

import torch
import torch.nn as nn
from torch import einsum
from torch.nn import functional as F
import numpy as np
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPTextModel, ChineseCLIPVisionModel, PreTrainedModel, PretrainedConfig
from einops import rearrange, repeat
from PIL import Image
# -----------------------------------------------------------------------------
# 模型参数定义
# pip install opencv-python
# export HF_ENDPOINT=https://hf-mirror.com
class MyConfig(PretrainedConfig):
    model_type = "GPT_traj"
    def __init__(self, traj_dim=10, hidden_size=768, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = 12
        self.n_heads = 8
        self.n_embd =  512
        self.dim_head = 64
        self.visual_dim = 768
        self.prompt_dim = 768

        self.traj_dim = 2
        self.block_size = 1000

        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.clip_path = "OFA-Sys/chinese-clip-vit-base-patch16"
        self.img_drop_rate = 0
        self.prompt_drop_rate = 0
        self.use_cfg = False


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# 交叉注意力
def default(val, d):
    if val is not None:
        return val
    return d


def exists(val):
    return val is not None


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., block_size=60):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, block_size, block_size))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=False):
        B, T, C = x.size()
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask:
            # mask = rearrange(mask, 'b ... -> b (...)')
            # max_neg_value = -torch.finfo(sim.dtype).max
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim = sim.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # causal attention
        self.attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = None, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # img-feature crossattention
        self.img_cross_attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = config.visual_dim, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        # prompt-feature crossattention
        self.prompt_cross_attn = CrossAttention(
            query_dim = config.n_embd, 
            context_dim = config.prompt_dim, 
            heads = config.n_heads, 
            dim_head = config.dim_head, 
            dropout=config.attn_pdrop, 
            block_size=config.block_size)
        self.ln_4 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
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
        return x


class multi_gpt(PreTrainedModel):
    """ GPT Language Model """
    config_class = MyConfig
    base_model_prefix = "GPT_traj"

    def __init__(self, config):
        super().__init__(config)
        assert config.traj_dim is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.clip_path = config.clip_path
        self.n_embd = config.n_embd
        assert config.n_heads*config.dim_head == config.n_embd, f"hidden size {config.n_embd} != {config.n_heads} heads of {config.dim_head}"

        # 初始化CLIP视觉编码器和文本编码器
        self.processor = ChineseCLIPProcessor.from_pretrained(self.clip_path)
        self.model = ChineseCLIPModel.from_pretrained(self.clip_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config.traj_dim, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.traj_dim)
        self.criterion = nn.MSELoss()
        # self.class_token = nn.Parameter(torch.randn(1, 1, config.n_embd))

        self.position_head = nn.Linear(config.n_embd, config.traj_dim)
        self.position_head = nn.Linear(config.n_embd, config.traj_dim)
        
        # 与cfg训练有关的参数
        self.use_cfg = config.use_cfg
        self.img_drop_rate = config.img_drop_rate
        self.prompt_drop_rate = config.img_drop_rate
        self.cfg_img = Image.new("RGB", (336, 336), (0, 0, 0))
        self.cfg_prompt = ""


        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
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

    @classmethod
    def create_model(self, config = None):
        if config is None:
            config = MyConfig()
        model = multi_gpt(config)
        return model


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


    # 获取图像和提示词的embedding
    def get_embd(self, imgs, prompts, device):
        if imgs is not None:
            inputs_img = self.processor(images=imgs, return_tensors="pt").to(device)
            img_embd = self.vision_model(**inputs_img).last_hidden_state
        else:
            img_embd = None
        if prompts is not None:
            inputs_prompt = self.processor(text=prompts, padding=True, return_tensors="pt").to(device)
            prompt_embd = self.text_model(**inputs_prompt).last_hidden_state
        else:
            prompt_embd = None
        return img_embd, prompt_embd


    def drop_data(self, imgs, prompts):
        """
        随机丢弃图像和提示词
        """
        if self.img_drop_rate > 0:
            use_img = torch.rand(len(imgs)) > self.img_drop_rate
            imgs = [img if flag else self.cfg_img for img, flag in zip(imgs, use_img)]
        if self.prompt_drop_rate > 0:
            use_prompt = torch.rand(len(prompts)) > self.prompt_drop_rate
            prompts = [prompt if flag else self.cfg_prompt for prompt, flag in zip(prompts, use_prompt)]
        return imgs, prompts


    # forward function, train with Classifier-Free Guidance
    def forward(self, past_traj, imgs, prompts, labels=None, is_inference=False):
        """
        参数:
            - past_traj (tensor): 历史轨迹，B*t*10
            - imgs_list (List[str]): 图像路径，大小B
            - prompts (List[str]): 提示词，大小B
        
        返回值：
            - logits (tensor): 预测的下一个轨迹，B*1*10
            - loss (tensor): 损失值，B*1
        """
        b, t , nd = past_traj.size()
        assert len(imgs) == len(prompts), f"imgs and past_traj should have the same length, but got {len(imgs)} and {len(prompts)}"


        """
        random drop image(prompt) to train with Classifier-Free Guidance
        it's a test version.
        """
        
        # if not is_inference and self.use_cfg:
        #     imgs, prompts = self.drop_data(imgs, prompts)
        if not is_inference and self.use_cfg:
            imgs, prompts = self.drop_data(imgs, prompts)
        img_embd, prompt_embd = self.get_embd(imgs, prompts, past_traj.device)
        # assert t < self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=past_traj.device).unsqueeze(0) # shape (1, t)


        # forward the GPT model itself
        tok_emb = self.transformer.wte(past_traj) # token embeddings of shape (b, t, traj_dim) --> (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb+pos_emb)
        for block in self.transformer.h:
            x = block(x, img_embd, prompt_embd)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired labels also calculate the loss
        loss = None
        if labels is not None:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)), ignore_index=-1)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)))

        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


    def generate_with_cfg_batch(self, x, imgs, prompts, guidance_scale=2.0):
        """
        批量使用 Classifier-Free Guidance 生成轨迹,
        参数:
            - x (tensor): 历史轨迹，B*t*embd,包含start token
            - imgs (List[str]): 图像路径，大小B
            - prompts (List[str]): 提示词，大小B
            - guidance_scale (float): 引导强度
        返回:
            - guided_logits (tensor): 引导后的预测轨迹，B*1*10
        """
        B, t, n_embd = x.size()
        
        x = x.repeat(2, 1, 1)
        imgs_extended = imgs * 2 if imgs is not None else imgs
        prompts_extended = [self.cfg_prompt]*B  + prompts
        img_embd, prompt_embd = self.get_embd(imgs_extended, prompts_extended, x.device)
        # outputs = self(past_traj_expanded, imgs_extended, prompts_extended, labels=None, is_inference=True)
        for block in self.transformer.h:
            x = block(x, img_embd, prompt_embd)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        uncond_logits = logits[:B]
        img_cond_logits = logits[B:]
        # prompt_cond_logits = logits[2*B:]
        guided_logits = uncond_logits + guidance_scale * (img_cond_logits - uncond_logits)
        return guided_logits


    # 自回归生成轨迹，(B, f, 10) --> (B, f+1, 10) -->...--> (B, 60, 10)
    @torch.no_grad()
    def generate_traj(self, imgs, prompts, total_frames=60, guidance_scale=2.0, do_sample=False, top_k=None):
        past = None
        for fm in range(1, total_frames+1):
            pos = torch.arange(0, fm, dtype=torch.long, device=self.start_token.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            if past is None:
                now_embd = self.start_token.repeat(len(prompts), 1, 1)
                s_now_embd = now_embd + pos_emb
            else:
                past_embd = self.transformer.wte(past)
                now_embd = torch.cat((self.start_token.repeat(len(prompts), 1, 1), past_embd), dim=1)
                s_now_embd = now_embd + pos_emb

            logits = self.generate_with_cfg_batch(s_now_embd, imgs, prompts, guidance_scale)
            # prediction = logits[:, -1, :].unsqueeze(1)
            if past is None:
                past = logits
            else:
                past = torch.stack((past, logits), dim=2)
                past = past.view(past.size(0), -1, past.size(-1))
        return past



if __name__ == "__main__":
    from PIL import Image
    model = multi_gpt.create_model().cuda()
    prompts = ["一辆小车在水平地滑行"]
    imgps = ["/ossfs/workspace/aieffects/test_pic/lcar.png"]
    imgs = [Image.open(ps) for ps in imgps]
    with torch.no_grad():
        labels = torch.rand(1,4,2).cuda()
        x = torch.rand(1,4,2).cuda()
        res = model(x, imgs, prompts,labels)
    print(res)
