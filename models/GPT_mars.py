import math
import torch
import torch.nn as nn
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, PreTrainedModel
from PIL import Image
from .utils import CrossAttention
from .MyConfig import MyConfig


# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


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
        x = x + self.attn(self.ln_1(x), mask=True)
        # if img_embd is not None:
        #     x = x + self.img_cross_attn(self.ln_2(x), img_embd)
        # if prompt_embd is not None:
        #     x = x + self.prompt_cross_attn(self.ln_3(x), prompt_embd)
        # x = x + self.mlpf(self.ln_4(x))
        return x

class GPT_mars(PreTrainedModel):
    """ GPT Language Model """
    config_class = MyConfig
    base_model_prefix = "GPT_mars"

    def __init__(self, config):
        super().__init__(config)
        assert config.traj_dim is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.clip_path = config.clip_path
        self.cond_type = config.cond_type

        assert config.n_heads*config.dim_head == config.n_embd, f"hidden size {config.n_embd} != {config.n_heads} heads of {config.dim_head}"


        self.transformer = nn.ModuleDict(dict(
            curve_embedding = nn.Linear(config.traj_dim, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.head = nn.Linear(config.n_embd, config.traj_dim)
        self.criterion = nn.MSELoss()
        self.start_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        
        
        # 与cfg训练有关的参数
        self.use_cfg = config.use_cfg
        self.img_drop_rate = config.img_drop_rate
        self.prompt_drop_rate = config.prompt_drop_rate
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
        
        # 初始化CLIP视觉编码器和文本编码器
        self.token_embd = nn.Linear(config.n_embd, config.n_embd)
        self.processor = ChineseCLIPProcessor.from_pretrained(self.clip_path)
        self.clip = ChineseCLIPModel.from_pretrained(self.clip_path)
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
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    # 获取图像和提示词的embedding
    @torch.no_grad()
    def get_embd(self, imgs, prompts, device):
        self.clip.eval()
        img_embd = None
        prompt_embd = None
        if prompts is not None:
            inputs_prompt = self.processor(text=prompts, padding=True, return_tensors="pt").to(device)
            prompt_embd = self.clip.text_model(**inputs_prompt).last_hidden_state
        if imgs is not None:
            inputs_img = self.processor(images=imgs, return_tensors="pt").to(device)
            img_embd = self.clip.vision_model(**inputs_img).last_hidden_state
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
                imgs = self.replace_with_cfg_efficient(imgs, self.cfg_img, self.img_drop_rate)
        elif self.config.cond_type == "prompt":
            imgs = None
            if self.prompt_drop_rate > 0 and prompts is not None:
                prompts = self.replace_with_empty_strings_efficient(prompts, self.prompt_drop_rate)
        else:
            imgs = None
            prompts = self.replace_with_empty_strings_efficient(prompts, self.prompt_drop_rate)
            # start_state[replacement_mask] = self.default_start_state.expand_as(start_state)[replacement_mask]
        return imgs, prompts
    
    
    # forward function, train with Classifier-Free Guidance
    def forward(self, past_traj, imgs, prompts, labels=None):
        """
        参数:
            - past_traj (tensor): 历史轨迹，B*t*10
            - imgs_list (List[str]): 图像路径，大小B
            - prompts (List[str]): 提示词，大小B
        
        返回值：
            - logits (tensor): 预测的下一个轨迹，B*1*10
            - loss (tensor): 损失值，B*1
        """
        b, t , dim = past_traj.size()

        if self.training and self.use_cfg:
            imgs, prompts  = self.drop_data(imgs, prompts)
        img_embd, prompt_embd = self.get_embd(imgs, prompts, x.device)


        x = self.transformer.curve_embedding(past_traj)
        x = torch.cat((self.token_embd(prompt_embd).unsqueeze(1), x), dim=1)
        pos = torch.arange(0, t+1, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x + pos_emb)



        assert t < self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        for block in self.transformer.h:
            x = block(x, img_embd, prompt_embd)
        x = self.transformer.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
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
        logits = self.head(x)
        uncond_logits = logits[:B]
        cond_logits = logits[B:]
        # prompt_cond_logits = logits[2*B:]
        guided_logits = cond_logits + guidance_scale * (cond_logits - uncond_logits)
        return guided_logits


    # 自回归生成轨迹，(B, f, 10) --> (B, f+1, 10) -->...--> (B, 60, 10)
    @torch.no_grad()
    def generate_traj(self, imgs, prompts, total_frames=60, guidance_scale=0.5):
        past = None
        assert total_frames > 0, "total_frames must be greater than 0"
        pos = torch.arange(0, total_frames+1, dtype=torch.long, device=self.start_token.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        for fm in range(1, total_frames+1):
            if past is None:
                now_embd = self.start_token.repeat(len(prompts), 1, 1)
                s_now_embd = now_embd + pos_emb[:,:fm,:]
            else:
                past_embd = self.transformer.curve_embedding(past)
                now_embd = torch.cat((self.start_token.repeat(len(prompts), 1, 1), past_embd), dim=1)
                s_now_embd = now_embd + pos_emb[:,:fm,:]

            logits = self.generate_with_cfg_batch(s_now_embd, imgs, prompts, guidance_scale)
            prediction = logits[:, -1, :].unsqueeze(1)
            past = prediction if past is None else torch.cat((past, prediction), dim=1)
        return past



if __name__ == "__main__":
    from PIL import Image
    config=MyConfig()
    model = GPT_mars(config).cuda()
    prompts = ["调大", "大赛"]
    # imgps = ["mingpt.jpg", "10jifen.png"]
    # imgs = [Image.open(ps) for ps in imgps]
    past = torch.rand((2,1,10)).cuda()
    print(past)
    with torch.no_grad():
        res = model.generate_traj(None, prompts, total_frames = 5)
    print(res.shape)
