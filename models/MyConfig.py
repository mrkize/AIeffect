from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    def __init__(self, 
                 n_layer=6, 
                 n_heads=8, 
                 n_embd=512, 
                 dim_head=64,
                 visual_dim=768,
                 prompt_dim=768,
                 head_act=False,
                 traj_dim=10,
                 block_size=100,
                 valid_idx=None,
                 denoise_model="DiT",
                 cond_type = "prompt",
                 embd_pdrop=0.1,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 clip_path="OFA-Sys/chinese-clip-vit-base-patch16",
                 img_drop_rate=0,
                 prompt_drop_rate=0,
                 use_cfg=False,
                 beta_1=1e-4,
                 beta_T=0.028,
                 T=1000,
                 PE="learned",
                 **kwargs):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dim_head = dim_head
        self.visual_dim = visual_dim
        self.prompt_dim = prompt_dim
        self.cond_dim = prompt_dim
        self.head_act = head_act
        self.PE = PE

        self.traj_dim = traj_dim
        self.block_size = block_size
        self.valid_idx = valid_idx if valid_idx is not None else []
        self.denoise_model = denoise_model
        self.net_path = ""
        self.cond_type = cond_type


        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.clip_path = clip_path
        self.img_drop_rate = img_drop_rate
        self.prompt_drop_rate = prompt_drop_rate
        self.use_cfg = use_cfg

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.c_init()

    def c_init(self):
        if self.cond_type == "prompt":
            self.cond_dim = self.prompt_dim
        elif self.cond_type == "img":
            self.cond_dim = self.visual_dim
        else:
            self.cond_dim = self.prompt_dim + int(self.n_embd/2)

        print("PE use ", self.PE)
