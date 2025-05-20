import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import PreTrainedModel
from .MyConfig import MyConfig
from .Diffusion_net import Net
from .DiT import DiT
from .utils import curve_smoothing

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def smoothness_penalty_second_order(x_0_pred):
    diff1 = x_0_pred[:, 1:, :] - x_0_pred[:, :-1, :]
    diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
    penalty = torch.mean(torch.square(diff2))
    return penalty


class GaussianDiffusionTrainer(PreTrainedModel):
    config_class = MyConfig
    def __init__(self, config):
        super().__init__(config)
        config.c_init()
        if config.net_path != "" and os.path.exists(config.net_path):
            if config.denoise_model == "net":
                self.model = Net.from_pretrained(config.net_path, config=config)
            else:
                self.model = DiT.from_pretrained(config.net_path, config=config)
        else:
            if config.denoise_model == "net":
                self.model = Net(config)
            else:
                self.model = DiT(config)
        # for param in self.model.clip.parameters():
        #     param.requires_grad = False
        self.T = config.T

        self.register_buffer(
            'betas', torch.linspace(config.beta_1, config.beta_T, config.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, past_traj, imgs, prompts, labels=None):
        """
        Algorithm 1.
        """
        x_0 = past_traj
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        noise_pred = self.model(x_t, t, imgs, prompts, labels)

        # apply filter
        noise_pred = curve_smoothing(noise_pred)
        # apply
        # x_0_pred = (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * noise_pred) / \
        #             extract(self.sqrt_alphas_bar, t, x_t.shape)
        # smooth_penalty = smoothness_penalty_second_order(x_0_pred)
        # alpha = 1e-6
        # loss = F.mse_loss(noise_pred, noise)+alpha*smooth_penalty
        loss = F.mse_loss(noise_pred, noise)
        return {"loss":loss}


class GaussianDiffusionSampler(PreTrainedModel):
    def __init__(self, model, config, w = 0.):
        super().__init__(config)
        config.c_init()
        self.model = model
        T = config.T
        self.T = config.T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w
        self.noc = ""

        self.register_buffer('betas', torch.linspace(config.beta_1, config.beta_T, config.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))


    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, img_embd, prompt_embd, start_state, nonep_embd):
        # nonep = [self.noc]*len(prompt_embd)
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model.diffusion_inference(x_t, t, img_embd, prompt_embd, start_state)
        nonEps = self.model.diffusion_inference(x_t, t, img_embd, nonep_embd, start_state)
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var


    @torch.no_grad()
    def forward(self, x_T, imgs, prompts, start_state=None, num_steps=50):
        """
        Algorithm 2.
        """
        x_t = x_T
        img_embd, prompt_embd = self.model.get_embd(imgs, prompts, x_T.device)
        _, nonep_embd = self.model.get_embd(imgs, [self.noc]*len(prompts), x_T.device)

        # steps = torch.linspace(0, self.T - 1, num_steps).long()

        for time_step in reversed(range(self.T)):
            # time_step = steps[time_step_idx]
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, img_embd=img_embd, prompt_embd=prompt_embd, start_state=start_state, nonep_embd=nonep_embd)
            if time_step > 0:
                noise = torch.randn_like(x_t)
                # noise=torch.randn([1,x_t.shape[1],x_t.shape[2]]).expand(x_t.shape).to(x_t.device)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return x_0
