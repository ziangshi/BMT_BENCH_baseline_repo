#from data import load_data
import argparse
import config_test as config
import os
import torch
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.temporal_models import HistoryNet, CondNet
from torchvision.utils import save_image
from torchvision.io import write_video
from joblib import Parallel, delayed
import os
import tarfile
import numpy as np
import torch
import numpy as np
import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as VF
from PIL import Image, ImageChops
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch
# set MASTER_ADDR environment variable
import torch.distributed as dist
#import torch.multiprocessing as mp
#from modules.denoising_diffusion import GaussianDiffusion
#from modules.unet import Unet
#from modules.trainer import Trainer
#from modules.temporal_models import HistoryNet, CondNet
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "63000"


'''
Diffusion needed functions and imports
'''
'''
util.py 
'''
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_tensor(a, t, place_holder=None):
    return a[t, torch.arange(len(t))]


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

'''
denoising_diffusion.py
'''
import torch
from torch import nn
from functools import partial
from tqdm import tqdm
from torch.distributions import Normal


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            history_fn,
            transform_fn=None,
            channels=3,
            timesteps=1000,
            # l1 loss
            loss_type="l1",
            betas=None,
            pred_mode="noise",
            clip_noise=True,
            aux_loss=True,
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.history_fn = history_fn
        self.transform_fn = transform_fn
        assert pred_mode in ["noise", "pred_true"]
        self.pred_mode = pred_mode
        self.clip_noise = clip_noise
        self.otherlogs = {}
        self.aux_loss = aux_loss

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)
        # explained unnoised element
        alphas = 1.0 - betas
        # cumulative product of unnoised element based on columns
        # diffusion process, X0= 0.99 * X1 = 0.94 = XoX1,forward
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # previous cumprod? 1.0, follows alphas cumprod[:-1], all element except the last one
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # betas.shape = (t,0)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        # Return a new partial object which when called will behave like func called with the positional
        # arguments args and keyword arguments keywords. If more arguments are supplied to the call, they are appended to args.
        # If additional keyword arguments are supplied, they extend and override keywords. Roughly equivalent t

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, context, clip_denoised: bool):
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        elif self.pred_mode == "pred_true":
            x_recon = self.denoise_fn(x, t, context=context)
        if clip_denoised:
            x_recon.clamp_(-2, 2)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, context, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, context=context, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, context):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        # res = [img]
        for count, i in enumerate(tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
        )):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,  # self.history_fn.context_time_scale(context, time),
                clip_denoised=self.clip_noise,
            )
            # if count % 100 == 0:
            #     res.append(img)
        # res.append(img)
        return img  # , res

    @torch.no_grad()
    def sample(self, init_frames, num_of_frames=3):
        video = [frame for frame in init_frames]
        # mu, res = [], []
        T, B, C, H, W = init_frames.shape
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
        for frame in video:
            context = self.history_fn(frame)
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(frame)
        for _ in range(num_of_frames):
            generated_frame = self.p_sample_loop(init_frames[0].shape, context)
            # generated_frame, res = self.p_sample_loop(init_frames[0].shape, context)
            if exists(self.transform_fn) and (
                    self.transform_fn.context_mode in ["residual"]
            ):
                # res.append(generated_frame)
                generated_frame = generated_frame * trans_shift_scale[1] + trans_shift_scale[0]
                # mu.append(trans_shift_scale[0])
                # Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively, this returns:
            context = self.history_fn(generated_frame.clamp(-1, 1))
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(generated_frame.clamp(-1, 1))
            video.append(generated_frame)
        return torch.stack(video, 0)  # , torch.stack(mu, 0), torch.stack(res, 0)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, context, t, trans_shift_scale):
        noise = torch.randn_like(x_start)
        cur_frame = x_start
        if exists(self.transform_fn):
            self.otherlogs["predict"].append(trans_shift_scale[0].detach())
            if self.transform_fn.context_mode in ["residual"]:
                x_start = (x_start - trans_shift_scale[0]) / trans_shift_scale[1]
            else:
                raise NotImplementedError

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, context=context)

        if self.pred_mode == "noise":
            if self.loss_type == "l1":
                loss = (noise - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "pred_true":
            if self.loss_type == "l1":
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def step_forward(self, x, context, t, trans_shift_scale):
        # _, _, h, w, img_size = *x.shape, self.image_size
        # assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        return self.p_losses(x, context, t, trans_shift_scale)

    def scan_context(self, x):
        # residual context model - > condnet-> either residual or actual output of image for denoising process
        context = self.history_fn(x)
        # learned mu and sigma
        trans_shift_scale = self.transform_fn(x) if exists(self.transform_fn) else None
        return context, trans_shift_scale

    def forward(self, video):
        device = video.device
        T, B, C, H, W = video.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        loss = 0
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
            self.otherlogs["predict"] = []

        for i in range(video.shape[0]):
            if i >= 2:
                # updating loss after 1 frame
                L = self.step_forward(video[i], context, t, trans_shift_scale)
                loss += L
            if i < video.shape[0] - 1:
                # get residual and (mu,sigma)
                # updating frame and mu+sigma every frame
                context, trans_shift_scale = self.scan_context(video[i])

        if exists(self.transform_fn):
            self.otherlogs["predict"] = torch.stack(self.otherlogs["predict"], 0)
        return loss / (video.shape[0] - 2)

'''
network_components.py
'''
import torch.nn as nn
import math
from einops import rearrange


def get_backbone(name, params):
    if name == "convnext":
        return ConvNextBlock(*params)
    elif name == "resnet":
        return ResnetBlock(*params)
    else:
        raise NotImplementedError


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, activation='leakyrelu'):
        super().__init__()
        assert activation in ['leakyrelu', 'tanh']
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out),
            nn.LeakyReLU(0.2) if activation == 'leakyrelu' else nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 1),
            nn.GELU(),
            LayerNorm(dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, n_layer=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.cur_states = [None for i in range(n_layer)]
        self.n_layer = n_layer

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
            ]
            + [
                nn.Conv2d(
                    in_channels=self.hidden_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
                for i in range(n_layer - 1)
            ]
        )

    def step_forward(self, input_tensor, layer_index=0):
        assert self.cur_states[layer_index] is not None
        h_cur, c_cur = self.cur_states[layer_index]
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.convs[layer_index](combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        self.cur_states[layer_index] = (h_next, c_next)

        return h_next

    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor

    def init_hidden(self, batch_shape):
        B, _, H, W = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = (
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device, ),
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device, ),
            )


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,  # for candidate neural memory
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=self.conv_cans[0].weight.device)

    def step_forward(self, input_tensor, index):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next

    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor


'''
temporal_models.py
'''
from numpy import identity
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math
'''
from .network_components import (
    LayerNorm,
    ResnetBlock,
    Upsample,
    Downsample,
    ConvGRUCell,
    Block,
    Residual,
    PreNorm,
    LinearAttention,
    SinusoidalPosEmb,
    get_backbone,
)
'''
class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp().clamp(math.sqrt(2), 20 * math.sqrt(2))


class SimpleHistoryNet(nn.Module):
    def __init__(
            self,
            dim,  # must be the same as main net
            dim_mults=(1, 2, 3, 4),
            channels=3,
            context_mode="residual",
            backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        assert context_mode in ["residual"]
        self.context_mode = context_mode
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mu = nn.Conv2d(dim, channels, 3, 1, 1)
        self.sigma = (
            nn.Sequential(nn.Conv2d(dim, channels, 3, 1, 1), Exp())
            if context_mode in ["transform"]
            else None
        )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid = ConvGRUCell(dim_out, dim_out, 3, n_layer=2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out, dim_in)),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        temp_shape = list(shape)
        temp_shape[-2] //= 2 ** (len(self.dim_mults) - 1)
        temp_shape[-1] //= 2 ** (len(self.dim_mults) - 1)
        self.mid.init_hidden(temp_shape)

    def forward(self, x):
        for idx, (resnet, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = downsample(x)
        x = self.mid(x)
        for idx, (resnet, upsample) in enumerate(self.ups):
            x = resnet(x)
            x = upsample(x)
        mu = self.mu(x)
        sigma = self.sigma(x) if self.context_mode in ["transform"] else torch.ones_like(mu)
        return (mu.clamp(-1, 1), sigma)


class HistoryNet(nn.Module):
    def __init__(
            self,
            dim,  # must be the same as main net
            dim_mults=(1, 2, 3, 4),
            channels=3,
            context_mode="residual",
            backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        assert context_mode in ["residual"]
        self.context_mode = context_mode
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mu = nn.Conv2d(dim, channels, 3, 1, 1)
        self.sigma = (
            nn.Sequential(nn.Conv2d(dim, channels, 3, 1, 1), Exp())
            if context_mode in ["transform"]
            else None
        )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        ConvGRUCell(dim_out, dim_out, 3, n_layer=1),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # self.mid = get_backbone(backbone, (dim_in[-1], dim_in[-1]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out if ind == 0 else dim_out * 2, dim_in)),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        input_frame = x

        h = []
        for idx, (resnet, gru, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = gru(x)
            if idx != (len(self.downs) - 1):
                h.append(x)
            x = downsample(x)
        # x = self.mid(x)
        for idx, (resnet, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1) if idx != 0 else x
            x = resnet(x)
            x = upsample(x)
        mu = self.mu(x)
        sigma = self.sigma(x) if self.context_mode in ["transform"] else torch.ones_like(mu)
        return (mu.clamp(-1, 1), sigma)


class CondNet(nn.Module):
    def __init__(
            self,
            dim,  # must be the same as main net
            dim_mults=(1, 1, 2, 2, 4, 4),  # must be the same as main net
            channels=3,
            backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        ConvGRUCell(dim_out, dim_out, 3, n_layer=1),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = conv(x)
            context.append(x)
            x = downsample(x)
        return context

'''
trainer.py
'''
import copy
import torch
import os
import shutil
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR


from inspect import isfunction
import numpy as np


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_tensor(a, t, place_holder=None):
    return a[t, torch.arange(len(t))]


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        diffusion_model,
        train_dl,
        val_dl,
        sample_num_of_frame,
        init_num_of_frame,
        scheduler_function, # cosine function
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=2,
        optimizer="adam",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sample_num_of_frame = sample_num_of_frame
        self.val_num_of_batch = val_num_of_batch

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        if optimizer == "adam":
            self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.init_num_of_frame = init_num_of_frame
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        if self.device == 0:
            data = {
                "step": self.step,
                "model": self.model.module.state_dict(),
                "ema": self.ema_model.module.state_dict(),
            }
            idx = (self.step // self.save_and_sample_every) % 3
            torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        self.model.module.load_state_dict(data["model"])
        self.ema_model.module.load_state_dict(data["ema"])

    def train(self):

        while self.step < self.train_num_steps:
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            print(self.step)
            data = next(self.train_dl).to(self.device)
            loss = self.model(data * 2.0 - 1.0)
            loss.backward()
            if self.device == 0:
                self.writer.add_scalar("sequence_length", data.shape[0], self.step)
                self.writer.add_scalar("loss", loss.item(), self.step)
            dist.barrier()

            self.opt.step()
            self.opt.zero_grad()

            if (self.update_ema_every > 0) and (self.step % self.update_ema_every == 0):
                self.step_ema()

            if (self.step % self.save_and_sample_every == 0) and (self.step != 0):
                # milestone = self.step // self.save_and_sample_every
                if exists(self.model.module.transform_fn) and len(self.model.module.otherlogs["predict"]) > 0:
                    self.writer.add_video(
                        f"predicted/device{self.device}",
                        (self.model.module.otherlogs["predict"].transpose(0, 1) + 1) * 0.5,
                        self.step // self.save_and_sample_every,
                    )
                for i, batch in enumerate(self.val_dl):
                    if i >= self.val_num_of_batch:
                        break
                    videos = self.ema_model.module.sample(
                        batch[: self.init_num_of_frame].to(self.device) * 2.0 - 1.0,
                        self.sample_num_of_frame,
                    )
                    videos = (videos + 1.0) * 0.5
                    self.writer.add_video(
                        f"samples_device/device{self.device}/num{i}",
                        videos.clamp(0.0, 1.0).transpose(0, 1),
                        self.step // self.save_and_sample_every,
                    )
                    self.writer.add_video(
                        f"true_frames/device{self.device}/num{i}",
                        batch.transpose(0, 1),
                        self.step // self.save_and_sample_every,
                    )
                if self.device == 0:
                    self.save()
                dist.barrier()

            self.step += 1
        if self.device == 0:
            self.save()
        print("training completed")

'''
unet.py
'''
import torch
from torch import nn
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        context_dim_factor=1,
        dim_mults=(1, 1, 2, 2, 4, 4),
        channels=3,
        with_time_emb=True,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out * 2, dim_in, time_dim)),
                        get_backbone(backbone, (dim_in, dim_in, time_dim)),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def encode(self, x, t, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        return x, h

    def decode(self, x, h, t):
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

    def forward(self, x, time=None, context=None):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)


'''
transform
'''
class Resize(object):
    """
    Resizes a PIL image or sequence of PIL images.
    img_size can be an int, list or tuple (width, height)
    """

    def __init__(self, img_size):
        if type(img_size) != tuple and type(img_size) != list:
            img_size = (img_size, img_size)
        self.img_size = img_size

    def __call__(self, input):
        if type(input) == list:
            return [im.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR) for im in input]
        return input.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)

class RandomSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, input):
        if type(input) == list:
            input_seq_len = len(input)
        elif "shape" in dir(input):
            input_seq_len = input.shape[0]
        max_start_ind = input_seq_len - self.seq_len + 1
        assert max_start_ind > 0, (
            "Sequence length longer than input sequence length: " + str(input_seq_len) + "."
        )
        # start_ind = np.random.choice(range(max_start_ind))
        start_ind = torch.randint(0, max_start_ind, (1,)).item()
        return input[start_ind : start_ind + self.seq_len]

class ImageToTensor(object):
    """
    Converts a PIL image or sequence of PIL images into (a) PyTorch tensor(s).
    """

    def __init__(self):
        self.to_tensor = torch_transforms.ToTensor()

    def __call__(self, input):
        if type(input) == list:
            return [self.to_tensor(i) for i in input]
        return self.to_tensor(input)

class ConcatSequence(object):
    """
    Concatenates a sequence (list of tensors) along a new axis.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        return torch.stack(input)

class FixedSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len, start_index=0):
        self.seq_len = seq_len
        self.start_index = start_index

    def __call__(self, input):
        return input[self.start_index : self.start_index + self.seq_len]
Compose = torch_transforms.Compose

'''
similar to big dataset
'''

class BIG(Dataset):
    """
    Dataset object for BAIR robot pushing dataset. The dataset must be stored
    with each video in a separate directory:
        /path
            /0
                /0.png
                /1.png
                /...
            /1
                /...
    """

    def __init__(self, path, transform=None, add_noise=False, img_mode=False):
        assert os.path.exists(
            path), 'Invalid path to UCF+HMDB data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)
        self.img_mode = img_mode
        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(
            self.path, self.video_list[ind]))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        img_names.sort(key=float)
        if not self.img_mode:
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], i + '.png')) for i in img_names]
        else:
            select = torch.randint(0, len(img_names), (1,))
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], img_names[select] + '.png'))]
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)

        # imgs = imgs.unsqueeze(1)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs)-0.5) / 256.

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
'''
load dataset 
'''
def load_dataset(data_config):
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    if dataset_name == 'bat':
        #from .datasets import BIG
        train_transforms = []

        transforms = [
            Resize(data_config["img_size"]),
            RandomSequenceCrop(data_config["sequence_length"]),
            ImageToTensor(),
            ConcatSequence(),
        ]
        val_transforms = [
            Resize(data_config["img_size"]),
            FixedSequenceCrop(data_config["sequence_length"]),
            ImageToTensor(),
            ConcatSequence(),
        ]
        train_trans = Compose(train_transforms + transforms)
        test_trans = Compose(val_transforms)
        train_path = "processed_video/video_train"
        val_path = "processed_video/video_val"
        train = BIG(
            os.path.join(train_path),
            train_trans,
            data_config["add_noise"],
        )
        val = BIG(
            os.path.join(val_path),
            test_trans,
            data_config["add_noise"],
        )
    else:
        raise Exception("Dataset name not found.")
    return train, val
'''
load data
'''
'''
requires transposed_collate
'''
from torch.utils.data.dataloader import default_collate
def train_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    # assert transposed_batch.shape[0] >= 4
    # idx = torch.randint(4, transposed_batch.shape[0] + 1, size=(1,)).item()
    # return transposed_batch[:idx]
    return transposed_batch


def test_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    return transposed_batch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_data(data_config, batch_size, num_workers=4, pin_memory=True, sequence=True, distributed=False):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a DataLoader.

    Args:
        data_config (dict): data configuration dictionary
        batch_size (dict): run configuration dictionary
        num_workers (int): number of threads of multi-processed data Loading
        pin_memory (bool): whether or not to pin memory in cpu
        sequence (bool): whether data examples are sequences, in which case the
                         data loader returns transposed batches with the sequence
                         step as the first dimension and batch index as the
                         second dimension
    """
    '''
    if data_config["img_size"] == 64:
    embed_dim = 48
    transform_dim_mults = (1, 2, 2, 4)
    dim_mults = (1, 2, 4, 8)
    batch_size = 2
elif data_config["img_size"] in [128, 256]:
    embed_dim = 64
    transform_dim_mults = (1, 2, 3, 4)
    dim_mults = (1, 1, 2, 2, 4, 4)
    batch_size = 1
else:
    raise NotImplementedErrorc
    '''
    train, val = load_dataset(data_config)
    train_spl = DistributedSampler(train) if distributed else None
    val_spl = DistributedSampler(val, shuffle=False) if distributed else None

    if train is not None:
        train = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False if distributed else True,
            collate_fn=train_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_spl
        )

    if val is not None:
        val = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_spl
        )
    return train, val





def get_dim(data_config):
    return 48 if data_config["img_size"] == 64 else 64


def get_transform_mults(data_config):
    return (1, 2, 3, 4) if data_config["img_size"] in [128, 256] else (1, 2, 2, 4)


def get_main_mults(data_config):
    return (1, 1, 2, 2, 4, 4) if data_config["img_size"] in [128, 256] else (1, 2, 4, 8)

#def ddp_setup(rank:int, world_size:int):
#    os.environ["MASTER_ADDR"] = "localhost"
#    os.environ["MASTER_PORT"] = "63000"
#    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int,
                        help='gpu rank')
    args = parser.parse_args()
    return vars(args)

def main(config_out):

    #ddp_setup(rank=rank, world_size=world_size)
    rank = config_out['rank']
    train_data, val_data = load_data(
        config.data_configs, config.BATCH_SIZE, pin_memory=False, num_workers=0, distributed=False)

    model_name = f"{config.backbone}-{config.optimizer}-{config.pred_modes}-l1-{config.data_configs['dataset_name']}-d{get_dim(config.data_configs)}-t{config.iteration_step}-{config.transform_modes}-al{config.aux_loss}{config.additional_note}"
    results_folder = os.path.join(config.result_root, f"{model_name}")
    loaded_param = torch.load(str(f"{results_folder}/{model_name}_{1}.pt"),
                    map_location=lambda storage, loc: storage,)

    denoise_model = Unet(
                    dim=get_dim(config.data_configs),
                    context_dim_factor=config.context_dim_factor,
                    channels=config.data_configs["img_channel"],
                    dim_mults=get_main_mults(config.data_configs),
                )

    context_model = CondNet(
                    dim=int(config.context_dim_factor * get_dim(config.data_configs)),
                    channels=config.data_configs["img_channel"],
                    backbone=config.backbone,
                    dim_mults=get_main_mults(config.data_configs),
                )

    transform_model = (
                    HistoryNet(
                        dim=int(config.transform_dim_factor * get_dim(config.data_configs)),
                        channels=config.data_configs["img_channel"],
                        context_mode=config.transform_modes,
                        backbone=config.backbone,
                        dim_mults=get_transform_mults(config.data_configs),
                    )
                    if config.transform_modes in ["residual", "transform", "flow", "ll_transform"]
                    else None
                )

    model = GaussianDiffusion(
                    denoise_fn=denoise_model,
                    history_fn=context_model,
                    transform_fn=transform_model,
                    pred_mode=config.pred_modes,
                    clip_noise=config.clip_noise,
                    timesteps=config.iteration_step,
                    loss_type=config.loss_type,
                    aux_loss=config.aux_loss,
                )



    model.load_state_dict(loaded_param["model"])
    print("loaded!")
    model.eval()
    N_SAMPLED = 16
                #model.to(args.device)
    for k, batch in enumerate(val_data):

        if k >= config.N_BATCH:
            break
        for i, b in enumerate(
            batch[config.N_CONTEXT : config.N_CONTEXT + N_SAMPLED].transpose(0, 1)
        ):
            if not os.path.isdir(f"evaluate/truth/{model_name}"):
                os.mkdir(f"evaluate/truth/{model_name}")
            Parallel(n_jobs=1)(
                delayed(save_image)(f, f"evaluate/truth/{model_name}/{k}-{i}-{j}.png")
                for j, f in enumerate(b.cpu())
            )
            write_video(
                 f"evaluate/truth/{model_name}/{k}-{i}.mp4",
                 torch.round(255 * b.permute(0, 2, 3, 1)).expand(-1,-1,-1,3),
                 fps=4,
            )
        batch = (batch - 0.5) * 2.0
        #batch = batch.to(args.device)
        sampled = model.sample(
            init_frames=batch[: config.N_CONTEXT], num_of_frames=N_SAMPLED
        ).transpose(
            0, 1
        )  # N T C H W
        sampled = (sampled + 1.0) / 2.0
        for i, b in enumerate(sampled.clamp(0, 1)):
            if not os.path.isdir(f"evaluate/generated/{model_name}"):
                os.mkdir(f"evaluate/generated/{model_name}")
            Parallel(n_jobs=1)(
                delayed(save_image)(f, f"evaluate/generated/{model_name}/{k}-{i}-{j}.png")
                for j, f in enumerate(b.cpu())
            )
            write_video(
                 f"evaluate/generated/{model_name}/{k}-{i}.mp4",
                 torch.round(255 * b.permute(0, 2, 3, 1)).expand(-1,-1,-1,3).cpu(),
                 fps=4,
            )

if __name__ == "__main__":
    config_out = parse_args()
    main(config_out)