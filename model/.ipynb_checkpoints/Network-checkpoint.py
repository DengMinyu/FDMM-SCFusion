import time
import math
from functools import partial
from typing import Optional, Callable
import pywt
import torch.fft
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat, einsum
import numpy as np
from pdb import set_trace as stx
import numbers
from thop import profile, clever_format
from einops import rearrange
# from .mamba import Mamba
from .ConvSSM import ConvSSM
from torchvision import models
from timm.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
# from mmseg.utils import get_root_logger
import math
import pywt
from torch.autograd import Function


class CD_DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class CD_Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 内容感知编码
        # self.content_encoder=nn.Sequential(
        #     nn.AdaptiveAvgPool2d(18),
        #     nn.Conv2d(512, 512, (1, 1)),
        #     # F.interpolate(pos_c, mode='bilinear',size= style.shape[-2:])
        # )
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = CD_DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv = CD_DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride,
                                        bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        shapex = x.shape

        bx, nx, _x, wx, hx = *shapex, self.heads
        # x = x.permute(0, 2, 3, 1).contiguous()
        # x=self.ConvSSM(x)
        # out_x = x.permute(0, 3, 1, 2).contiguous()
        qx = self.to_q(x)

        qx = rearrange(qx, 'b (h d) x y -> (b h) (x y) d', h=hx)
        shapey = y.shape
        by, ny, _y, wy, hy = *shapey, self.heads
        ky, vy = self.to_kv(y).chunk(2, dim=1)

        ky, vy = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=hy), (ky, vy))
        dotsx = torch.einsum('b i d, b j d -> b i j', qx, ky) * self.scale
        attnx = self.attend(dotsx)
        attnx = self.dropout(attnx)
        outx = torch.einsum('b i j, b j d -> b i d', attnx, vy)
        outx = rearrange(outx, '(b h) (x y) d -> b (h d) x y', h=hx, y=wx)

        # dotsy = einsum('b i d, b j d -> b i j', qy, kx) * self.scale
        # attny = self.attend(dotsy)
        # attny = self.dropout(attny)
        # outy = einsum('b i j, b j d -> b i d', attny, vx)
        # outy = rearrange(outy, '(b h) (x y) d -> b (h d) x y', h=hy, y=wy)

        # out = torch.cat([outx, outy], dim=1)

        return self.to_out(outx), y + self.to_out(outx)


class CD_LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class CD_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CD_LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        x = self.norm(x)
        y = self.norm(y)
        return self.fn(x, y, **kwargs)


class CD_PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CD_LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)

        return self.fn(x, **kwargs)


class CD_FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CD_Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.Norm = CD_LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CD_PreNorm(dim, CD_Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                             dim_head=dim_head, dropout=dropout)),
                CD_PreNorm2(dim, CD_FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x, y):
        for attn, ff in self.layers:
            x1, add = attn(x, y)
            # x2 = x1 + x
            # y2 = y1 + y

            add1 = ff(add)
            out = add1 + add
            # y4 = y3 + y2
            out = self.Norm(out)
        return out


class CD_conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print('\nCUDA is available. Calculation is performing on ' + str(
            torch.cuda.get_device_name(torch.cuda.current_device())) + '.\n')
    else:
        DEVICE = 'cpu'
        print('\nOOPS! CUDA is not available! Calculation is performing on CPU.\n')


DEVICE = GPUorCPU.DEVICE


class SS_Block(nn.Module):
    def __init__(self):
        super(SS_Block, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        coeffs2 = pywt.dwt2(x.cpu().detach().numpy(), 'haar', mode='zero')  # 二维离散小波变换
        cA, (cH, cV, cD) = coeffs2  # cA:低频部分，cH:水平高频部分，cV:垂直高频部分，cD:对角线高频部分
        return cA, (cH, cV, cD)


class MSSE_Attention(nn.Module):
    def __init__(self, channels):
        super(MSSE_Attention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.mish(self.bn(out))
        out = torch.mean(out, dim=(2, 3))
        out = self.fc(out)
        out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)
        out = out * x
        return out


class ParallelBlock(nn.Module):
    def __init__(self, channels):
        super(ParallelBlock, self).__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        # self.layer = nn.Conv2d(channels*3, channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out1 = F.mish(self.branch1(x))
        out2 = F.mish(self.branch2(x))
        out3 = F.mish(self.branch3(x))
        out = torch.cat((out1, out2, out3), dim=1)
        # out = self.layer(out)
        return out


class LaplacianEnhance(nn.Module):
    def __init__(self):
        super(LaplacianEnhance, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        lap = F.conv2d(x, self.kernel.expand(c, 1, 3, 3), padding=1, groups=c)
        return x + lap  # alpha 默认 1，可作为参数加入


class MSSE_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSSE_Module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )
        self.attention = MSSE_Attention(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.parallel = ParallelBlock(out_channels)
        self.skip_connection = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)

        self.channel_splitter = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        conn_x = out
        out = self.attention(out)
        out = out + residual
        out = self.parallel(out)
        out = out + self.skip_connection(x)
        out = self.channel_splitter(out)

        return out


class FSE_Module(nn.Module):
    def __init__(self, in_channels):
        super(FSE_Module, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
        )
        self.SS_Block = SS_Block()
        self.high_layer = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        x = x + residual
        cA, (cH, cV, cD) = self.SS_Block(x)
        cA_tensor = torch.from_numpy(cA).to(x.device)
        cH_tensor = torch.from_numpy(cH).to(x.device)
        cV_tensor = torch.from_numpy(cV).to(x.device)
        cD_tensor = torch.from_numpy(cD).to(x.device)
        x_low = cA_tensor
        x_high = torch.cat([cH_tensor, cV_tensor, cD_tensor], dim=1)
        x_high = self.high_layer(x_high)
        return x_low, x_high


class LayerNorm1(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))  # 创建可供网络更新的张量
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm1(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, scale_factor, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      dilation=scale_factor, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, dim_head, scale_factor=None, dropout=0.):
        super().__init__()
        if scale_factor is None:
            scale_factor = [1, 2, 4]
        self.num_group = len(scale_factor)
        inner_dim = dim_head * self.num_group
        self.heads = self.num_group
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.Multi_scale_Token_Embeding = nn.ModuleList([])
        for i in range(len(scale_factor)):
            self.Multi_scale_Token_Embeding.append(nn.ModuleList([
                DepthWiseConv2d(dim, dim_head, proj_kernel, padding=scale_factor[i], stride=1,
                                scale_factor=scale_factor[i], bias=False),
                DepthWiseConv2d(dim, dim_head * 2, proj_kernel, padding=scale_factor[i], stride=kv_proj_stride,
                                scale_factor=scale_factor[i], bias=False),
            ]))

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, d, h, w = x.shape
        Q, K, V = [], [], []
        for to_q, to_kv in self.Multi_scale_Token_Embeding:
            q = to_q(x)
            k, v = to_kv(x).chunk(2, dim=1)
            q, k, v = map(lambda t: rearrange(t, 'b d x y -> b (x y) d'), (q, k, v))
            Q.append(q)
            K.append(k)
            V.append(v)
        random.shuffle(Q)
        Q = torch.cat([Q[0], Q[1], Q[2]], dim=0)
        K = torch.cat([K[0], K[1], K[2]], dim=0)
        V = torch.cat([V[0], V[1], V[2]], dim=0)
        dots = einsum('b i d, b j d -> b i j', Q, K) * self.scale  # C = np.einsum('ij,jk->ik', A, B)保留想要的维度
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, V)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=self.num_group, x=h, y=w)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, dim_head, scale_factor, mlp_mult=4, dropout=0.):
        # dim=64, proj_kernel=3, kv_proj_stride=1, depth=2, scale_factor=[1, 2, 4], mlp_mult=8, dim_head=64, dropout=dropout
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                       dim_head=dim_head, scale_factor=scale_factor, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # PreNorm(1)
            x = ff(x) + x  # PreNorm(2)
        return x


class conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm1(dim_out),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv3x3 = conv3x3(input_dim=dim_in, output_dim=dim_out)

    def forward(self, x, h, w):
        x = F.interpolate(x, mode='bilinear', size=(int(h), int(w)))
        x = self.conv3x3(x)
        return x


#######################################################################
# transformer
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        w_ll = w_ll.to(dtype=x.dtype, device=x.device)
        w_lh = w_lh.to(dtype=x.dtype, device=x.device)
        w_hl = w_hl.to(dtype=x.dtype, device=x.device)
        w_hh = w_hh.to(dtype=x.dtype, device=x.device)

        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            filters = filters.to(dtype=dx.dtype, device=dx.device)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        filters = filters.to(dtype=x.dtype, device=x.device)
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            filters = filters.to(dtype=dx.dtype, device=dx.device)

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float16)
        self.w_lh = self.w_lh.to(dtype=torch.float16)
        self.w_hl = self.w_hl.to(dtype=torch.float16)
        self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


def frozen_batch_norm(dim):
    """构建一个不参与训练的 BatchNorm2d 层"""
    bn = nn.BatchNorm2d(dim)
    for p in bn.parameters():
        p.requires_grad = False
    return bn


class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.shift_size = window_size // 2
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        # self.dwt = DWT_2D(wave='haar')
        # self.idwt = IDWT_2D(wave='haar')

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            frozen_batch_norm(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            frozen_batch_norm(dim),
            nn.ReLU(inplace=True),
        )

        self.qkv = nn.Linear(dim, 3*dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim + dim // 4, dim)

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_win_H, num_win_W, win_h, win_w, C)
        windows = x.view(-1, window_size * window_size, C)  # flatten windows
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """ Reverse windows to original layout """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x

    def forward(self, x, H, W,mask=None):
        B, N, C = x.shape
        assert N == H * W, "Input token count doesn't match H x W"

        # Q: B, N, C -> B, H, W, C
        x_img = x.view(B, H, W, C)
        q = self.qkv(x).view(B, H, W, 3*C).permute(0, 3, 1, 2)

        # Save original size
        H_ori, W_ori = H, W

        # Compute padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        H_pad, W_pad = H + pad_h, W + pad_w
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))

        if mask != None:
            shifted_q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_q = q

        # Partition Q to windows
        q_windows = self.window_partition(shifted_q, self.window_size)  # (B*num_windows, Ws*Ws, C)
        B_ = q_windows.shape[0]
        q_windows = q_windows.view(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3,1, 4) # B*nwin, heads, tokens, dim
        # KV from wavelet
        q , k , v = q_windows[0], q_windows[1],q_windows[2]  # (B*num_windows, heads, tokens, dim)

        # Attention in each window
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, 49, 49) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, 49, 49)


        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)

        # Merge windows
        x_out = self.window_reverse(out, self.window_size, H_pad, W_pad).reshape(B, H_pad * W_pad, C)
        if mask != None:
            x_out = torch.roll(to_4d(x_out, H_pad, W_pad), shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            x_out  = to_3d(x_out)
        x_out = x_out[:, :N, :]  # remove padding tokens

        return x_out

class LaplacianEnhance(nn.Module):
    def __init__(self):
        super(LaplacianEnhance, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        lap = F.conv2d(x, self.kernel.expand(c, 1, 3, 3), padding=1, groups=c)
        return x + lap  # alpha 默认 1，可作为参数加入

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class DifferentiableSE(Function):
    """可微分结构元素生成"""

    @staticmethod
    def forward(ctx, size, device):
        # 生成正方形结构元素（可微分版本，用平滑矩阵近似）
        se = torch.ones((size, size), device=device)
        ctx.save_for_backward(se)
        return se

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时梯度全为1
        se, = ctx.saved_tensors
        return None, None


class DifferentiableDilation(Function):
    """可微分膨胀操作"""

    @staticmethod
    def forward(ctx, input, se):
        # 用卷积近似膨胀（padding=same）
        b, c, h, w = input.shape
        k = se.shape[0]
        padding = k // 2
        # 将结构元素转换为卷积核
        kernel = se.view(1, 1, k, k).repeat(c, 1, 1, 1)
        # 最大池化近似膨胀
        output = F.max_pool2d(input, kernel_size=k, stride=1, padding=padding)
        ctx.save_for_backward(input, se, kernel)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 手动定义反向传播（简化处理，实际需更复杂梯度计算）
        input, se, kernel = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # 此处为简化示例，实际需实现梯度反传
            grad_input = grad_output.clone()
        return grad_input, None


class MSMGLayer(nn.Module):
    """可微分MSMG层"""

    def __init__(self, scales=3, size=3):
        super().__init__()
        self.scales = scales
        self.size = size
        self.weights = nn.Parameter(
            torch.tensor([1.0 / (2 * (t + 1) + 1) for t in range(scales)],
                         dtype=torch.float32),
            requires_grad=True  # 权重可训练
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        device = x.device
        msmg_features = torch.zeros_like(x)

        for t in range(self.scales):
            # 生成结构元素（尺寸随尺度t增加）
            se_size = self.size + 2 * t
            se = DifferentiableSE.apply(se_size, device)
            # 计算膨胀和腐蚀（用可微分操作近似）
            dilated = DifferentiableDilation.apply(x, se)
            # 腐蚀用反向卷积近似（或用1 - 膨胀(1 - x)）
            eroded = 1 - DifferentiableDilation.apply(1 - x, se)
            gradient = dilated - eroded
            # 加权累加
            msmg_features += self.weights[t] * gradient

        # 归一化
        min_val = msmg_features.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_val = msmg_features.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized = (msmg_features - min_val) / (max_val - min_val + 1e-8)

        return normalized


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 window_size=7
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # self.lap = LaplacianEnhance()
        self.msmg = MSMGLayer(scales=3)
        self.mamba_block = MambaBlock(dim, LayerNorm_type='WithBias')
        self.line = nn.Linear(dim*2, dim)
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = self.norm_layer(dim)
        self.window_size=window_size
        self.shift_size = window_size // 2
        self.attn = WaveAttention(dim, num_heads, sr_ratio)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.reduce_conv = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    def forward(self, x):
        B, C, H, W = x.shape
        # x = self.lap(x)
        gradient = self.msmg(x)  # [B, C, H, W]
        x = torch.cat([x, gradient], dim=1)  # [B, 2C, H, W]
        x = self.reduce_conv(x)
        y=x
        x1 = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        x = x1 + self.drop_path(self.attn(self.norm1(x1), H, W))
        x2 = self.mamba_block(y)
        x2=to_3d(x2)
        out = torch.cat((x, x2), 2)
        x = self.line(out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        x1=self.mamba_block(to_4d(x, H, W))
        x1 = to_3d(x1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W,mask=attn_mask))
        out = torch.cat((x, x1), 2)
        x = self.line(out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


#######################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ##########################################################################
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=2,
#                                 groups=hidden_features * 2, bias=bias, dilation=2)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#         self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
#         self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
#
#     def pad(self, x, factor):
#         hw = x.shape[-1]
#         t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
#         x = F.pad(x, t_pad, 'constant', 0)
#         return x, t_pad
#
#     def unpad(self, x, t_pad):
#         hw = x.shape[-1]
#         return x[..., t_pad[0]:hw - t_pad[1]]
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x = self.dwconv(x)
#         x, pad_w = self.pad(x, 2)
#         x = torch.fft.rfft2(x)
#         x = self.fft_channel_weight * x + self.fft_channel_bias
#         #        x = torch.nn.functional.normalize(x, 1)
#         x = torch.fft.irfft2(x)
#         x = self.unpad(x, pad_w)
#         x1, x2 = x.chunk(2, dim=1)
#
#         x = F.silu(x1) * x2
#         x = self.project_out(x)
#         return x


##########################################################################
class TransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransMambaBlock, self).__init__()

        # self.trans_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.trans_block = Block(dim, num_heads, ffn_expansion_factor)

        # self.mamba_block = MambaBlock(dim, LayerNorm_type)
        # self.conv = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.trans_block(x)
        # x2 = self.mamba_block(x)
        # out = torch.cat((x1, x2), 1)
        # out = self.conv(out)
        return out


# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(TransformerBlock, self).__init__()
#         # self.lap = LaplacianEnhance()
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#
#     def forward(self, x):
#         # x=self.lap(x)
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#
#         return x


class MambaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type, ):
        super(MambaBlock, self).__init__()
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = ConvSSM(dim)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = ConvSSM(dim)  # FeedForward(dim, ffn_expansion_factor, bias, True)

    def forward(self, x):
        x=rearrange(x, 'b c h w -> b h w c')
        x = x + self.mamba1(self.norm1(x))

        x = x + self.mamba2(self.norm2(x))
        x=rearrange(x, 'b h w c -> b c h w ')
        return x


class DRMamba(nn.Module):
    def __init__(self, dim, reverse):
        super(DRMamba, self).__init__()
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.reverse = reverse

    def forward(self, x):
        b, c, h, w = x.shape
        if self.reverse:
            x = x.flip(1)
        x = self.mamba(x)
        if self.reverse:
            x = x.flip(1)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self, dim_in, out_channels):
        super(CAFM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )
        self.residual = nn.Conv2d(dim_in, out_channels, kernel_size=1)
        self.skip_connection = nn.Conv2d(dim_in, out_channels * 3, kernel_size=1)
        self.parallel = ParallelBlock(out_channels)
        self.channel_splitter = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )
        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(out_channels, out_channels // 2, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(out_channels, out_channels // 2, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(out_channels, out_channels // 2, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(out_channels, out_channels // 2, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(out_channels // 2, out_channels, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(out_channels // 2, out_channels, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(out_channels // 2, out_channels, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(out_channels // 2, out_channels, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        y1, y2 = f1, f2
        residual1 = self.residual(f1)
        residual2 = self.residual(f2)
        f1 = self.conv(f1)
        f2 = self.conv(f2)
        b, c, h, w = f1.size()
        _, _, h2, w2 = f2.size()
        x1, x2 = f1, f2

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)

        a2 = a2.reshape([b, c, h2, w2])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)

        out1 = x1 * a1 + residual1
        out2 = x2 * a2 + residual2

        out1 = self.parallel(out1)
        out2 = self.parallel(out2)

        out1 = out1 + self.skip_connection(y1)
        out2 = out2 + self.skip_connection(y2)

        out1 = self.channel_splitter(out1)
        out2 = self.channel_splitter(out2)

        return out1, out2


class Network(nn.Module):
    def __init__(self, img_channels=3, dropout=0.):
        super().__init__()

        self.Encoder1 = CAFM(img_channels, 32)
        self.Encoder2 = TransMambaBlock(dim=32, num_heads=2, ffn_expansion_factor=2.66,
                                        bias=False, LayerNorm_type='WithBias').to(DEVICE)
        self.Encoder3 = TransMambaBlock(dim=48, num_heads=4, ffn_expansion_factor=2.66,
                                        bias=False, LayerNorm_type='WithBias').to(DEVICE)
        self.latent = nn.Sequential(*[
            TransMambaBlock(dim=64, num_heads=8, ffn_expansion_factor=2.66,
                            bias=False, LayerNorm_type='WithBias') for i in range(2)])
        self.down1 = Downsample(64, 32, kernel_size=1, stride=1, padding=0)
        self.down2 = Downsample(32, 48, kernel_size=1, stride=2, padding=0)
        self.down3 = Downsample(48, 64, kernel_size=3, stride=2, padding=1)

        self.t0 = CD_Transformer(dim=64, proj_kernel=3, kv_proj_stride=4, heads=4, depth=1, mlp_mult=2,
                                 dropout=dropout)
        self.tu1 = CD_Transformer(dim=48, proj_kernel=3, kv_proj_stride=4, heads=2, depth=1, mlp_mult=2,
                                  dropout=dropout)
        self.tu2 = CD_Transformer(dim=32, proj_kernel=3, kv_proj_stride=4, heads=1, depth=1, mlp_mult=2,
                                  dropout=dropout)

        self.up = Upsample(32, 32)
        self.up1 = Upsample(64, 64)
        self.up2 = Upsample(64, 48)
        self.up3 = Upsample(48, 32)

        self.u0 = Upsample(64, 32)
        self.u1 = Upsample(64, 32)
        self.u2 = Upsample(48, 32)
        # self.u3 = Upsample(48, 32)

        self.skip_conn0 = conv3x3(input_dim=64, output_dim=32)
        self.skip_conn1 = conv3x3(input_dim=96, output_dim=48)
        self.skip_conn2 = conv3x3(input_dim=128, output_dim=64)

        self.Mixer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.Mish(inplace=True),
            # nn.Conv2d(64, 48, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(48, eps=1e-5, momentum=0.1),
            # nn.Mish(inplace=True),
        )
        # self.Mixer1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=7, stride=4, padding=3),
        #     nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
        #     nn.Mish(inplace=True),
        # )
        # self.Mixer2 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
        #     nn.Mish(inplace=True),
        # )

        self.ReconstructD = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):  # 3 224 224
        b, c, h, w = A.shape

        out1, out2 = self.Encoder1(A, B)
        # skip_conn = self.skip_conn(torch.cat([conn_x, conn_y], dim=1))

        skip_conn0 = self.skip_conn0(torch.cat([out1, out2], dim=1))  # 32 /2
        # mix0 = torch.cat([x, y], dim=1)

        # x, y = self.down1(x), self.down1(y)  # 64-32

        out1 = self.Encoder2(out1)
        out2 = self.Encoder2(out2)
        out1, out2 = self.down2(out1), self.down2(out2)  # 32-48  /4

        skip_conn1 = self.skip_conn1(torch.cat([out1, out2], dim=1))  # 48 /4
        # mix1 = torch.cat([x, y], dim=1)

        out1 = self.Encoder3(out1)
        out2 = self.Encoder3(out2)
        out1, out2 = self.down3(out1), self.down3(out2)  # 48-64 /8
        skip_conn2 = self.skip_conn2(torch.cat([out1, out2], dim=1))  # 64 /8

        concatenation = torch.cat([out1, out2], dim=1)
        # D = torch.cat([self.Mixer(concatenation), self.Mixer1(mix0), self.Mixer2(mix1)], dim=1)  # 64 /16
        D = self.Mixer(concatenation)  # 64 /16

        D = self.latent(D)

        o0 = self.u0(D, h, w)
        D = self.up1(D, math.ceil(h / 8), math.ceil(w / 8))  # 64

        D = self.t0(D, skip_conn2)
        o1 = self.u1(D, h, w)
        D = self.up2(D, math.ceil(h / 4), math.ceil(w / 4))  # 48

        D = self.tu1(D, skip_conn1)
        # o2=self.u2(D, h, w)
        D = self.up3(D, math.ceil(h / 2), math.ceil(w / 2))  # 32

        D = self.tu2(D, skip_conn0)

        D = self.up(D, h, w)
        D = self.ReconstructD(D)
        o0 = self.ReconstructD(o0)
        o1 = self.ReconstructD(o1)
        # o2=self.ReconstructD(o2)
        return D, o0, o1


if __name__ == '__main__':
    test_tensor_A = torch.rand((1, 3, 520, 520)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 520, 520)).to(DEVICE)
    model = Network().to(DEVICE)
    # model(test_tensor_A, test_tensor_B)
    # print(model)
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))
    Pre = model(test_tensor_A, test_tensor_B)
    print(Pre[0].shape)