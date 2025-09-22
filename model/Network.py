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
from .MambaBlock import MambaBlock
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math
import pywt
from torch.autograd import Function

class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print('\nCUDA is available. Calculation is performing on ' + str(
            torch.cuda.get_device_name(torch.cuda.current_device())) + '.\n')
    else:
        DEVICE = 'cpu'
        print('\nOOPS! CUDA is not available! Calculation is performing on CPU.\n')


DEVICE = GPUorCPU.DEVICE

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



class ParallelBlock(nn.Module):
    def __init__(self, channels):
        super(ParallelBlock, self).__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        # self.layer = nn.Conv2d(channels*3, channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out1 = F.ReLU(self.branch1(x))
        out2 = F.ReLU(self.branch2(x))
        out3 = F.ReLU(self.branch3(x))
        out = torch.cat((out1, out2, out3), dim=1)
        # out = self.layer(out)
        return out




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



class conv3x3(nn.Module):
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



class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        # self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
        # bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class FRDB_Modulated(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(FRDB_Modulated, self).__init__()
        self.nChannels = nChannels

        # DenseNet-style 对 magnitude 的增强
        nChannels_1 = nChannels
        modules1 = []
        for _ in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)

        # DenseNet-style 对 phase 的增强
        nChannels_2 = nChannels
        modules2 = []
        for _ in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)


        self.fft_scale = nn.Parameter(torch.ones(1, nChannels, 1, 1))
        self.fft_bias = nn.Parameter(torch.zeros(1, nChannels, 1, 1))

    def forward(self, x):
    
        _, _, H, W = x.shape


        x_freq = torch.fft.rfft2(x, norm='backward')  # shape: (B, C, H, W//2 + 1)


        x_freq_real = x_freq.real * self.fft_scale + self.fft_bias
        x_freq = torch.complex(x_freq_real, x_freq.imag)

        mag = torch.abs(x_freq)  # 幅值
        pha = torch.angle(x_freq)  # 相位

 
        mag = self.dense_layers1(mag)
        mag = self.conv_1(mag)

        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)


        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

    
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')


        return out + x


class MambaModule(nn.Module):
    def __init__(self, dim, LayerNorm_type, ):
        super(MambaModule, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = MambaBlock(dim)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = MambaBlock(dim)  # FeedForward(dim, ffn_expansion_factor, bias, True)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = x + self.norm1(self.mamba1(x))

        x = x.flip(dims=[3])
        x = x + self.norm2(self.mamba2(x))
        x = x.flip(dims=[3])
        x = rearrange(x, 'b h w c -> b c h w ')
        return x

##########################################################################
class FDM_Mamba(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FDM_Mamba, self).__init__()

        # self.trans_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.Block = FRDB_Modulated(dim)

        self.mamba_block = MambaModule(dim, LayerNorm_type)
        # self.conv = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # out = self.trans_block(x)
        #x = self.lap(x)
        x = self.Block(x)
        x = self.mamba_block(x)

        # out = torch.cat((x1, x2), 1)
        out = self.conv(x)
        return out




class CAFM(nn.Module):  
    def __init__(self, dim_in, out_channels):
        super(CAFM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(dim_in, out_channels, kernel_size=1)
        self.skip_connection = nn.Conv2d(dim_in, out_channels * 3, kernel_size=1)
        self.parallel = ParallelBlock(out_channels)
        self.channel_splitter = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
        #a1 = F.softmax(a1, dim=1)

        a2 = a2.reshape([b, c, h2, w2])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        #a2 = F.softmax(a2, dim=1)

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
        self.Encoder2 = FDM_Mamba(dim=32, num_heads=2, ffn_expansion_factor=2.66,
                                        bias=False, LayerNorm_type='WithBias').to(DEVICE)
        self.Encoder3 = FDM_Mamba(dim=48, num_heads=4, ffn_expansion_factor=2.66,
                                        bias=False, LayerNorm_type='WithBias').to(DEVICE)
        self.latent = nn.Sequential(*[
            FDM_Mamba(dim=64, num_heads=8, ffn_expansion_factor=2.66,
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

        self.skip_conn0 = conv3x3(input_dim=64, output_dim=32)
        self.skip_conn1 = conv3x3(input_dim=96, output_dim=48)
        self.skip_conn2 = conv3x3(input_dim=128, output_dim=64)

        self.Mixer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.ReconstructD1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )
        self.ReconstructD2 = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )
        self.ReconstructD = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):  # 3 224 224
        b, c, h, w = A.shape

        out1, out2 = self.Encoder1(A, B)

        skip_conn0 = self.skip_conn0(torch.cat([out1, out2], dim=1))  # 32 /2
        

        out1 = self.Encoder2(out1)
        out2 = self.Encoder2(out2)
        out1, out2 = self.down2(out1), self.down2(out2)  # 32-48  /4

        skip_conn1 = self.skip_conn1(torch.cat([out1, out2], dim=1))  # 48 /4
        

        out1 = self.Encoder3(out1)
        out2 = self.Encoder3(out2)
        out1, out2 = self.down3(out1), self.down3(out2)  # 48-64 /8
        skip_conn2 = self.skip_conn2(torch.cat([out1, out2], dim=1))  # 64 /8

        concatenation = torch.cat([out1, out2], dim=1)
        
        D = self.Mixer(concatenation)  # 64 /16

        D = self.latent(D)

        # o0 = self.u0(D, h, w)

        D = self.up1(D, math.ceil(h / 8), math.ceil(w / 8))  # 64

        D = self.t0(D, skip_conn2)
        o1 = D

        D = self.up2(D, math.ceil(h / 4), math.ceil(w / 4))  # 48

        D = self.tu1(D, skip_conn1)
        o2 = D

        D = self.up3(D, math.ceil(h / 2), math.ceil(w / 2))  # 32

        D = self.tu2(D, skip_conn0)

        D = self.up(D, h, w)
        D = self.ReconstructD(D)
        # o0 = self.ReconstructD(o0)
        o1 = self.ReconstructD1(o1)  # 1/8
        o2 = self.ReconstructD2(o2)  # 1/4
        # print(D.shape)
        # print(o1.shape)
        # print(o2.shape)
        return D, o1, o2


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
