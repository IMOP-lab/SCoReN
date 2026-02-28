import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import math
from math import sqrt
from typing import Tuple


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    

class SEfromOther(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x, other):  # x:(B,C,D,H,W), other:(B,C,D,H,W)
        B, C, D, H, W = x.shape
        ctx = other.mean(dim=(2,3,4))               # (B, C)
        s = self.fc2(F.gelu(self.fc1(ctx))).sigmoid().view(B, C, 1, 1, 1)
        return x * s

class BlurPool3D(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        a = torch.tensor([1., 2., 1.])
        k = (a[:, None, None] * a[None, :, None] * a[None, None, :])
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None, :, :, :])  # (1,1,3,3,3)
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        B, C, _, _, _ = x.shape
        k = self.kernel.expand(C, 1, 3, 3, 3)                  # (C,1,3,3,3)
        return F.conv3d(x, k, stride=self.stride, padding=1, groups=C)

class LightUp3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.dw(x)
        x = F.gelu(x)
        return self.pw(x)

class FusionGate(nn.Module):
    def __init__(self, in_ch_a, in_ch_b, out_ch):
        super().__init__()
        hidden = max(16, out_ch // 2)
        self.fc1 = nn.Linear(in_ch_a + in_ch_b, hidden)
        self.fc2 = nn.Linear(hidden, out_ch)

    def forward(self, a, b):   
        ga = a.mean(dim=(2, 3, 4))
        gb = b.mean(dim=(2, 3, 4))
        g = torch.cat([ga, gb], dim=1)
        w = self.fc2(F.gelu(self.fc1(g))).sigmoid().view(g.size(0), -1, 1, 1, 1)
        return w                                                   # (B,out_ch,1,1,1)

class CrossCorrGate(nn.Module):
    def __init__(self, c_in, hidden=64, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc2 = nn.Linear(hidden, c_in)

    def forward(self, a, b):   # a,b: (B,C,D,H,W)
        B, C = a.shape[:2]
        fa = a.view(B, C, -1)
        fb = b.view(B, C, -1)
        ma = fa.mean(-1, keepdim=True)
        mb = fb.mean(-1, keepdim=True)
        sa = fa.std(-1, keepdim=True)
        sb = fb.std(-1, keepdim=True)
        corr = ((fa - ma) * (fb - mb)).mean(-1) / (sa.squeeze(-1) * sb.squeeze(-1) + self.eps)  # (B,C)
        w = self.fc2(F.gelu(self.fc1(corr))).sigmoid().view(B, C, 1, 1, 1)
        return w

class MLP3D_CN(nn.Module):
    def __init__(self, dim, expansion=4, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.pw1  = nn.Conv3d(dim, hidden, 1)
        self.dw   = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv3d(hidden, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop  = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        y = self.pw2(self.act(self.dw(self.pw1(self.norm(x)))))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1, 1) * y
        return self.drop(y)

class CrossScaleTokenAttentionMH(nn.Module):
    def __init__(self, c_low, gsize=(2, 2, 2), num_heads=4):
        super().__init__()
        assert c_low % num_heads == 0, "c_low"
        self.h = num_heads
        self.d = c_low // num_heads
        self.q = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.k = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.v = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.token_dw = nn.Conv3d(c_low, c_low, 3, padding=1, groups=c_low, bias=False)
        self.proj = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.gsize = gsize
        self.norm_q = LayerNorm(c_low, data_format="channels_first")
        self.norm_kv = LayerNorm(c_low, data_format="channels_first")

    def forward(self, x2, x1_low):  # both (B,2C,D2,H2,W2)
        B, C, D2, H2, W2 = x2.shape
        gD, gH, gW = self.gsize
        tokens = F.adaptive_avg_pool3d(x1_low, (gD, gH, gW))
        tokens = tokens + self.token_dw(tokens)                   

        q = self.q(self.norm_q(x2)).view(B, self.h, self.d, -1).transpose(2, 3)  # (B,h,N,d)
        k = self.k(self.norm_kv(tokens)).view(B, self.h, self.d, -1)             # (B,h,d,G)
        v = self.v(tokens).view(B, self.h, self.d, -1).transpose(2, 3)           # (B,h,G,d)

        attn = (q @ k) / math.sqrt(self.d)                        # (B,h,N,G)
        attn = attn.softmax(-1)
        out = attn @ v                                            # (B,h,N,d)
        out = out.transpose(2, 3).reshape(B, C, D2, H2, W2)
        return self.proj(out)                                     # (B,2C,D2,H2,W2)

class GatedAxialCrossEnhancePlus(nn.Module):
    def __init__(self, c_high, k=5):
        super().__init__()
        p = k // 2
        self.ax_d = nn.Conv3d(c_high, c_high, (k, 1, 1), padding=(p, 0, 0), groups=c_high, bias=False)
        self.ax_h = nn.Conv3d(c_high, c_high, (1, k, 1), padding=(0, p, 0), groups=c_high, bias=False)
        self.ax_w = nn.Conv3d(c_high, c_high, (1, 1, k), padding=(0, 0, p), groups=c_high, bias=False)
        self.mix  = nn.Conv3d(c_high, c_high, 1, bias=False)

        self.gate_gen = nn.Sequential(
            nn.Conv3d(c_high, c_high, 3, padding=1, groups=c_high, bias=False),
            nn.Conv3d(c_high, c_high, 1, bias=False),
            nn.Sigmoid()
        )
        self.se_from_other = SEfromOther(c_high, reduction=4)

    def forward(self, x1, x2_up):
        y = self.mix(self.ax_d(x1) + self.ax_h(x1) + self.ax_w(x1))
        g = self.gate_gen(x2_up)
        y = y * (1.0 + g)
        y = self.se_from_other(y, x2_up)
        return y


class DiSCo(nn.Module):
    
    'Duplex inter-Scale Statistical Concordance operator (DiSCo)'
    
    def __init__(self, C, token_grid=(2, 2, 2), mlp_expansion=4, heads=4, drop_path=0.0):
        super().__init__()
        self.down_x1 = nn.Sequential(
            BlurPool3D(C, stride=2),
            nn.Conv3d(C, 2 * C, kernel_size=1, bias=False)
        )
        self.proj_x2_up = LightUp3D(2 * C, C)

        self.csta = CrossScaleTokenAttentionMH(c_low=2 * C, gsize=token_grid, num_heads=heads)

        self.gace = GatedAxialCrossEnhancePlus(c_high=C)

        self.cc_low  = CrossCorrGate(c_in=2 * C, hidden=2 * C)
        self.cc_high = CrossCorrGate(c_in=C,     hidden=2 * C)

        self.fuse_low_skip         = nn.Conv3d(2 * C, 2 * C, kernel_size=1, bias=False)
        self.fuse_high_from_lowup  = nn.Conv3d(2 * C, C,     kernel_size=1, bias=False)

        self.gate_low  = FusionGate(in_ch_a=2 * C, in_ch_b=2 * C, out_ch=2 * C)
        self.gate_high = FusionGate(in_ch_a=C,     in_ch_b=2 * C, out_ch=C)

        self.mlp1 = MLP3D_CN(C,     expansion=mlp_expansion, drop_path=drop_path)
        self.mlp2 = MLP3D_CN(2 * C, expansion=mlp_expansion, drop_path=drop_path)

    def forward(self, x1, x2):
        """
        x1: (B, C,  D,   H,   W)
        x2: (B,2C, D/2, H/2, W/2)
        """
        B, C, D, H, W = x1.shape
        _, C2, D2, H2, W2 = x2.shape
        assert C2 == 2 * C and D2 * 2 == D and H2 * 2 == H and W2 * 2 == W, "Shape mismatch."

        x1_low = self.down_x1(x1)                         # (B,2C,D/2,H/2,W/2)
        x2_up  = self.proj_x2_up(x2)                      # (B,C, D,  H,  W)

        y2_A = self.csta(x2, x1_low)                      # (B,2C,D/2,H/2,W/2)

        y1_B = self.gace(x1, x2_up)                       # (B,C,D,H,W)

        w_low  = self.cc_low (a=x1_low, b=x2)             # (B,2C,1,1,1)
        w_high = self.cc_high(a=x1,     b=x2_up)          # (B,C,1,1,1)
        y2_A = y2_A * w_low
        y1_B = y1_B * w_high

        gate_low = self.gate_low(x1_low, x2)              # (B,2C,1,1,1)
        y2 = gate_low * y2_A + (1.0 - gate_low) * self.fuse_low_skip(x1_low)

        y2_up_feat = F.interpolate(y2, scale_factor=2, mode='trilinear', align_corners=False)  # (B,2C,D,H,W)
        y1_from_low = self.fuse_high_from_lowup(y2_up_feat)                                     # (B,C,D,H,W)
        gate_high = self.gate_high(x1, y2)                                                     # (B,C,1,1,1)
        y1 = gate_high * y1_B + (1.0 - gate_high) * y1_from_low

        y1_final = y1 + self.mlp1(x1)
        y2_final = y2 + self.mlp2(x2)
        return y1_final, y2_final
    
    
    
class SCoReN(nn.Module):
    
    'Statistical Correlation Reconciliation Network (SCoReN)'
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        conv_block: bool = True,
        res_block: bool = True,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        
        self.disco_1 = DiSCo(C=fea[0], token_grid=(2,2,2), mlp_expansion=4, heads=2)
        
        self.disco_2 = DiSCo(C=fea[2], token_grid=(2,2,2), mlp_expansion=4, heads=4)

        
        


    def forward(self, x: torch.Tensor):
                
        x0 = self.conv_0(x)
        
        x1 = self.down_1(x0) 
                      
        x2 = self.down_2(x1) 
                
        x3 = self.down_3(x2) 
        
        x4 = self.down_4(x3) 

        x0, x1 = self.disco_1(x0,x1)
        x2, x3 = self.disco_2(x2,x3)

        
        u4 = self.upcat_4(x4, x3)
        
        u3 = self.upcat_3(u4, x2)
        
        u2 = self.upcat_2(u3, x1)

        u1 = self.upcat_1(u2, x0)
        
        logits = self.final_conv(u1)

        return logits