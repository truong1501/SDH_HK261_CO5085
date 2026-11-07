import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        time_proj = self.time_mlp(self.act(t_emb))[(..., ) + (None, ) * 2]
        h = h + time_proj
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        attn_out, _ = self.mha(x_flat, x_flat, x_flat, need_weights=False)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return x + attn_out

class UNetDDPM(nn.Module):
    """
    Standard U-Net architecture for DDPM (Variant 1: CNN Baseline).
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList()
        current_ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(Block(current_ch, out_ch, time_emb_dim))
            current_ch = out_ch

        self.downsample = nn.AvgPool2d(2)

        # Bottleneck with Self-Attention
        self.mid1 = Block(current_ch, current_ch, time_emb_dim)
        self.mid_attn = SelfAttention(current_ch)
        self.mid2 = Block(current_ch, current_ch, time_emb_dim)

        # Upsampling
        self.ups = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(Block(current_ch + out_ch, out_ch, time_emb_dim))
            current_ch = out_ch

        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for i, down in enumerate(self.downs):
            x = down(x, t_emb)
            skips.append(x)
            if i < len(self.downs) - 1:
                x = self.downsample(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for i, up in enumerate(self.ups):
            if i > 0:
                x = self.upsample(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x, t_emb)

        return self.final_conv(F.silu(self.final_norm(x)))
