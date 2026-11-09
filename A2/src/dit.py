import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    """adaLN modulation: scale and shift normalized tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations via frequency embeddings + MLP.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    (Key innovation in Peebles & Xie, ICCV 2023)
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        # Regresses 6 modulation parameters (shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention with adaLN-Zero
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaLN-Zero
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT: adaLN modulation + Linear projection + unpatchify.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) architecture for Image Generation.
    (Peebles & Xie, ICCV 2023 - Variant 2: Modern Transformer Model)
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0
    ):
        super().__init__()
        if input_size < patch_size or input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2

        # 1. Patch embedding
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # 2. Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        # 3. Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # 4. DiT Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 5. Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with small normal values
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation linear layers as in the paper (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output linear projection
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: [N, T, patch_size**2 * C]
        returns imgs: [N, C, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t):
        # x: [N, C, H, W]
        x = self.x_embedder(x)  # [N, hidden_size, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [N, num_patches, hidden_size]
        x = x + self.pos_embed

        t_emb = self.t_embedder(t)  # [N, hidden_size]

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_layer(x, t_emb)  # [N, num_patches, patch_size * patch_size * out_channels]
        return self.unpatchify(x)
