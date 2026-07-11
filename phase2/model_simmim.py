from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch.nn as nn
import torch

# image [B,3,224,224]
#  → patch_embed → [B,196,768] + pos_embed
#  → RANDOM MASK: replace 60% of tokens with a learnable mask_token (KEEP all 196)   ← SimMIM core
#  → prepend cls → ENCODER (12 ViT blocks on all 197 tokens)  ← sees 100%, no efficiency trick
#  → LINEAR head → predict pixels [B,196,768]
#  → LOSS: L1 on MASKED patches only, raw pixels (no per-patch norm)s


class SIMMIM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,          # encoder = ViT-B
                 mask_ratio=0.6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches           # 196
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for cls
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, patch_size**2 * in_chans)

    def random_mask(self, x):                     # x: [B, 196, 768]
        B, N, D = x.shape
        len_mask = int(N * self.mask_ratio)       # 196 * 0.6 ≈ 117 masked
        ids = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids[:, :len_mask], 1.0)  # 1 = masked (original order)
        w = mask.unsqueeze(-1)
        x = x * (1 - w) + self.mask_token.expand(B, N, -1) * w   # swap masked → mask_token
        return x, mask
    
    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]                 # 16
        h = w = imgs.shape[2] // p                         # 14
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)              # gather each patch's pixels together
        return x.reshape(imgs.shape[0], h * w, p * p * 3)  # [B, 196, 768]

    def forward(self, imgs):
        x = self.patch_embed(imgs)                  # [B, 196, 768]
        x, mask = self.random_mask(x)               # KEEP all 196, masked → mask_token
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)              # [B, 197, 768]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.head(x[:, 1:, :])               # drop cls → [B, 196, 768] predicted pixels

        target = self.patchify(imgs)                # [B, 196, 768]
        loss = (pred - target).abs().mean(dim=-1)   # L1, per-patch → [B, 196]
        loss = (loss * mask).sum() / mask.sum()     # masked patches only
        return loss, pred, mask


