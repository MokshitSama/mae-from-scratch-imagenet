from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch.nn as nn
import torch


#image [B,3,224,224]
#  → patch_embed → tokens [B, 196, 768]  + pos_embed
#  → RANDOM MASK: keep 25% (49 visible), drop 75% (147)   ← the novel core
#  → prepend cls → ENCODER (ViT blocks on 49+1 tokens only) → norm   ← cheap: sees 25%
#  → project to decoder dim (512), insert mask tokens, UNSHUFFLE to original order
#  → + decoder pos_embed → DECODER (8 blocks, 512-d) → predict pixels [B,196,768]
#  → LOSS: MSE on MASKED patches only, target = per-patch-normalized pixels


def random_masking(x, mask_ratio):
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))          # 196 * 0.25 = 49

    noise = torch.rand(B, N, device=x.device)     # random score per patch
    ids_shuffle = torch.argsort(noise, dim=1)      # ascending -> random permutation
    ids_restore = torch.argsort(ids_shuffle, dim=1) # inverse permutation (to un-shuffle later)

    ids_keep = ids_shuffle[:, :len_keep]           # indices of the kept patches
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))  # [B, 49, D]

    # binary mask for the LOSS: 0 = kept, 1 = masked (in original patch order)
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)      # un-shuffle so it aligns with original order

    return x_masked, mask, ids_restore


class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,          # encoder = ViT-B
                 dec_dim=512, dec_depth=8, dec_heads=16,          # decoder = light
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches           # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for cls
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # decoder params 
        self.decoder_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, dec_dim)) #+1 is for CLS token
        self.decoder_blocks = nn.ModuleList([Block(dec_dim, dec_heads, qkv_bias=True) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, patch_size ** 2 * in_chans) # 512 -> 768(16*16*3 pixels per patch)


    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs)                      # [B, 196, 768]
        x = x + self.pos_embed[:, 1:, :]                # add pos to PATCHES (slot 0 reserved for cls)
        x, mask, ids_restore = random_masking(x, self.mask_ratio)   # [B, 49, 768]
        cls = self.cls_token + self.pos_embed[:, :1, :] # cls gets position-0 embed
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                  # [B, 50, 768]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)                              # [B, 50, 512]  (cls + 49 visible)

        # build mask tokens for the 147 hidden patches
        n_masked = ids_restore.shape[1] - x.shape[1] + 1       # 196 - 50 + 1 = 147
        mask_tokens = self.mask_token.expand(x.shape[0], n_masked, -1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)      # drop cls: [B, 196, 512] — but SHUFFLED order
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # UN-shuffle → original order
        x = torch.cat([x[:, :1, :], x_], dim=1)                # re-attach cls: [B, 197, 512]

        x = x + self.decoder_pos_embed                         # add pos (now everything's in order)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)                               # [B, 197, 768] — predicted pixels per token
        return x[:, 1:, :]                                     # drop cls → [B, 196, 768]

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]                 # 16
        h = w = imgs.shape[2] // p                         # 14
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)              # gather each patch's pixels together
        return x.reshape(imgs.shape[0], h * w, p * p * 3)  # [B, 196, 768]
    
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)                       # [B, 196, 768] ground-truth pixels
        if self.norm_pix_loss:                             # the +0.5% trick from the paper
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                           # [B, 196] per-patch MSE
        loss = (loss * mask).sum() / mask.sum()            # average over MASKED patches only
        return loss
    
    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)   # [B, 196, 768]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


