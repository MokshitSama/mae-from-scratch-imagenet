import math, json
from pathlib import Path
from time import time
import torch
# ---- inductor pin-memory workaround: this box can't pin the ~310MB autotune buffer,
#      so force pin_memory=False in empty_strided (harmless; our loader is pin_memory=False too) ----
_es = torch.empty_strided
def _es_nopin(*a, **k):
    k['pin_memory'] = False
    return _es(*a, **k)
torch.empty_strided = _es_nopin
# --------------------------------------------------------------------------------------------------
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
from transformers import get_cosine_schedule_with_warmup

from config import CFG
from dataset import build_mae_loader
from model_simmim import SIMMIM
from utils import build_param_groups, save_ckpt
from tqdm import tqdm


def train_one_epoch_mae(model, loader, optimizer, scheduler, accelerator, epoch):
    model.train()
    running, n_seen, t0 = 0.0, 0, time()
    bar = tqdm(loader, disable=not accelerator.is_main_process,
               desc=f"ep {epoch}/{CFG.epochs}", dynamic_ncols=True)
    for step, (imgs, _) in enumerate(bar):          # loop over BAR so the fraction/ETA advance
        with accelerator.accumulate(model):
            loss, _, _ = model(imgs)
            accelerator.backward(loss)
            if accelerator.sync_gradients and CFG.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), CFG.clip_grad)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()
        running += loss.item()
        n_seen  += imgs.size(0)
        if accelerator.is_main_process:
            img_s = n_seen * accelerator.num_processes / (time() - t0)
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            avg=f"{running/(step+1):.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                            img_s=f"{img_s:.0f}")
    return running / len(loader), n_seen * accelerator.num_processes / (time() - t0)

def main():
    set_seed(CFG.seed)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=CFG.acc_steps)
    out = Path("try1") / f"simmim_V{CFG.V}"
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)

    loader = build_mae_loader()
    loader = accelerator.prepare(loader)                     # shard FIRST, then compute steps
    updates_per_epoch = math.ceil(len(loader) / CFG.acc_steps)
    num_training_steps = updates_per_epoch * CFG.epochs
    num_warmup_steps   = updates_per_epoch * CFG.warmup_epochs

    model = SIMMIM(mask_ratio=CFG.mask_ratio)                   # NOT the classifier
    optimizer = torch.optim.AdamW(build_param_groups(model, CFG.weight_decay),
                                  lr=CFG.peak_lr, betas=(0.9, 0.95))     # β2=0.95
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if CFG.compile:
        model = torch.compile(model)                        # compile BEFORE prepare -> DDP(compile), 0 graph breaks
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.num_processes > 1:                       # register_comm_hook is a DDP method; skip on 1-GPU smoke
        model.register_comm_hook(None, dh.bf16_compress_hook)   # the comm-wall fix

    for epoch in range(CFG.epochs):
        loss, img_s = train_one_epoch_mae(model, loader, optimizer, scheduler, accelerator, epoch)
        save_ckpt(out / "last.pth", model, accelerator)     # save FULL MAE (Phase 3 pulls the encoder)
        if accelerator.is_main_process:
            accelerator.print(f"ep {epoch:3d} | recon_loss {loss:.4f} | {img_s:.0f} img/s")

if __name__ == "__main__":
    main()
