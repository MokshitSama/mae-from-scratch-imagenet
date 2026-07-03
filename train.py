import math
import json
from pathlib import Path

import timm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from timm.utils import ModelEmaV2
from timm.data import Mixup

from config import CFG
from dataset import build_loaders
from model import Model
from utils import get_criterion_optimizer_scheduler, train_one_epoch, evaluate, save_ckpt, load_ckpt


accelerator = Accelerator(mixed_precision="bf16",
                         gradient_accumulation_steps=CFG.acc_steps)

set_seed(CFG.seed)

out = Path("/mnt/md0/mokshit/codes/mae_imagenet/try1") / f"Version_{CFG.V}_{CFG.model_name}_fold{CFG.FOLD}"
if accelerator.is_main_process:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: v for k, v in vars(CFG).items() if not k.startswith("__")}, indent=2, default=str))

model = Model(model_name=CFG.model_name ,n_classes=1000, drop_path_rate=CFG.drop_path)
train_loader, val_loader = build_loaders()

train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
updates_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)

num_training_steps = updates_per_epoch * CFG.epochs
num_warmup_steps   = updates_per_epoch * CFG.warmup_epochs

criterion, optimizer, scheduler = get_criterion_optimizer_scheduler(
    model, num_warmup_steps, num_training_steps)
model, optimizer = accelerator.prepare(model, optimizer)                   # wrap model+opt

ema = ModelEmaV2(accelerator.unwrap_model(model), decay=CFG.ema_decay)
mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                 mode="batch", label_smoothing=0.1, num_classes=1000)

best_score = 0.0
for epoch in range(CFG.epochs):
    loss, img_s = train_one_epoch(model, train_loader, criterion, optimizer,
                           scheduler, accelerator, mixup_fn, ema)
    t1, t5 = evaluate(ema.module, val_loader, accelerator)   # ALL processes (gather)s                          # gate only logging/saving
    improved = t1 > best_score
    if improved:
        best_score = t1
    save_ckpt(out / "last.pth", ema.module, accelerator)
    if improved:
        save_ckpt(out / "best.pth", ema.module, accelerator)

    if accelerator.is_main_process:
        accelerator.print(f"ep {epoch} loss {loss:.3f} val@1 {t1:.2f} val@5 {t5:.2f} img/s {img_s:.0f}")
