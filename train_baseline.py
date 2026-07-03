"""
Phase 1 - supervised ViT-B/16 from scratch on ImageNet-1K.
Follows the MAE paper Table 11 recipe (He et al., 2022, arXiv:2111.06377):
  AdamW (b1=0.9, b2=0.95), wd 0.3, base_lr 1e-4 x eff_bs/256, cosine, 20ep warmup,
  300 epochs, RandAug(9,0.5) + Mixup 0.8 + CutMix 1.0 + label smoothing 0.1,
  drop_path 0.1, EMA 0.9999. Eval = single 224 center crop, top-1 (on EMA weights).

NOTE: our val/ is peeled from train (in-distribution), so the ABSOLUTE % won't match
the paper's 82.3 - track the Phase1 vs Phase3 delta, not the raw number.

Smoke (single GPU on the idle 3090, PCI index 4):
    python train_baseline.py --dev --gpu 4

Full run on 6x RTX 5090 (DDP via torchrun):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 \
    torchrun --nproc_per_node=6 train_baseline.py \
        --data_root /mnt/md0/imagenet --epochs 300 --batch_size 128 --accum_steps 5
"""
import argparse
import csv
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image, ImageFile
import timm  # noqa: F401  (ensures timm is importable up front)
from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

from dataset import build_index
from model import Model

ImageFile.LOAD_TRUNCATED_IMAGES = True  # don't die overnight on a truncated JPEG


# ----------------------------- distributed utils -----------------------------
def setup_distributed(args):
    if "RANK" in os.environ:  # launched via torchrun
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.device = torch.device(f"cuda:{args.local_rank}")
    else:  # single-process
        args.distributed = False
        args.rank, args.world_size, args.local_rank = 0, 1, 0
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}")
    return args


def is_main(args):
    return args.rank == 0


def reduce_sum(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ----------------------------- data -----------------------------
class ImageListDataset(Dataset):
    """PIL-loading dataset so timm's RandAugment (PIL-based) applies. Reuses
    build_index() from dataset.py for enumeration. (The cv2+Albumentations path
    in dataset.py is for Phase 2 MAE pretrain, which only needs RRC.)"""

    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_loaders(args):
    train_tf = create_transform(
        input_size=224, is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",   # the exact MAE/DeiT RandAug policy
        interpolation="bicubic",
        re_prob=0.25, re_mode="pixel",          # random erasing (DeiT default)
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
    )
    val_tf = create_transform(                  # is_training=False -> resize256/centercrop224
        input_size=224, is_training=False, interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
    )

    tr_paths, tr_labels, _ = build_index(os.path.join(args.data_root, "train"))
    va_paths, va_labels, _ = build_index(os.path.join(args.data_root, "val"))
    if args.dev:
        tr_paths, tr_labels = tr_paths[:2000], tr_labels[:2000]
        va_paths, va_labels = va_paths[:1000], va_labels[:1000]

    train_ds = ImageListDataset(tr_paths, tr_labels, train_tf)
    val_ds = ImageListDataset(va_paths, va_labels, val_tf)

    if args.distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=False, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, train_sampler


# ----------------------------- optim / schedule -----------------------------
def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)      # norms + biases get NO weight decay
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step, warmup, total, peak, floor):
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


# ----------------------------- eval -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = torch.zeros((), device=device)
    top5 = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs)
        _, pred = out.topk(5, dim=1)
        correct = pred.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum()
        top5 += correct[:, :5].sum()
        total += targets.numel()
    reduce_sum(top1); reduce_sum(top5); reduce_sum(total)
    return (100 * top1 / total).item(), (100 * top5 / total).item()


# ----------------------------- train -----------------------------
def train_one_epoch(model, model_raw, ema, loader, optimizer, criterion, mixup_fn,
                    epoch, state, args):
    model.train()
    device, accum = args.device, args.accum_steps
    running = seen = 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)

    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        raw_targets = targets
        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, targets)   # -> soft targets

        is_update = ((i + 1) % accum == 0) or (i + 1 == n_batches)
        # skip DDP gradient allreduce until the accumulation window closes
        sync_ctx = model.no_sync() if (args.distributed and not is_update) else nullcontext()
        with sync_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)
                loss = criterion(out, targets)
            if not torch.isfinite(loss):          # NaN/Inf guard: drop the batch
                if is_main(args):
                    print(f"[warn] non-finite loss @ ep{epoch} step{i}; skipping")
                optimizer.zero_grad(set_to_none=True)
                continue
            (loss / accum).backward()

        running += loss.item() * raw_targets.size(0)
        seen += raw_targets.size(0)

        if is_update:
            lr = lr_at(state["step"], state["warmup_steps"], state["total_steps"],
                       args.peak_lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model_raw)
            state["step"] += 1

    stat = torch.tensor([running, seen], device=device)
    reduce_sum(stat)
    avg_loss = (stat[0] / stat[1]).item()
    img_s = stat[1].item() / (time.time() - t0)
    return avg_loss, img_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/md0/imagenet")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128, help="per-GPU micro-batch")
    p.add_argument("--accum_steps", type=int, default=5)
    p.add_argument("--base_lr", type=float, default=1e-4, help="lr per 256 samples")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=0.0, help="0 = off (paper default)")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", default="./checkpoints/baseline")
    p.add_argument("--resume", default="")
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0, help="device id for single-GPU runs")
    p.add_argument("--dev", action="store_true", help="tiny smoke run")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.dev:
        args.epochs, args.warmup_epochs, args.accum_steps, args.num_workers = 2, 1, 1, 4

    setup_distributed(args)
    torch.manual_seed(args.seed + args.rank)
    torch.backends.cudnn.benchmark = True
    device = args.device

    args.eff_batch = args.batch_size * args.world_size * args.accum_steps
    args.peak_lr = args.base_lr * args.eff_batch / 256

    train_loader, val_loader, train_sampler = build_loaders(args)
    updates_per_epoch = max(1, len(train_loader) // args.accum_steps)
    state = {
        "step": 0,
        "warmup_steps": args.warmup_epochs * updates_per_epoch,
        "total_steps": args.epochs * updates_per_epoch,
    }

    model_raw = Model(n_classes=1000, pretrained=False,
                      drop_path_rate=args.drop_path).to(device)
    ema = ModelEmaV2(model_raw, decay=args.ema_decay, device=device)
    model = DDP(model_raw, device_ids=[args.local_rank]) if args.distributed else model_raw

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                     mode="batch", label_smoothing=0.1, num_classes=1000)
    criterion = SoftTargetCrossEntropy()   # mixup always on -> soft-target CE

    optimizer = torch.optim.AdamW(
        build_param_groups(model_raw, args.weight_decay),
        lr=args.peak_lr, betas=(0.9, 0.95),
    )

    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_raw.load_state_dict(ckpt["model"])
        ema.module.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        state["step"] = ckpt.get("step", start_epoch * updates_per_epoch)
        if is_main(args):
            print(f"resumed from {args.resume} @ epoch {start_epoch}")

    log_w = log_f = None
    if is_main(args):
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"world={args.world_size} bs/gpu={args.batch_size} accum={args.accum_steps} "
              f"eff_batch={args.eff_batch} peak_lr={args.peak_lr:.2e}")
        print(f"train batches/gpu={len(train_loader)} updates/epoch={updates_per_epoch}")
        log_path = os.path.join(args.out_dir, "log.csv")
        new = not os.path.isfile(log_path)
        log_f = open(log_path, "a", newline="")
        log_w = csv.writer(log_f)
        if new:
            log_w.writerow(["epoch", "train_loss", "lr", "val_top1", "val_top5", "img_s", "secs"])

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        te = time.time()
        train_loss, img_s = train_one_epoch(model, model_raw, ema, train_loader,
                                            optimizer, criterion, mixup_fn, epoch, state, args)
        cur_lr = optimizer.param_groups[0]["lr"]

        val1 = val5 = float("nan")
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            val1, val5 = evaluate(ema.module, val_loader, device)   # EMA weights

        if is_main(args):
            secs = time.time() - te
            print(f"ep {epoch:3d} | loss {train_loss:.3f} | lr {cur_lr:.2e} | "
                  f"val@1 {val1:.2f} val@5 {val5:.2f} | {img_s:.0f} img/s | {secs:.0f}s")
            log_w.writerow([epoch, f"{train_loss:.4f}", f"{cur_lr:.3e}",
                            f"{val1:.3f}", f"{val5:.3f}", f"{img_s:.0f}", f"{secs:.0f}"])
            log_f.flush()
            ckpt = {"epoch": epoch, "step": state["step"], "model": model_raw.state_dict(),
                    "ema": ema.module.state_dict(), "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc, "args": vars(args)}
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_last.pth"))
            if val1 > best_acc:
                best_acc = val1
                ckpt["best_acc"] = best_acc
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best.pth"))
                print(f"  new best val@1 {best_acc:.2f}")

    if args.distributed:
        dist.destroy_process_group()
    if is_main(args):
        log_f.close()
        print(f"done. best val@1 = {best_acc:.2f}")


if __name__ == "__main__":
    main()
