import torch
from timm.loss import SoftTargetCrossEntropy

from transformers import get_cosine_schedule_with_warmup

from config import CFG
from time import time
from tqdm import tqdm

def save_ckpt(path, model, accelerator):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(accelerator.unwrap_model(model).state_dict(), path)


def load_ckpt(path, model, accelerator):
    state = torch.load(path, map_location="cpu", weights_only=True)
    accelerator.unwrap_model(model).load_state_dict(state)

def build_param_groups(model, weight_decay):
    """Weight decay on weight matrices only; none on norms/biases (1-D params)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_criterion_optimizer_scheduler(model, num_warmup_steps, num_training_steps):
    criterion = SoftTargetCrossEntropy()                      # mixup -> soft targets
    optimizer = torch.optim.AdamW(
        build_param_groups(model, CFG.weight_decay),          # NOT model.parameters()
        lr=CFG.peak_lr, betas=(0.9, 0.95),
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
    return criterion, optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scheduler, accelerator, mixup_fn, ema=None):
    model.train()
    running = 0.0
    n_seen, t0 = 0, time()
    bar = tqdm(loader, disable=not accelerator.is_main_process)
    for images, targets in loader:                            # accelerate puts batch on device
        images = images.to(memory_format=torch.channels_last)

        images, targets = mixup_fn(images, targets)           # -> soft targets
        with accelerator.accumulate(model):                   # grad-accum + loss scaling handled
            logits = model(images)                            # bf16 autocast handled by Accelerator
            loss = criterion(logits, targets)
            accelerator.backward(loss)                        # do NOT divide loss yourself
            if accelerator.sync_gradients and CFG.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), CFG.clip_grad)
            optimizer.step()
            if accelerator.sync_gradients:                    # i.e. once per real update
                scheduler.step()
                if ema is not None:
                    ema.update(accelerator.unwrap_model(model))
            optimizer.zero_grad()
        running += loss.item()

        n_seen += images.size(0)
        if accelerator.is_main_process:
            img_s = n_seen * accelerator.num_processes / (time() - t0)
            bar.set_postfix(loss=f"{loss.item():.3f}", img_s=f"{img_s:.0f}")
    return running / len(loader), n_seen * accelerator.num_processes / (time() - t0)


@torch.no_grad()
def evaluate(model, loader, accelerator):
    """Top-1 / top-5. Runs on ALL processes (gather needs them) - gate only logging to main."""
    model.eval()
    top1 = top5 = total = 0
    for images, targets in loader:
        images = images.to(memory_format=torch.channels_last)
        logits = model(images)
        # gather across the 6 GPUs and de-pad the last (uneven) batch
        logits, targets = accelerator.gather_for_metrics((logits, targets))
        _, pred = logits.topk(5, dim=1)
        correct = pred.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        total += targets.size(0)
    return 100 * top1 / total, 100 * top5 / total
