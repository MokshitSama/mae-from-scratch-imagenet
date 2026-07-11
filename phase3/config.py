import os

class CFG:
    seed = 1123
    V = int(os.environ.get("V", "1"))       # set on the command line: V=3 bash run_phase3.sh  (no manual editing)
    model_name = "vit_base_patch16_224"
    dev = False

    # data
    data_root = "/mnt/md0/imagenet"
    img_size = (224, 224)
    n_splits = 10
    FOLD = 0
    batch_size = 128                        # per-GPU
    num_workers = 8

    # the MAE checkpoint we finetune FROM (Phase 2 output)
    mae_ckpt = "/mnt/md0/mokshit/codes/mae_imagenet/phase2/try1/simmim_V1/last.pth"


    # train  (MAE ViT-B fine-tuning recipe, paper Table 8)
    epochs = 100                            # was 5;  ViT-B finetune = 100 epochs
    warmup_epochs = 5                       # was 20; short warmup — encoder is already good
    base_lr = 1e-3                          # was 1e-4; peak = base_lr * eff_batch/256
    weight_decay = 0.05                     # was 0.3
    layer_decay = 0.65                      # NEW — LLRD, 0.65 for ViT-B (0.75 is ViT-L)
    drop_path = 0.1
    acc_steps = 2                           # was 5; finetune wants a SMALL effective batch (~1k)
    clip_grad = 0.0
    ema_decay = 0.9999                      # optional (see note in train.py)
    validate_every = 1

    # mixup / cutmix (finetune augmentation)
    mixup = 0.8
    cutmix = 1.0
    label_smoothing = 0.1

    # derived
    world_size = 6
    eff_batch = batch_size * world_size * acc_steps   # 768  (~1024 target)
    peak_lr = base_lr * eff_batch / 256               # 3e-3  (the HEAD's LR; early layers ≪ via LLRD)
