class CFG:
    seed = 1123
    V = 1                                   # bump for a new run version
    model_name = "vit_base_patch16_224"
    dev = False                             # tiny smoke subset (first 20 classes)

    # data
    data_root = "/mnt/md0/imagenet"
    img_size = (224, 224)
    n_splits = 10
    FOLD = 0
    batch_size = 128                        # per-GPU micro-batch (was 128; encoder sees only 25% patches, room on 32GB)
    num_workers = 8

    # train
    epochs = 400                            # was 800; finetune acc 400ep 83.3 vs 800ep 83.4 -> ~0.1% for 2x time
    warmup_epochs = 40                      # keep the paper's absolute 40ep warmup (safe for the high peak_lr)
    base_lr = 1.5e-4
    weight_decay = 0.05
    mask_ratio = 0.6
    drop_path = 0.1
    ema_decay = 0.9999
    acc_steps = 6                           # was 5; 192*6*4 = 4608 eff_batch, ~half the micro-batches of before
    clip_grad = 0.0                         # 0 = off (paper default)
    compile = False                         # measured +1.6% at 6-GPU (comm-bound) -> not worth it; keep eager
    validate_every = 1

    # derived
    world_size = 6
    eff_batch = batch_size * world_size * acc_steps   # 4608
    peak_lr = base_lr * eff_batch / 256               # 2.7e-3 (linear-scaled from base_lr)
