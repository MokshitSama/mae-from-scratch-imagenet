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
    batch_size = 256                        # per-GPU micro-batch
    num_workers = 8

    # train
    epochs = 5
    warmup_epochs = 20
    base_lr = 1e-4
    weight_decay = 0.3
    drop_path = 0.1
    ema_decay = 0.9999
    acc_steps = 5
    clip_grad = 0.0                         # 0 = off (paper default)
    validate_every = 1

    # derived
    world_size = 6
    eff_batch = batch_size * world_size * acc_steps   # 3840
    peak_lr = base_lr * eff_batch / 256               # 1.5e-3
