import os
import glob

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import CFG

ImageFile.LOAD_TRUNCATED_IMAGES = True   # don't die on a rare truncated JPEG


def build_index(split_root):
    classes = sorted(os.listdir(split_root))            # deterministic, alphabetical
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for p in glob.glob(os.path.join(split_root, c, "*")):
            paths.append(p)
            labels.append(class_to_idx[c])
    return paths, labels, class_to_idx


class IMAGENET_DATASET(Dataset):
    """PIL-loading (like Phase 1) so timm's PIL-based RandAugment applies identically."""
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


# EXACTLY Phase 1's augmentation (timm/DeiT recipe) — so the finetune runs on a level field:
#   RandomResizedCrop + flip + RandAugment(rand-m9) + Random Erasing(0.25), bicubic.
train_transforms = create_transform(
    input_size=CFG.img_size[0], is_training=True,
    auto_augment="rand-m9-mstd0.5-inc1",         # RandAugment — same policy as Phase 1
    interpolation="bicubic",
    re_prob=0.25, re_mode="pixel",               # Random Erasing — same as Phase 1
    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
)
val_transforms = create_transform(                # is_training=False -> resize256 / centercrop224
    input_size=CFG.img_size[0], is_training=False, interpolation="bicubic",
    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
)


def build_loaders():
    # train on ALL of train, validate on the held-out 50k /val — identical setup to Phase 1 (fair delta)
    tr_paths, tr_labels, _ = build_index(os.path.join(CFG.data_root, "train"))
    va_paths, va_labels, _ = build_index(os.path.join(CFG.data_root, "val"))

    if CFG.dev:                                    # tiny smoke: first 20 classes on BOTH splits
        tr = [i for i, l in enumerate(tr_labels) if l < 20]
        va = [i for i, l in enumerate(va_labels) if l < 20]
        tr_paths, tr_labels = [tr_paths[i] for i in tr], [tr_labels[i] for i in tr]
        va_paths, va_labels = [va_paths[i] for i in va], [va_labels[i] for i in va]

    train_dataset = IMAGENET_DATASET(tr_paths, tr_labels, transform=train_transforms)
    val_dataset = IMAGENET_DATASET(va_paths, va_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True,       # pin+prefetch = the overlap speedup
                              drop_last=True, persistent_workers=CFG.num_workers > 0, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True,
                            drop_last=False, persistent_workers=CFG.num_workers > 0)
    return train_loader, val_loader


if __name__ == "__main__":
    tl, vl = build_loaders()
    imgs, lbls = next(iter(tl))
    print("train batch:", tuple(imgs.shape), imgs.dtype)
    imgs, lbls = next(iter(vl))
    print("val batch:", tuple(imgs.shape))
