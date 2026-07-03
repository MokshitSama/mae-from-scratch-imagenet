import os
import numpy as np
import cv2
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import glob

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
    def __init__(self, paths, labels, transforms=None):
        self.paths = paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, x):
        path = self.paths[x]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:             # rare ImageNet files cv2 chokes on
            img = np.asarray(Image.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, self.labels[x]

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=(CFG.img_size[0], CFG.img_size[1]), scale=(0.08, 1.0)),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.SmallestMaxSize(max_size=256),                          # was missing -> variable sizes
    A.CenterCrop(height=CFG.img_size[0], width=CFG.img_size[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def build_loaders():
    paths, labels, _ = build_index(os.path.join(CFG.data_root, "train"))

    if CFG.dev:                                    # tiny smoke: first 20 classes (k-fold safe)
        keep = [i for i, l in enumerate(labels) if l < 20]
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    train_idx, val_idx = list(skf.split(paths, labels))[CFG.FOLD]   # labels drive stratification

    tr_paths = [paths[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    va_paths = [paths[i] for i in val_idx]
    va_labels = [labels[i] for i in val_idx]

    train_dataset = IMAGENET_DATASET(tr_paths, tr_labels, transforms=train_transforms)
    val_dataset = IMAGENET_DATASET(va_paths, va_labels, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=False,
                              drop_last=True, persistent_workers=CFG.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=False,
                            drop_last=False, persistent_workers=CFG.num_workers > 0)
    return train_loader, val_loader

if __name__ == "__main__":
    tl, vl = build_loaders()
    imgs, lbls = next(iter(tl))
    print("train batch:", tuple(imgs.shape), imgs.dtype, lbls.min().item(), lbls.max().item())
    imgs, lbls = next(iter(vl))
    print("val batch:", tuple(imgs.shape))
