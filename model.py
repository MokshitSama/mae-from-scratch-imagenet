import torch
import timm
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=False,
                 n_classes=1000, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,        # recipe; helps stability
        )                                         # no global_pool='' -> cls-token pooled [B,768]
        self.num_features = self.backbone.num_features
        self.head = nn.Linear(self.num_features, n_classes)

    def forward(self, x):
        features = self.backbone(x)               # [B, 768] (cls token)
        return self.head(features)                # [B, 1000] logits


if __name__ == "__main__":
    model = Model()
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.1f}M")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("output:", tuple(out.shape))
    assert out.shape == (2, 1000)
    print("smoke passed")
