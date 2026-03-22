import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class TransferMobileNetV2(nn.Module):
    """MobileNetV2 transfer learning model with custom classifier head."""

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        try:
            backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            # Fallback for offline environments.
            backbone = models.mobilenet_v2(weights=None)

        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self.model = backbone

    def unfreeze_last_blocks(self, num_blocks: int = 4):
        """Unfreeze last feature blocks for fine-tuning."""
        total = len(self.model.features)
        start_idx = max(0, total - num_blocks)
        for idx in range(start_idx, total):
            for param in self.model.features[idx].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
