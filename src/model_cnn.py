import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CustomPlantCNN(nn.Module):
    """High-accuracy CNN using a ResNet18 backbone with lightweight head."""

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        try:
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            # Offline fallback when pretrained weights cannot be downloaded.
            backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for name, param in backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )

        self.model = backbone

    def unfreeze_last_stage(self):
        """Optionally unfreeze the last residual stage for a short fine-tuning phase."""
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
