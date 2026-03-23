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
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

    def load_state_dict(self, state_dict, strict: bool = True):
        """Support older checkpoints that used `backbone.*` and `fc.*` key prefixes."""
        if any(k.startswith("backbone.") or k.startswith("fc.") for k in state_dict.keys()):
            converted = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    converted[f"model.{key[len('backbone.'):]}"] = value
                elif key.startswith("fc."):
                    converted[f"model.fc.{key[len('fc.'):]}"] = value
                elif key.startswith("avgpool."):
                    # Legacy standalone pool module no longer used as a separate key.
                    continue
                else:
                    converted[key] = value
            state_dict = converted

        return super().load_state_dict(state_dict, strict=strict)
