from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM implementation for convolutional models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach() if grad_output[0] is not None else None

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        # Standardize input for gradients
        input_tensor.requires_grad = True
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None:
            # Fallback if specific layer backward hook fails
            # This can happen with certain fused operations or in-place activations
            print("[GradCAM] Warning: Gradients not captured. Returning zeros.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        cam = torch.sum(activations, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def close(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_heatmap_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on RGB image (0-1 float image)."""
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * (image * 255).astype(np.uint8) + alpha * heatmap, 0, 255)
    return overlay


def save_gradcam_examples(
    original_images,
    cam_images,
    titles,
    output_path: str,
):
    n = len(original_images)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(n):
        axes[i].imshow(cam_images[i])
        axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
