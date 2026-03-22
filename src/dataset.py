import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder variant that tolerates transient missing/corrupt files on Windows."""

    def __init__(self, root: str, transform=None):
        super().__init__(root=root, transform=transform)
        self._prune_missing_samples()

    def _prune_missing_samples(self) -> None:
        valid_samples = []
        for path, target in self.samples:
            if Path(path).exists():
                valid_samples.append((path, target))

        self.samples = valid_samples
        self.imgs = valid_samples
        self.targets = [target for _, target in valid_samples]

    def __getitem__(self, index: int):
        # Retry a few times because some Windows file operations can be briefly non-deterministic.
        last_exc = None
        for _ in range(5):
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except (FileNotFoundError, OSError) as exc:
                last_exc = exc
                index = (index + 1) % len(self.samples)

        raise last_exc


def _is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _count_class_dirs_with_images(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        return 0

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        return 0

    valid = 0
    for class_dir in class_dirs:
        if any(_is_image_file(fp) for fp in class_dir.rglob("*")):
            valid += 1
    return valid


def _contains_class_folders(root: Path, min_class_dirs: int = 2) -> bool:
    # Require at least 2 immediate child class folders with images.
    # This prevents accidentally treating an intermediate container (e.g. only "PlantVillage")
    # as a class root.
    return _count_class_dirs_with_images(root) >= min_class_dirs


def _class_has_direct_images(class_dir: Path) -> bool:
    # True class folders usually contain images directly.
    # Container folders (e.g. nested "PlantVillage") often only contain subfolders.
    return any(p.is_file() and _is_image_file(p) for p in class_dir.iterdir())


def _normalize_label(label: str) -> str:
    # Normalize class labels to compare naming variations such as
    # Tomato___Early_blight vs Tomato_Early_blight.
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _resolve_selected_classes(
    selected_classes: List[str], available_classes: List[str]
) -> Tuple[List[str], List[str]]:
    available_lookup = {_normalize_label(name): name for name in available_classes}

    resolved: List[str] = []
    missing: List[str] = []
    for label in selected_classes:
        if label in available_classes:
            resolved.append(label)
            continue

        mapped = available_lookup.get(_normalize_label(label))
        if mapped is not None:
            resolved.append(mapped)
        else:
            missing.append(label)

    # Preserve order and remove duplicates.
    deduped = list(dict.fromkeys(resolved))
    return deduped, missing


def _discover_image_root(base_path: Path) -> Optional[Path]:
    """Find the folder that directly contains class subfolders with images."""
    if _contains_class_folders(base_path):
        return base_path

    for path in base_path.rglob("*"):
        if path.is_dir() and _contains_class_folders(path):
            return path
    return None


def download_plantvillage(raw_root: str) -> Path:
    """
    Download PlantVillage dataset using kagglehub (preferred) if not available locally.
    Returns the directory containing class subfolders.
    """
    raw_root_path = Path(raw_root)
    raw_root_path.mkdir(parents=True, exist_ok=True)

    # Reuse existing local dataset if already present.
    local_root = _discover_image_root(raw_root_path)
    if local_root is not None:
        print(f"[dataset] Using existing dataset at: {local_root}")
        return local_root

    try:
        import kagglehub  # type: ignore

        print("[dataset] Downloading PlantVillage via kagglehub...")
        download_path = Path(kagglehub.dataset_download("emmarex/plantdisease"))
        image_root = _discover_image_root(download_path)

        if image_root is None:
            raise FileNotFoundError(
                "Downloaded dataset does not contain discoverable class folders."
            )

        print(f"[dataset] Downloaded dataset path: {image_root}")
        return image_root
    except Exception as exc:
        raise RuntimeError(
            "Could not download PlantVillage automatically. "
            "Install kagglehub and ensure internet access, or place dataset manually in data/raw."
        ) from exc


def _split_class_files(
    files: List[Path], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    files_copy = files.copy()
    rng.shuffle(files_copy)

    n_total = len(files_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = files_copy[:n_train]
    val_files = files_copy[n_train : n_train + n_val]
    test_files = files_copy[n_train + n_val : n_train + n_val + n_test]

    return train_files, val_files, test_files


def _copy_files(file_list: List[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        dst = destination_dir / src.name
        # Avoid duplicate names by prefixing stem if needed.
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while (destination_dir / f"{stem}_{idx}{suffix}").exists():
                idx += 1
            dst = destination_dir / f"{stem}_{idx}{suffix}"
        shutil.copy2(src, dst)


def _split_already_prepared(
    data_dir: Path,
    expected_class_names: Optional[List[str]] = None,
    min_required_classes: int = 2,
) -> bool:
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            return False
        # At least one class folder with one image.
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            return False
        if len(class_dirs) < min_required_classes:
            return False

        if expected_class_names is not None:
            actual = {p.name for p in class_dirs}
            if set(expected_class_names) != actual:
                return False

        if not any(any(_is_image_file(fp) for fp in c.rglob("*")) for c in class_dirs):
            return False
    return True


def _clear_existing_splits(data_path: Path) -> None:
    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def prepare_data_splits(
    data_dir: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    selected_classes: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Path]:
    """
    Create train/val/test splits under data_dir from PlantVillage source.
    Splits are created only if missing.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    expected_classes = selected_classes if selected_classes else None
    min_required_classes = max(2, len(selected_classes)) if selected_classes else 2

    if not force_rebuild and _split_already_prepared(
        data_path,
        expected_class_names=expected_classes,
        min_required_classes=min_required_classes,
    ):
        print("[dataset] Existing split found. Skipping split generation.")
        return {
            "train": data_path / "train",
            "val": data_path / "val",
            "test": data_path / "test",
        }

    if force_rebuild:
        print("[dataset] Force rebuild requested. Clearing existing splits...")
    else:
        print("[dataset] Existing split is missing/invalid. Rebuilding splits...")

    _clear_existing_splits(data_path)

    source_root = download_plantvillage(str(data_path / "raw"))

    available_classes = sorted(
        [p.name for p in source_root.iterdir() if p.is_dir() and _class_has_direct_images(p)]
    )
    if not available_classes:
        raise RuntimeError("No class folders found in source dataset.")

    if selected_classes:
        classes_to_use, missing = _resolve_selected_classes(selected_classes, available_classes)
        if missing:
            raise ValueError(
                f"Selected classes not found: {missing}. Available examples: {available_classes[:10]}"
            )
    else:
        classes_to_use = available_classes

    print(f"[dataset] Preparing splits for {len(classes_to_use)} classes...")

    for split in ["train", "val", "test"]:
        (data_path / split).mkdir(parents=True, exist_ok=True)

    for class_name in classes_to_use:
        class_dir = source_root / class_name
        files = [fp for fp in class_dir.rglob("*") if fp.is_file() and _is_image_file(fp)]
        if len(files) < 10:
            print(f"[dataset] Warning: class '{class_name}' has low sample count: {len(files)}")

        train_files, val_files, test_files = _split_class_files(
            files, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        _copy_files(train_files, data_path / "train" / class_name)
        _copy_files(val_files, data_path / "val" / class_name)
        _copy_files(test_files, data_path / "test" / class_name)

    print("[dataset] Split creation complete.")
    return {
        "train": data_path / "train",
        "val": data_path / "val",
        "test": data_path / "test",
    }


def get_transforms(img_size: int = 224, augment_level: str = "strong"):
    if augment_level == "light":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(18),
                transforms.RandomPerspective(distortion_scale=0.18, p=0.2),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.12,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.14), ratio=(0.3, 3.3), value="random"),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def _build_weighted_sampler(targets: List[int]) -> WeightedRandomSampler:
    target_tensor = torch.tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(target_tensor).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    # Softer than full inverse-frequency to avoid overfitting rare classes.
    class_weights = class_counts.pow(-0.5)
    sample_weights = class_weights[target_tensor]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
    balanced_sampling: bool = False,
    augment_level: str = "strong",
):
    train_transform, eval_transform = get_transforms(img_size=img_size, augment_level=augment_level)

    train_ds = SafeImageFolder(root=str(Path(data_dir) / "train"), transform=train_transform)
    val_ds = SafeImageFolder(root=str(Path(data_dir) / "val"), transform=eval_transform)
    test_ds = SafeImageFolder(root=str(Path(data_dir) / "test"), transform=eval_transform)
    sampler = _build_weighted_sampler(train_ds.targets) if balanced_sampling else None

    pin_memory = torch.cuda.is_available()
    # Keep this off for better stability on Windows.
    persistent_workers = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    class_names = train_ds.classes

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }, class_names


def denormalize_image(tensor):
    """
    Inverse-normalization for ImageNet-normalized tensors.
    Expected input shape: (C, H, W)
    Result: (C, H, W) in range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img = tensor.clone().detach() * std + mean
    return torch.clamp(img, 0, 1)


def show_image_with_gradcam(image, heatmap, alpha=0.5):
    """
    Overlays a Grad-CAM heatmap on a denormalized image.
    image: (C, H, W) or (H, W, C) range [0, 1]
    heatmap: (H, W) range [0, 1]
    """
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()

    # Resize and colormap heatmap
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # Blend and clip
    overlay = (1 - alpha) * image + alpha * heatmap_color
    return np.clip(overlay, 0, 1)


def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
