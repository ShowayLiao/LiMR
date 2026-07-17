# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ZipperAD Dataset.

This module provides a PyTorch Dataset implementation for the ZipperAD dataset.
ZipperAD is an image-level anomaly classification dataset for zipper products,
organized in a category-based folder structure similar to MVTec AD.

Dataset structure:
    root/
    ├── injection/
    │   ├── train/good/
    │   └── test/
    │       ├── good/
    │       └── <defect_name>/
    ├── nylon/
    │   └── ...
    └── plastic_steel/
        └── ...

The dataset is image-level only (classification), without pixel-level ground
truth masks.
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils.path import get_datasets_dir

IMG_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


class ZipperADDataset(AnomalibDataset):
    """ZipperAD dataset class for image-level anomaly classification.

    Args:
        root (Path | str | None): Path to root directory containing the dataset.
            Defaults to ``None``, which uses ``./datasets/ZipperAD``.
        category (str): Category name (e.g. ``"injection"``, ``"nylon"``,
            ``"plastic_steel"``).
        augmentations (Transform | None, optional): Augmentations to apply to
            the input images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - ``Split.TRAIN``
            or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets.image.zipperad import ZipperADDataset
        >>> dataset = ZipperADDataset(
        ...     root="./datasets/ZipperAD",
        ...     category="injection",
        ...     split="train",
        ... )
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']
        >>> sample["image"].shape
        torch.Size([3, H, W])
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "injection",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "ZipperAD"

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = _make_zipperad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def _make_zipperad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create ZipperAD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``root/train/good/image.jpg``
        ``root/test/good/image.jpg``
        ``root/test/<defect_name>/image.jpg``

    Note:
        Unlike MVTec AD, the ZipperAD dataset does not have ground truth
        segmentation masks. The task is image-level classification only.

    Args:
        root (Path | str): Path to dataset category root directory.
        split (str | Split | None, optional): Dataset split to filter by.
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions.
            Defaults to ``IMG_EXTENSIONS``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset category
            - split: Dataset split (train/test)
            - label: Class label (good or defect name)
            - image_path: Full path to image file
            - mask_path: Empty string (no masks available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Raises:
        RuntimeError: If no valid images are found.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root), *f.parts[-3:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Build full image path
    samples["image_path"] = (
        samples["path"] + "/" + samples["split"] + "/" + samples.label + "/" + samples.image_path
    )

    # Create label index: 0 for normal (good), 1 for anomalous
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # Filter out ground_truth if present (user doesn't have pixel annotations)
    samples = samples[samples["split"] != "ground_truth"].reset_index(drop=True)

    # No masks available - set empty mask_path for classification-only task
    samples["mask_path"] = ""

    # Sort by image path for deterministic ordering
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Image-level classification only
    samples.attrs["task"] = "classification"

    # Filter by split if specified
    if split:
        samples = samples[samples["split"] == split].reset_index(drop=True)

    return samples
