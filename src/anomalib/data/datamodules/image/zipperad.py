# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ZipperAD Data Module.

This module provides a PyTorch Lightning DataModule for the ZipperAD dataset,
a zipper product anomaly classification dataset.
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.zipperad import ZipperADDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from anomalib.utils.path import resolve_with_warning

logger = logging.getLogger(__name__)


class ZipperAD(AnomalibDataModule):
    """ZipperAD DataModule for image-level anomaly classification.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``"./datasets/ZipperAD"``.
        category (str): Category of the ZipperAD dataset (e.g.
            ``"injection"``, ``"nylon"``, ``"plastic_steel"``).
            Defaults to ``"injection"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the
            training images. Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the
            validation images. Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the
            test images. Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if
            stage-specific augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data import ZipperAD
        >>> datamodule = ZipperAD(
        ...     root="./datasets/ZipperAD",
        ...     category="injection",
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image'])
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/ZipperAD",
        category: str = "injection",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        root = resolve_with_warning(root, "ZipperAD")
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.
        """
        self.train_data = ZipperADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = ZipperADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )
