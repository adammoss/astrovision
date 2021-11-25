import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import h5py
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class LensChallengeSpace1(VisionDataset):
    """`Gravitational Lens Finding Challenge <http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``strong-lensing-space-based-challenge1`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "strong-lensing-space-based-challenge1"
    url = "https://storage.googleapis.com/strong-lensing-challenge/strong-lensing-space-based-challenge1.tar.gz"
    filename = "strong-lensing-space-based-challenge1.tar.gz"
    train_list = [
        ["train1.h5", "974162b87307597831ac0ecd2d3f8255"],
    ]
    test_list = [
        ["test1.h5", "47357720fbb398212018568c30f80dd1"],
    ]
    meta = {
    }
    target_class = "is_lens"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pair_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the h5py arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            entry = h5py.File(file_path, 'r')
            self.data.append(entry['data'][:])
            self.targets.extend(entry[self.target_class])
            entry.close()

        self.data = np.vstack(self.data).reshape(-1, 101, 101)

        self.pair_transform = pair_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.pair_transform is not None:
            pair_img = Image.fromarray(img)
            pair_img = self.pair_transform(pair_img)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.pair_transform is not None:
            return (img, pair_img), target
        else:
            return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
