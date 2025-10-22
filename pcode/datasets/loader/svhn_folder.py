from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

def define_svhn_folder(root, is_train, transform, target_transform, download):
    return SVHN(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        is_download=download,
    )

class SVHN(data.Dataset):
    
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root,
        is_train="train",
        transform=None,
        target_transform=None,
        is_download=False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.is_download = is_download

        if self.is_train:
            tr_data = self.load_svhn_data("train")
            ex_data = self.load_svhn_data("extra")
            self.data, self.labels = self.build_training(tr_data, ex_data)
        else:
            self.data, self.labels = self.load_svhn_data("test")

    def load_svhn_data(self, data_type):
        url = self.split_list[data_type][0]
        filename = self.split_list[data_type][1]
        file_md5 = self.split_list[data_type][2]

        if self.is_download:
            self.download(url, filename, file_md5)

        if not self._check_integrity(data_type, filename):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        data, labels = self._load_svhn_data(filename)
        return data, labels

    def _load_svhn_data(self, filename):
        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self.root, filename))

        data = loaded_mat["X"]
        labels = loaded_mat["y"].astype(np.int64).squeeze()

        np.place(labels, labels == 10, 0)
        data = np.transpose(data, (3, 2, 0, 1))
        return data, labels

    def build_training(self, tr_data, ex_data):
        def get_include_indices(total, exclude):
            return list(set(list(total)) - set(exclude))

        def exclude_samples(data, size_per_class):
            images, labels = data
            exclude_indices = []

            for label in range(min(labels), max(labels) + 1):
                matched_indices = np.where(labels == label)[0]
                exclude_index = matched_indices.tolist()[:size_per_class]
                exclude_indices += exclude_index

            include_indices = get_include_indices(
                range(images.shape[0]), exclude_indices
            )
            images = images[include_indices, :, :, :]
            labels = labels[include_indices]
            return images, labels

        def build_train(tr_data, ex_data):
            selected_tr_images, selected_tr_labels = exclude_samples(tr_data, 400)
            selected_ex_images, selected_ex_labels = exclude_samples(ex_data, 200)
            images = np.concatenate([selected_tr_images, selected_ex_images])
            labels = np.concatenate([selected_tr_labels, selected_ex_labels])
            return images, labels

        return build_train(tr_data, ex_data)

    def __getitem__(self, index):
        
        img, target = self.data[index], int(self.labels[index])

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self, data_type, filename):
        root = self.root
        md5 = self.split_list[data_type][2]
        fpath = os.path.join(root, filename)
        return check_integrity(fpath, md5)

    def download(self, url, filename, file_md5):
        download_url(url, self.root, filename, file_md5)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.is_train)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
