"""Generate syntetic dataset for testing."""

from math import pi, exp
from typing import Optional

import lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


class ConstantDataModule(pl.LightningDataModule):
    """Create dataset from random numbers in [-1, 1]."""

    def __init__(self, 
                 batch_size  : Optional[int]=None, 
                 num_workers : int = 96, 
                 num_samples : int = 100_000,
                 c           : float = 0.01, 
                 path        : str ="./data"):
        """Initialize parameters."""
        super().__init__()
        self.batch_size = num_samples if batch_size is None else batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.path        = path
        self.c           = c

    def prepare_data(self):
        """Generate data if not present."""  
        filepath = "/".join([self.path, f"constant_data_N{self.num_samples}_c{self.c}"]) 
        # Try to load the dataset, if not found, generate it. 
        try: 
            self.data = torch.load(filepath)
        except:
            # Generate a constant valued function in the range -1, 1 with value
            # self.c.
            x = (torch.rand(self.num_samples, 1) - 0.5) * 2.0
            y = self.c * torch.ones(*x.size()) 
            self.data = TensorDataset(x,y)
            torch.save(self.data, filepath) 

    def setup(self, stage):
        """Setup data split."""

        if stage == "fit":
            self.train, self.val = random_split(self.data, [0.9, 0.1])

    def train_dataloader(self):
        """Return train DataLoader."""
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=False)


class SinDataModule(pl.LightningDataModule):
    """Create dataset from random numbers in [-1, 1]."""

    def __init__(self, 
                 batch_size:Optional[int]=None, 
                 num_workers : int = 96, 
                 num_samples : int = 100_000,
                 c : float = 0.01, 
                 path : str ="./data"):
        """Initialize parameters."""
        super().__init__()
        self.batch_size = num_samples if batch_size is None else batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.path        = path
        self.c           = c

    def __check_sample_dataset(self, 
                               dt : torch.tensor,
                               c : float,
                               N : int,
                               interval : Optional[list[float]] = None):
        """
            Checks whether a tensor `dt` satisfies the separability
            condition:
                |<x_i, x_j>| \le \frac{e^{-4c}}{8N}

            with `c` being a parameter, while `N` is the length of the dataset
            `dt`. And such that the norm is in the specified interval if not
            None.

            Returns a mask of the dt flattened vector.
        """
        bound = exp(-4 * c) / (8*N)
        v     = dt.flatten()
        
        outs  = (v != v) # 0 bit array

        if interval is not None: # check if it is in the interval
            lower_b, higher_b = interval
            outs |= (lower_b > v.abs()) | (higher_b < v.abs())
       
        # checks separability / orthogonality condition
        for xi in v:
            idx   = (v * xi) >= bound
            outs |= idx

        return outs

    def create_dataset(self, max_range : float = 10,
                   min_range : float = 2,
                   c : float = 0.001,
                   N : int = 100,
                   MAX_ITER : int = 10):
        """
            Creates a dataset of N elements, x_i such that 
                |<x_i, x_j>| \le \exp(-4c)/(8*N)
            in a range specified by `max_range` and `min_range`.
        """

        training_set = torch.ones(N,1) 
        outliers_mask = training_set.flatten() == training_set.flatten()
        nout = N 

        for i in range(MAX_ITER):
            # re-sample the outliers 
            noise                         = (torch.rand(nout, 1) - 0.5) * 2 *  max_range
            training_set[outliers_mask,0] = noise[:, 0]
            outliers_mask = self.__check_sample_dataset(training_set, c, N, interval=[min_range, max_range])
            nout = sum(outliers_mask)

            # exit if no outliers found
            if nout == 0:
                break
        
        # returns the wholes tensor or a part of it, if after 10 steps it
        # doesn't reach the goal. 
        return training_set[~outliers_mask]

    def prepare_data(self):
        """Generate data if not present."""  
        filepath = "/".join([self.path, f"sin_data_N{self.num_samples}_c{self.c}"]) 
        # Try to load the dataset, if not found, generate it. 
        try: 
            self.data = torch.load(filepath)
        except:
            K = exp(-4 * self.c) / self.num_samples / 8
            x = self.create_dataset(K ** (0.5), 0, self.c, self.num_samples)
            y = torch.sin(pi * x / K ** (0.5))
            self.data = TensorDataset(x,y)
            torch.save(self.data, filepath) 

    def setup(self, stage):
        """Setup data split."""

        if stage == "fit":
            self.train, self.val = random_split(self.data, [0.9, 0.1])

    def train_dataloader(self):
        """Return train DataLoader."""
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=False)

class CIFAR10DataModule(pl.LightningDataModule):
    """Data Loader manager."""

    def __init__(self, batch_size : Optional[int] = None,
                 training_data_percentage : float = 0.2, 
                 num_workers : int = 96,
                 path : str = "./data"):
        """Initialize parameters."""
        super().__init__()
        self.path = path
        self.training_data_percentage = training_data_percentage 
        # In case the batch_size is larger than the available training data,
        # set it to a unique batch.
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                  (0.2023, 0.1994, 0.2010))])

    def prepare_data(self):
        """Download data if not present."""
        CIFAR10(self.path, train=True, download=True)
        CIFAR10(self.path, train=False, download=True)

    def setup(self, stage):
        """Setup data splits."""
        if stage == "fit":
            cifar = CIFAR10(self.path, train=True,
                                 transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar,
                                            [self.training_data_percentage, 
                                            1.0 -
                                             self.training_data_percentage])
            # Run full GD if batch size is not specified
            if self.batch_size is None:
                self.batch_size = len(self.cifar10_train)


        if stage == "test":
            self.cifar10_test = CIFAR10(
                self.path, train=False, transform=self.transform
            )
            if self.batch_size is None:
                self.batch_size = len(self.cifar10_test) // 10

        if stage == "predict":
            self.cifar10_predict = CIFAR10(
                self.path, train=False, transform=self.transform
            )

    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(
            self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(
            self.cifar10_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return test DataLoader."""
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Return prediction DataLoader."""
        return DataLoader(
            self.cifar10_predict, batch_size=self.batch_size, num_workers=self.num_workers
        )

class MNISTDataModule(pl.LightningDataModule):
    """Data Loader manager."""

    def __init__(self, 
                 batch_size : Optional[int] = None,
                 training_data_percentage : float = 0.2, 
                 num_workers : int = 96,
                 path : str = "./data"):
        """Initialize parameters."""
        super().__init__()
        self.path = path
        self.training_data_percentage = training_data_percentage
        self.batch_size = batch_size 

        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        """Download data if not present."""
        MNIST(self.path, train=True, download=True)
        MNIST(self.path, train=False, download=True)

    def setup(self, stage):
        """Setup data splits."""
        if stage == "fit":
            mnist_full = MNIST(self.path, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full,
                                            [self.training_data_percentage, 
                                             1.0-self.training_data_percentage])
            # Run full GD if batch size is not specified
            if self.batch_size is None:
                self.batch_size = len(self.mnist_train)

        if stage == "test":
            self.mnist_test = MNIST(
                self.path, train=False, transform=self.transform
            )
            if self.batch_size is None:
                self.batch_size = len(self.mnist_test) // 10

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.path, train=False, transform=self.transform
            )

    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return test DataLoader."""
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Return prediction DataLoader."""
        return DataLoader(
            self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers
        )
