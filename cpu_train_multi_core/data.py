import os
import torch
from dotenv import load_dotenv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

load_dotenv()


class MNISTDataLoader:
    def __init__(self):
        self.data_dir = os.getenv("DATA_LOCATION")
        self.train_batch_size = int(os.getenv("TRAIN_BATCH_SIZE"))
        self.validation_batch_size = int(os.getenv("VALIDATION_BATCH_SIZE"))
        self.shuffle_train_data = os.getenv("TRAIN_SHUFFLE") == "TRUE"
        self.shuffle_validation_data = os.getenv("VALIDATION_SHUFFLE") == "FALSE"
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))]
        )
        self.worker_count = int(os.getenv("WORKER_COUNT"))

    def setup(self, rank, world_size):
        dataset = datasets.FashionMNIST(
            root=self.data_dir, train=True, transform=self.transform, download=True
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        data_loader = DataLoader(
            dataset, batch_size=self.train_batch_size, shuffle=self.shuffle_train_data, sampler=sampler, num_workers=self.worker_count
        )
        return dataset, sampler, data_loader
    
    def val_loader(self):
        validation_dataset = datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.validation_batch_size, shuffle=self.shuffle_validation_data
        )
        return validation_loader