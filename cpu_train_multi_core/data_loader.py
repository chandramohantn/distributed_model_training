from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler


class MNISTDataLoader:
    def __init__(self, config):
        self.data_dir = config.data.data_dir
        self.train_batch_size = config.data.train_batch_size
        self.validation_batch_size = config.data.validation_batch_size
        self.shuffle_train_data = config.data.shuffle_train_data
        self.shuffle_validation_data = config.data.shuffle_validation_data
        self.worker_count = config.data.worker_count
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))]
        )

    def setup(self, rank, world_size):
        dataset = datasets.FashionMNIST(
            root=self.data_dir, train=True, transform=self.transform, download=True
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=self.shuffle_train_data)
        data_loader = DataLoader(
            dataset, batch_size=self.train_batch_size, sampler=sampler, num_workers=self.worker_count
        )
        return dataset, sampler, data_loader
    
    def validation_loader(self):
        dataset = datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        loader = DataLoader(
            dataset, batch_size=self.validation_batch_size, shuffle=self.shuffle_validation_data
        )
        return loader