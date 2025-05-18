import hydra
from omegaconf import DictConfig
import torch.multiprocessing as mp
from classifier import GarmentClassifier
from data_loader import MNISTDataLoader
from train import Trainer
from utils import setup_ddp, cleanup_ddp


def train_model(rank, world_size, config):
    print(f"[{rank}] Starting training")
    setup_ddp(rank, world_size)

    data_loader = MNISTDataLoader(config)
    _, sampler, train_loader = data_loader.setup(rank, world_size)
    validation_loader = data_loader.validation_loader()

    model = GarmentClassifier(config)
    trainer = Trainer(config, rank, model, train_loader, validation_loader, sampler)
    trainer.train()

    cleanup_ddp()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    world_size = 3
    mp.spawn(
        train_model, args=(world_size, config), nprocs=world_size, join=True
    )

if __name__ == "__main__":
    main()