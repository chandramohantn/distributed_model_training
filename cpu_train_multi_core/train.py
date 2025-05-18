import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
            self, config, rank, model, train_loader, validation_loader, sampler
        ):
        self.rank = rank
        self.model = DDP(model)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.sampler = sampler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.learning_rate)
        self.writer = SummaryWriter(log_dir=os.path.join(config.train.log_dir, f"rank_{self.rank}"))
        self.save_dir = config.train.checkpoint_dir
        self.config = config
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.early_stop_counter = config.train.early_stopping_patience
        self.start_epoch = 0

        if config.train.resume_from_checkpoint:
            self._load_checkpoint(config.train.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            print(f"[{self.rank}] No checkpoint found at {checkpoint_path}, starting from scratch")
            return
            
        print(f"[{self.rank}] loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"[{self.rank}] Resumed from epoch: {self.start_epoch}")


    def train_one_epoch(self, epoch):
        epoch_loss = 0.0
        for batch_index, data in enumerate(self.train_loader):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss

            if (batch_index + 1) % 1000 == 0:
                print(f"Epoch: {epoch + 1} Batch: {batch_index + 1} Loss: {loss.item():.4f}")

        return epoch_loss

    def train(self):
        number_of_epochs = self.config.train.epochs
        for epoch in range(number_of_epochs):
            self.sampler.set_epoch(epoch)
            self.model.train()

            epoch_loss = self.train_one_epoch(epoch)
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"[{self.rank}] Epoch: {epoch + 1} Avg Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

            print("Performing Validation")
            val_loss, val_accuracy = self.validate(epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_accuracy, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                print("Saving the model")
                self.save_checkpoint(epoch)
            else:
                self.early_stop_counter += 1
                print(f"[{self.rank}] No improvement, patience: {self.early_stop_counter}")
                if self.early_stop_counter >= self.config.train.early_stopping_patience:
                    print(f"[{self.rank}] Early stopping at epoch: {epoch}")
                    break

    def validate(self, epoch):
        self.model.eval()
        validation_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                validation_loss += loss

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = validation_loss / len(self.validation_loader)
        accuracy = 100 * correct / total
        print(f"[{self.rank}] Epoch: {epoch} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch):
        if self.rank == 0:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
            }, checkpoint_path)
            print(f"[{self.rank}] Saved checkpoint to {checkpoint_path}")