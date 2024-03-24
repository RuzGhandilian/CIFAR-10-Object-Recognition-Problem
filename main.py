import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from network import BasicBlock, ResNet
from data import CIFAR10Dataset
from modeltrainer import ModelTrainer

if __name__ == "__main__":
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10Dataset(root='./data', train=True, transform=transform)
    test_dataset = CIFAR10Dataset(root='./data', train=False, transform=transform)

    # Split dataset into training, validation, and test sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize the ResNet instance
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18-like architecture

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

    # Train the network
    trainer = ModelTrainer(resnet, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.train(num_epochs=3)