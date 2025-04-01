import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from src.network import BasicBlock, ResNet
#from src.network2 import BottleneckBlock, ResNet
from src.data import CIFAR10Dataset
from src.modeltrainer import ModelTrainer

if __name__ == "__main__":
    # Enhanced data transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset with different transforms for train and test
    train_dataset = CIFAR10Dataset(root='data', train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(root='data', train=False, transform=test_transform)

    # Split dataset into training and validation sets
    train_size = int(0.9 * len(train_dataset))  # 90% for training
    val_size = len(train_dataset) - train_size  # 10% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])

    # Update validation dataset to use test transform (no augmentation)
    val_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the ResNet with dropout
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18-like architecture
    #resnet = ResNet(BottleneckBlock, [3, 4, 6, 3])

    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Enhanced optimizer with weight decay
    optimizer = optim.SGD(
        resnet.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    # Better learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Initialize trainer with early stopping
    trainer = ModelTrainer(
        model=resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train the network
    trainer.train(num_epochs=30)

    # Evaluate on test set
    test_loss, test_acc = trainer._validate(loader=test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # Save the final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
    }, 'saved_models/final_model.pth')

    print("Model saved in 'saved_models/final_model.pth'")