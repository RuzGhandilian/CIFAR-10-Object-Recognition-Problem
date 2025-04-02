import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import time


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history tracking
        self.train_loss_history = []
        self.train_acc_history = []
        self.clean_train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.epoch_times = []

        self.model.to(self.device)
        self.best_val_acc = 0.0
        self.best_model_state = None

    def train(self, num_epochs):
        plt.figure(figsize=(12, 5))  # Clean figure with two subplots

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            start_time = time()

            # Train with mixup augmentation
            train_loss, train_accuracy = self._train_epoch_mixup()

            # Calculate clean training accuracy
            clean_train_acc = self._calculate_clean_accuracy()

            # Validate
            val_loss, val_accuracy = self._validate()

            # Track history
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_accuracy)
            self.clean_train_acc_history.append(clean_train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_accuracy)
            self.epoch_times.append(time() - start_time)

            # Save best model
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.best_model_state = self.model.state_dict()

            # Print training info
            epoch_time = self.epoch_times[-1]
            remaining_time = np.mean(self.epoch_times) * (num_epochs - epoch - 1)

            print(f"\nEpoch [{epoch + 1}/{num_epochs}] - {epoch_time:.1f}s (ETA: {remaining_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(
                f"Mixup Train Acc: {train_accuracy:.2f}% | Clean Train Acc: {clean_train_acc:.2f}% | Val Acc: {val_accuracy:.2f}%")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Update plots
            self._update_plots()

        # Restore best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete. Best validation accuracy: {self.best_val_acc:.2f}%")

    def _train_epoch_mixup(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        alpha = 0.5  # Mixup hyperparameter

        for inputs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Mixup augmentation
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs.size(0)).to(self.device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels[index])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = 100.0 * correct / total
        return train_loss, train_accuracy

    def _calculate_clean_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.train_loader, desc="Clean Eval", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    def _validate(self, loader=None):
        """Validate model on either validation or test data"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Use provided loader or default to validation loader
        loader = loader if loader is not None else self.val_loader

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(loader)
        val_accuracy = 100.0 * correct / total
        return val_loss, val_accuracy

    def _update_plots(self):
        plt.clf()

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss', linewidth=2)
        plt.plot(self.val_loss_history, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training & Validation Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='Mixup Train Acc', linewidth=2)
        plt.plot(self.clean_train_acc_history, label='Clean Train Acc', linewidth=2, linestyle='--')
        plt.plot(self.val_acc_history, label='Val Acc', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training & Validation Accuracy', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.pause(0.1)
        plt.draw()


    def plot_history(self):
        """Plot training history after training is complete"""
        self._update_plots()
        plt.show()

    def save_model(self, path):
        """Save the best model state"""
        torch.save({
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'train_acc_history': self.train_acc_history,
            'clean_train_acc_history': self.clean_train_acc_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
            'best_val_acc': self.best_val_acc,
        }, path)

    def load_model(self, path):
        """Load a saved model state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_model_state = checkpoint['model_state_dict']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.train_acc_history = checkpoint.get('train_acc_history', [])
        self.clean_train_acc_history = checkpoint.get('clean_train_acc_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.val_acc_history = checkpoint.get('val_acc_history', [])