# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging
import os
from utils import ensure_dir

class Trainer:
    def __init__(self, model, dataset, config, device):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.config['training']['learning_rate']
        )
        self.best_val_loss = np.inf
        self.epochs_no_improve = 0

    def train(self):
        X_train, y_train = self.dataset.train_data
        X_val, y_val = self.dataset.validation_data

        train_loader = self.create_dataloader(X_train, y_train)
        val_loader = self.create_dataloader(X_val, y_val, shuffle=False)

        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            train_losses = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            val_loss = self.evaluate(val_loader)
            logging.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']}, "
                f"Train Loss: {np.mean(train_losses):.4f}, Validation Loss: {val_loss:.4f}"
            )

            # Arret précoce (Early Stopping)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint()
            else:
                self.epochs_no_improve +=1
                if self.epochs_no_improve >= self.config['training']['patience']:
                    logging.info("Arrêt précoce déclenché")
                    break

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                losses.append(loss.item())
        return np.mean(losses)

    def create_dataloader(self, X, y, shuffle=True):
        dataset = TensorDataset(
            torch.Tensor(X), torch.Tensor(y)
        )
        return DataLoader(
            dataset, batch_size=self.config['training']['batch_size'],
            shuffle=shuffle, num_workers=0
        )

    def save_checkpoint(self):
        model_dir = self.config['output']['model_dir']
        ensure_dir(model_dir)
        model_path = os.path.join(model_dir, f"{self.dataset.ticker}_model.pth")
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Checkpoint du modèle sauvegardé à {model_path}")

    def test(self):
        X_test, y_test = self.dataset.test_data
        test_loader = self.create_dataloader(X_test, y_test, shuffle=False)
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.numpy())
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals)
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        logging.info(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        return predictions, actuals
