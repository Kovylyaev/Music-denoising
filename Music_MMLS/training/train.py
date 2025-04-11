import time
import torch
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np
import wandb
from Music_MMLS.metrics.metrics import snr, lsd, si_snr

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None,
                 device=torch.device('cpu'), use_wandb=False, config=None):
        """
        Инициализирует тренер для обучения модели.
        
        Args:
            model (torch.nn.Module): Модель для обучения.
            optimizer (torch.optim.Optimizer): Оптимизатор.
            criterion: Функция потерь.
            scheduler (optional): Планировщик скорости обучения.
            device (torch.device): Устройство (CPU или GPU).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="music-mmls", config=config)
            wandb.watch(self.model, log="all", log_freq=10)

    @staticmethod
    def epoch_time(start_time, end_time):
        """Вычисляет время эпохи в минутах и секундах."""
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def calc_metrics(self, true_spec, pred_spec):
        """
        Вычисляет метрики качества, используя импортированные функции.
        
        Args:
            true_spec (torch.Tensor): Истинные значения.
            pred_spec (torch.Tensor): Предсказанные значения.
            
        Returns:
            dict: Словарь с рассчитанными метриками.
        """
        scores = {
            "snr": snr,
            "lsd": lsd,
            "si_snr": si_snr,
        }
        results = {}
        for name, func in scores.items():
            results[name] = func(true_spec, pred_spec).item()
        return results

    def __train_epoch(self, loader):
        """
        Выполняет обучение модели за одну эпоху.
        
        Args:
            loader: DataLoader для обучающей выборки.
            
        Returns:
            tuple: Среднее значение потерь и список метрик по батчам.
        """
        self.model.train()
        epoch_loss = 0
        epoch_metrics = []
        for specs_noisy, specs_clean in tqdm(loader, desc='Train'):
            specs_noisy = specs_noisy.to(self.device)
            specs_clean = specs_clean.to(self.device)

            self.optimizer.zero_grad()
            predicted_clean = self.model(specs_noisy)
            loss = self.criterion(predicted_clean, specs_clean)
            epoch_metrics.append(self.calc_metrics(specs_clean, predicted_clean))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader), epoch_metrics

    def evaluate_epoch(self, loader):
        """
        Выполняет валидацию модели за одну эпоху.
        
        Args:
            loader: DataLoader для валидационной выборки.
            
        Returns:
            tuple: Среднее значение потерь и список метрик по батчам.
        """
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = []
        with torch.no_grad():
            for specs_noisy, specs_clean in tqdm(loader, desc='Val'):
                specs_noisy = specs_noisy.to(self.device)
                specs_clean = specs_clean.to(self.device)
                predicted_clean = self.model(specs_noisy)
                loss = self.criterion(predicted_clean, specs_clean)
                epoch_metrics.append(self.calc_metrics(specs_clean, predicted_clean))
                epoch_loss += loss.item()
        return epoch_loss / len(loader), epoch_metrics

    def fit(self, train_loader, val_loader, num_epochs, save_best_path=None):
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_metrics = self.__train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate_epoch(val_loader)
            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best_path:
                    torch.save(self.model.state_dict(), f"{save_best_path}_model.pt")
                    torch.save(self.optimizer.state_dict(), f"{save_best_path}_optimizer.pt")

            clear_output(wait=True)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}")
            print(f"\tVal Loss: {val_loss:.3f}")
            print(f"\tTrain Metrics: {train_metrics}")
            print(f"\tVal Metrics: {val_metrics}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

                if self.use_wandb:
                    avg_train_metrics = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()}
                    avg_val_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
                    log_data = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"train_{k}": v for k, v in avg_train_metrics.items()},
                        **{f"val_{k}": v for k, v in avg_val_metrics.items()}
                    }
                    wandb.log(log_data)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics": train_metrics_history,
            "val_metrics": val_metrics_history,
        }

