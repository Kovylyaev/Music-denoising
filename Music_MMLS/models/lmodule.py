import lightning as L
import torchmetrics

from Music_MMLS.metrics.metrics import SNR, LSD, SI_SNR
from Music_MMLS.training.gets import get_model, get_optimizer, get_criterion, get_scheduler

class MusicModelModule(L.LightningModule):
    def __init__(self, model_config, training_config):
        """
            Инициализирует модель.
            
            Args:
                model_config - конфигурация модели
                training_config - конфигурация обучения
        """
        super().__init__()

        self.model = get_model(model_config.model)(n_channels=model_config.n_channels)
        self.optimizer = get_optimizer(training_config.optimizer)(self.model.parameters(), lr=training_config.learning_rate)
        self.criterion = get_criterion(training_config.criterion)()
        self.scheduler = get_scheduler(training_config.scheduler)(self.optimizer)

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "snr": SNR(),
                "lsd": LSD(),
                "si_snr": SI_SNR(),
            },
            prefix="train_"
        )
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x):
        """
            Проход вперёд для инференса

            Args:
                x - входные данные
        """
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        """
            Шаг обучения
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)

        batch_value = self.train_metrics(preds, y)
        self.log_dict(batch_value)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
            Шаг тестирования
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)

        batch_value = self.test_metrics(preds, y)
        self.log_dict(batch_value)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()
    
    def on_test_epoch_end(self):
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        """
        Возвращает оптимизатор и, если задан, планировщик скорости обучения.
        """
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
