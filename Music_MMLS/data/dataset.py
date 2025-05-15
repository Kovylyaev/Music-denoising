import os
import torch
import torchaudio
from torchaudio.functional import resample
from torch import clamp
from .download import download_data 
from ..transforms.audio_transforms import pad, do_db_mel_spec, transform
from torchvision.transforms import v2


class Music_Dataset(torch.utils.data.Dataset):
    """
    Класс датасета для работы с музыкой, где осуществляется загрузка файлов с чистыми записями и шумом.
    """
    def __init__(self, size, clean_files, noise_files):
        """
        Инициализирует датасет.
        
        Args:
            size (int): Количество различных комбинаций (пара clean + noise).
            clean_files (iterable): Список путей к файлам чистой музыки.
            noise_files (iterable): Список путей к файлам шума.
        """
        self.size = size
        self.clean_music = list(clean_files)
        self.clean_num = len(self.clean_music)
        self.noise = list(noise_files)
        self.noise_num = len(self.noise)

    def __getitem__(self, idx: int):
        """
        Возвращает пару записей: чистую запись и запись с наложенным шумом.
        
        Args:
            idx (int): Индекс выборки.
            
        Returns:
            Tuple: (запись с шумом, чистая запись) после необходимых преобразований.
        """
        musc_ind = idx % (self.size // 10)

        clean_record, sample_rate_clean = torchaudio.load(self.clean_music[musc_ind])
        
        noise_ind = idx % 10
        noise_record, sample_rate_noise = torchaudio.load(self.noise[noise_ind])
        
        # Приведение частоты дискретизации
        clean_record = resample(clean_record, orig_freq=sample_rate_clean, new_freq=16000)
        noise_record = resample(noise_record, orig_freq=sample_rate_noise, new_freq=16000)
        
        return noise_record, clean_record

    def __len__(self):
        """
        Возвращает общее количество комбинаций.
        """
        return self.size
