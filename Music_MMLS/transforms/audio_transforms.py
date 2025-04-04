import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import v2


def do_db_mel_spec(waveform, sample_rate):
    """
    Вычисляет dB-мел-спектрограмму для заданного аудиосигнала.
    
    Args:
        waveform (torch.Tensor): Входной аудиосигнал.
        sample_rate (int): Частота дискретизации.
        
    Returns:
        torch.Tensor: dB-мел-спектрограмма.
    """
    db_mel = nn.Sequential(
        MelSpectrogram(sample_rate, n_fft=2000, win_length=2000),
        AmplitudeToDB()
    )
    return db_mel(waveform)

def pad(arr, max_len, val):
    """
    Дополняет тензор до заданной длины значением val.
    
    Args:
        arr (torch.Tensor): Входной тензор.
        max_len (int): Желаемая длина (по первому измерению).
        val: Значение для дополнения.
        
    Returns:
        torch.Tensor: Дополненный тензор с сохранением размерности.
    """
    arr = torch.squeeze(arr)
    if arr.shape[0] >= max_len:
        return torch.unsqueeze(arr[:max_len], 0)
    
    padding = torch.tensor([val] * (max_len - len(arr)))
    arr = torch.cat((arr, padding), 0)
    return torch.unsqueeze(arr, 0)

def transform(pic):
    pic = (pic - pic.min()) / (pic.max() - pic.min())

    trans = v2.Compose([
        v2.Normalize(mean=[0.5], std=[0.225])
    ])

    return trans(pic)