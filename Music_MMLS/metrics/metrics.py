import numpy as np
import torch
import torchaudio
from torchmetrics import Metric


class SNR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, clean_batch, noisy_batch):
        if clean_batch.shape != noisy_batch.shape:
            raise ValueError("clean_batch and noisy_batch must have the same shape")

        noise_batch = noisy_batch - clean_batch
        self.power_clean = torch.sum(clean_batch ** 2, dim=[1, 2, 3])
        self.power_noise = torch.sum(noise_batch ** 2, dim=[1, 2, 3])

    def compute(self):
        return torch.mean(10 * torch.log10(self.power_clean / self.power_noise))
    

class SDR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, clean_batch, estimate_batch):
        if clean_batch.shape != estimate_batch.shape:
            raise ValueError("clean_batch and estimate_batch must have the same shape")

        distortion_batch = clean_batch - estimate_batch
        self.power_clean = torch.sum(clean_batch ** 2, dim=[1, 2, 3])
        self.power_distortion = torch.sum(distortion_batch ** 2, dim=[1, 2, 3])

    def compute(self):
        return torch.mean(10 * torch.log10(self.power_clean / self.power_distortion))
    

class LSD(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, clean_batch, estimate_batch, eps=1e-5):
        if clean_batch.shape != estimate_batch.shape:
            raise ValueError("clean_batch and estimate_batch must have the same shape")

        clean_batch = clean_batch ** 2
        estimate_batch = estimate_batch ** 2
        log_clean = torch.log10(clean_batch + eps)
        log_estimate = torch.log10(estimate_batch + eps)
        self.squared_diff = (log_clean - log_estimate) ** 2

    def compute(self):
        return torch.mean(torch.sqrt(torch.mean(self.squared_diff, dim=2)))


class SI_SNR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, clean_db, noisy_db, eps=1e-8):
        if clean_db.shape != noisy_db.shape:
            raise ValueError("clean_db and noisy_db must have the same shape")

        clean_linear = 10 ** (clean_db / 10)
        noisy_linear = 10 ** (noisy_db / 10)
        batch_size = clean_linear.shape[0]
        clean_flat = clean_linear.view(batch_size, -1)
        noisy_flat = noisy_linear.view(batch_size, -1)
        s = clean_flat - torch.mean(clean_flat, dim=1, keepdim=True)
        s_hat = noisy_flat - torch.mean(noisy_flat, dim=1, keepdim=True)
        s_target = torch.sum(s_hat * s, dim=1, keepdim=True) * s / (torch.sum(s ** 2, dim=1, keepdim=True) + eps)
        e = s_hat - s_target
        self.si_snr_val = 10 * torch.log10((torch.sum(s_target ** 2, dim=1) + eps) / (torch.sum(e ** 2, dim=1) + eps))

    def compute(self):
        return torch.mean(self.si_snr_val)
    

def invert_spectrogram(spectrogram_batch):
    """
    Преобразует спектрограмму обратно в аудиоволну с использованием Griffin-Lim.
    """
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=254)
    batch_waveforms = []
    batch_size = spectrogram_batch.shape[0]
    for i in range(batch_size):
        spec = spectrogram_batch[i, 0]
        waveform = griffin_lim(spec)
        batch_waveforms.append(waveform)
    return batch_waveforms

# def compute_stoi(clean_signal, enhanced_signal, fs=16000):
#     """
#     Вычисляет STOI (Short-Time Objective Intelligibility).
#     """
#     from pystoi import stoi  # Импортируем локально, чтобы не зависеть от глобальных настроек
#     return stoi(clean_signal, enhanced_signal, fs, extended=False)

# def _stoi(clean_waveforms, enhanced_waveforms, fs=16000):
#     stoi_scores = []
#     for clean_sig, enh_sig in zip(clean_waveforms, enhanced_waveforms):
#         score = compute_stoi(clean_sig, enh_sig, fs)
#         stoi_scores.append(score)
#     return stoi_scores

# def stoi_score(clean_spectrogram_batch, enhanced_spectrogram_batch):
#     griffin_lim = torchaudio.transforms.GriffinLim(n_fft=254)
#     clean_waveforms = invert_spectrogram(clean_spectrogram_batch)
#     enhanced_waveforms = invert_spectrogram(enhanced_spectrogram_batch)
#     stoi_batch = _stoi(clean_waveforms, enhanced_waveforms)
#     return np.mean(stoi_batch)
