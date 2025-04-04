import torch
import numpy as np
import torchaudio

def snr(clean_batch, noisy_batch):
    noise_batch = noisy_batch - clean_batch
    power_clean = torch.sum(clean_batch ** 2, dim=[1, 2, 3])
    power_noise = torch.sum(noise_batch ** 2, dim=[1, 2, 3])
    snr_batch = 10 * torch.log10(power_clean / power_noise)
    return torch.mean(snr_batch)

def sdr(clean_batch, estimate_batch):
    distortion_batch = clean_batch - estimate_batch
    power_clean = torch.sum(clean_batch ** 2, dim=[1, 2, 3])
    power_distortion = torch.sum(distortion_batch ** 2, dim=[1, 2, 3])
    sdr_batch = 10 * torch.log10(power_clean / power_distortion)
    return torch.mean(sdr_batch)

def lsd(clean_batch, estimate_batch, eps=1e-5):
    clean_batch = clean_batch ** 2
    estimate_batch = estimate_batch ** 2
    log_clean = torch.log10(clean_batch + eps)
    log_estimate = torch.log10(estimate_batch + eps)
    squared_diff = (log_clean - log_estimate) ** 2
    lsd_per_frame = torch.sqrt(torch.mean(squared_diff, dim=2))
    lsd_value = torch.mean(lsd_per_frame)
    return lsd_value

def si_snr(clean_db, noisy_db, eps=1e-8):
    """
    Вычисляет Scale-Invariant SNR (SI-SNR) для каждого примера в батче.
    Принимает dB мел-спектрограммы и возвращает среднее значение SI-SNR.
    """
    clean_linear = 10 ** (clean_db / 10)
    noisy_linear = 10 ** (noisy_db / 10)
    batch_size = clean_linear.shape[0]
    clean_flat = clean_linear.view(batch_size, -1)
    noisy_flat = noisy_linear.view(batch_size, -1)
    s = clean_flat - torch.mean(clean_flat, dim=1, keepdim=True)
    s_hat = noisy_flat - torch.mean(noisy_flat, dim=1, keepdim=True)
    s_target = torch.sum(s_hat * s, dim=1, keepdim=True) * s / (torch.sum(s ** 2, dim=1, keepdim=True) + eps)
    e = s_hat - s_target
    si_snr_val = 10 * torch.log10((torch.sum(s_target ** 2, dim=1) + eps) / (torch.sum(e ** 2, dim=1) + eps))
    return torch.mean(si_snr_val)

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
