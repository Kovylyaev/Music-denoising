import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import clamp
import torchaudio
from ..transforms.audio_transforms import pad, do_db_mel_spec, transform


def collate_fn(batch):
    noise_records, clean_records = zip(*batch)

    clean_records = pad_sequence(clean_records, batch_first=True)
    noise_records = pad_sequence(noise_records, batch_first=True)  # (batch, time)

    clean_specs, noise_specs = [], []
    for clean, noise in zip(clean_records, noise_records):
        with_noise = clamp(clean + noise, min=-1, max=1)

        # Функция do_db_mel_spec и transform также должна быть определена отдельно
        noise_specs.append(transform(do_db_mel_spec(with_noise)))
        clean_specs.append(transform(do_db_mel_spec(clean)))

    clean_specs = torch.stack(clean_specs) 
    noise_specs = torch.stack(noise_specs)
    
    return noise_specs, clean_specs