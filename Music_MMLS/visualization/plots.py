import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd

def plot_waveform(waveform, sample_rate, suptitle="waveform"):
    """
    Строит график временной области аудиосигнала.
    
    Args:
        waveform (torch.Tensor): Аудиосигнал (тензор).
        sample_rate (int): Частота дискретизации.
        suptitle (str): Заголовок графика.
    """
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(1, 1, figsize=(10, 2))
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(suptitle)
    plt.show()

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    """
    Отображает спектрограмму.
    
    Args:
        specgram (array-like): Спектрограмма для отображения.
        title (str, optional): Заголовок графика.
        ylabel (str, optional): Название оси Y.
        ax (matplotlib.axes.Axes, optional): Если указан, строится на заданном ax.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")
    plt.show()

def plot_metrics(df_train, df_val):
    """
    Строит графики метрик для обучающей и валидационной выборок.
    
    Args:
        df_train (pd.DataFrame): DataFrame с метриками для обучающей выборки.
        df_val (pd.DataFrame): DataFrame с метриками для валидационной выборки.
    """
    for col in df_train.columns:
        # Преобразуем данные в длинный формат
        df_train_temp = pd.DataFrame({
            'Iteration': df_train.index,
            'Accuracy': df_train[col],
            'Dataset': 'Train'
        })
        df_val_temp = pd.DataFrame({
            'Iteration': df_val.index,
            'Accuracy': df_val[col],
            'Dataset': 'Validation'
        })
        df_long = pd.concat([df_train_temp, df_val_temp])
        
        # Строим график
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_long, x='Iteration', y='Accuracy', hue='Dataset', marker="o")
        plt.title(f'{col}')
        plt.xlabel('Iteration')
        plt.ylabel('Metric value')
        plt.grid()
        plt.tight_layout()
        plt.show()
