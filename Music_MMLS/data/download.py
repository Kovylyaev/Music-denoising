import os
import kagglehub
import math
from pathlib import Path
from pydub import AudioSegment
import shutil
import sys

def download_data(dataset_path, out_path, dataset_path_add=None, audio_duration=None):
    """
    Скачивает данные с Kaggle и перемещает файлы в указанную директорию.
    
    Args:
        dataset_path (str): Имя датасета на Kaggle.
        out_path (str): Путь, куда нужно сохранить данные.
        dataset_path_add (str, optional): Дополнительный путь внутри скачанного архива.
        audio_duration (int, optional): Длительность аудио (в секундах) для разбиения файла.
    """
    if os.listdir(out_path):
        print(f'ℹ️ Папка {out_path} не пуста, возможно, датасет уже там')
        return

    path = kagglehub.dataset_download(dataset_path)
    path = shutil.move(path, 'content/sample_data/temp_Data')

    if dataset_path_add is not None:
        path = os.path.join(path, dataset_path_add)

    # Обходим папки внутри скачанного датасета
    for folder in os.listdir(path):
        for file_name in os.listdir(os.path.join(path, folder)):
            full_path = os.path.join(path, folder, file_name)

            if audio_duration is not None:
                sound = AudioSegment.from_file(full_path)
                num_segments = math.ceil(len(sound) / (audio_duration * 1000))

                for i in range(num_segments):
                    start_time = i * audio_duration * 1000
                    end_time = min((i + 1) * audio_duration * 1000, len(sound))
                    segment = sound[start_time:end_time]

                    segment.export(f"{out_path}/{file_name}_part_{i+1}.wav", format="wav")
            else:
                shutil.move(full_path, os.path.join(out_path, file_name))
                
    # Удаляем повреждённый файл, если он существует
    broken_file = 'content/sample_data/Data/all_records/jazz.00054.wav'
    if os.path.exists(broken_file):
        os.remove(broken_file)

    print(f'✅ Датасет {dataset_path} успешно скачан')
