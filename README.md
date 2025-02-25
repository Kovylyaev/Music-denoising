# Music_MMLS

## Веха 1
### Планы:
Договориться про то, по каким метрикам принимаем бейзлайн и последующие модели + про первичную работу с данными (сюда входит и поиск нужных датасетов, и скраппинг если нужного датасета нет, и разметка).

### Результаты:

#### Данные
Выбран датасет чистой музыки: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
Выбран датасет шума: https://www.kaggle.com/datasets/mlneo07/random-noise-audio

Также, если будет необходимо больше данных, найдены: музыка - https://www.kaggle.com/datasets/imsparsh/musicnet-dataset и https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset, шум - https://www.kaggle.com/datasets/javohirtoshqorgonov/noise-audio-data

В ноубуке находятся скрипты по скачиванию и первичной обработке данных + реализация класса датасета (__getitem__ ->  mel-spec-noised-audio + mel-spec-clean-audio).

Изначально был выбран другой датасет, но он не подошёл(

#### Метрики
...


### Почему выбрана одна метрика, а не другая + обзор метрик
...
