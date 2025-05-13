# 🎵 Music Denoising

**Music Denoising** — это проект по удалению шума из музыкальных записей с использованием методов глубокого обучения

##  Начало работы

### Установка зависимостей

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/Kovylyaev/Music-denoising.git
   cd Music-denoising
   ```

2. Создайте и активируйте виртуальное окружение:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Windows: venv\Scripts\activate
   ```

3. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

### Структура проекта

* `Music_MMLS/` — основной код проекта
* `configs/` — конфигурационные файлы
* `notebooks/` — Jupyter-ноутбуки для экспериментов и анализа
* `tests/` — тесты
* `Dockerfile` — для контейнеризации проекта
* `requirements.txt` — список зависимостей

## Обучение модели

Для обучения модели запустите `notebooks/metals-demo.ipynb`

Метрики:

* PESQ
* STOI
* SNR

## Использование Docker

1. Соберите Docker-образ:

```bash
docker build -t music-denoising
```

2. Запустите контейнер:

```bash
docker run -it --rm -v $(pwd):/app music-denoising
```


## TODO: Тестирование

Для запуска тестов:

```bash
pytest tests/
```

## Лицензия

Этот проект лицензирован по лицензии MIT

---

Если у вас есть вопросы или предложения, откройте issue в репозитории
