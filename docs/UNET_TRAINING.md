# U-Net Training Guide

Этот документ нужен, чтобы быстро запустить обучение U-Net на домашней RTX 5070
или на серверной NVIDIA T4 и потом сравнить результат с текущим SegFormer.

## 1. Что уже подготовлено

- U-Net на `segmentation-models-pytorch` с encoder `resnet34`.
- DeepFashion2 dataset adapter.
- Конфиги:
  - `configs/train_rtx5070.yaml`
  - `configs/train_t4.yaml`
  - `configs/app_unet.yaml`
- Скрипты установки окружения:
  - `scripts/setup_training_env.ps1`
  - `scripts/setup_training_env.sh`
- Сравнение отчётов:
  - `baselines/segformer_demo_report.json`
  - `scripts/compare_search_reports.py`

## 2. Датасет

Ожидаемая структура DeepFashion2:

```text
data/raw/DeepFashion2/
+-- train/
|   +-- image/
|   +-- annos/
+-- validation/
    +-- image/
    +-- annos/
```

В Git датасет не добавляется. На домашнем ПК и сервере его нужно положить
отдельно в эту же структуру.

DeepFashion2 в проекте используется только для категорий:

- `top`
- `bottom`
- `dress`
- `outerwear`

Категории `shoes`, `bag`, `accessories` остаются в общем интерфейсе приложения,
но DeepFashion2 их не размечает.

## 3. RTX 5070, Windows

Из корня репозитория:

```powershell
.\scripts\setup_training_env.ps1 -Profile rtx5070
```

Скрипт создаст локальный `venv`, поставит PyTorch CUDA 13.0:

```powershell
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Затем поставит ML-зависимости проекта:

```powershell
venv\Scripts\python.exe -m pip install -e ".[ml,dev]"
```

Проверить датасет и размеры без старта обучения:

```powershell
venv\Scripts\python.exe -m clothing_search.segmentation.train `
  --config configs/train_rtx5070.yaml `
  --dry-run
```

Запустить обучение:

```powershell
venv\Scripts\python.exe -m clothing_search.segmentation.train `
  --config configs/train_rtx5070.yaml
```

## 4. NVIDIA T4, Linux-сервер

Если репозиторий переносится zip-файлом, распакуй его на сервере, положи
DeepFashion2 в `data/raw/DeepFashion2`, затем:

```bash
bash scripts/setup_training_env.sh t4
```

Для T4 выбран PyTorch CUDA 12.8:

```bash
venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Dry-run:

```bash
venv/bin/python -m clothing_search.segmentation.train \
  --config configs/train_t4.yaml \
  --dry-run
```

Обучение:

```bash
venv/bin/python -m clothing_search.segmentation.train \
  --config configs/train_t4.yaml
```

Если серверный NVIDIA driver слишком старый для CUDA 12.8, проверь команду на
официальной странице PyTorch и замени индекс на `cu126`.

## 5. Что будет видно во время обучения

Перед стартом печатается summary:

- путь к датасету;
- image size;
- batch size;
- train/validation size;
- steps per epoch;
- total train steps;
- max epochs.

Во время обучения Lightning показывает progress bar. После каждой эпохи
печатается длительность эпохи и примерный ETA. В progress bar/logs будут:

- `train_loss`
- `val_loss`
- `val_miou`
- learning rate

Логи сохраняются в:

```text
results/training_logs/
```

Чекпоинты:

```text
models/unet_rtx5070_best.ckpt
models/unet_t4_best.ckpt
models/last.ckpt
```

`models/` и `results/` исключены из Git.

## 6. Resume

Если обучение прервалось:

```powershell
venv\Scripts\python.exe -m clothing_search.segmentation.train `
  --config configs/train_rtx5070.yaml `
  --resume-from-checkpoint models\last.ckpt
```

На Linux:

```bash
venv/bin/python -m clothing_search.segmentation.train \
  --config configs/train_t4.yaml \
  --resume-from-checkpoint models/last.ckpt
```

## 7. Быстрая smoke-проверка

Чтобы проверить весь training loop на 1-2 batch без долгого обучения:

```powershell
venv\Scripts\python.exe -m clothing_search.segmentation.train `
  --config configs/train_rtx5070.yaml `
  --max-epochs 1 `
  --limit-train-batches 2 `
  --limit-val-batches 2
```

## 8. Подключить обученный U-Net к evaluation

После обучения скопируй лучший checkpoint в путь из `configs/app_unet.yaml`:

```text
models/unet_best.ckpt
```

Построй или перенеси индекс каталога `data/qdrant`, затем запусти evaluation:

```powershell
venv\Scripts\python.exe scripts\evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --config configs/app_unet.yaml `
  --top-k 2 `
  --output results/unet-report.json `
  --pretty
```

Сравнить с текущим SegFormer baseline:

```powershell
venv\Scripts\python.exe scripts\compare_search_reports.py `
  --baseline baselines/segformer_demo_report.json `
  --current results/unet-report.json `
  --baseline-name SegFormer `
  --current-name U-Net `
  --output results/segformer-vs-unet.md
```

На сервер не нужно переносить сам SegFormer, если есть
`baselines/segformer_demo_report.json`.

## 9. Источники по PyTorch CUDA

- Официальный PyTorch install selector:
  <https://pytorch.org/get-started/locally/>
- PyTorch CUDA 13.0 wheel index:
  <https://download.pytorch.org/whl/cu130/torch/>
- PyTorch CUDA 12.8 wheel index:
  <https://download.pytorch.org/whl/cu128/torch/>

На 18 июня 2026 PyTorch install selector показывает CUDA 12.8 как стабильный
вариант, а `cu130` доступен через официальный wheel index PyTorch.
