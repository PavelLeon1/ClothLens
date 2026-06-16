# ClothLens

Система поиска визуально схожей одежды на основе сегментации и семантических
эмбеддингов.

Подробное человеко-читаемое объяснение архитектуры, пайплайна, индексации и
demo-данных находится в [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Состояние проекта

Проект разрабатывается в четыре этапа:

1. Основа проекта и подготовка контура обучения U-Net.
2. Рабочий поиск на предобученном сегментаторе, FashionCLIP и Qdrant.
3. FastAPI и простой веб-интерфейс.
4. Оценка качества и сравнение SegFormer с U-Net.

На первом этапе U-Net не обучается. Код обучения и обработки DeepFashion2
готовится заранее, чтобы позднее получить checkpoint на RTX 5070 или T4 без
изменения приложения.

## Локальная среда

Все зависимости устанавливаются только в виртуальное окружение:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Запуск тестов:

```powershell
python -m pytest
```

ML-зависимости устанавливаются отдельно и также только в `venv`:

```powershell
python -m pip install -e ".[ml]"
```

## Подготовленный контур U-Net

Архитектура U-Net использует ImageNet-предобученный ResNet34 и восемь выходных
классов. На первом этапе обучение не запускается.

Ожидаемая структура DeepFashion2:

```text
data/raw/DeepFashion2/
├── train/
│   ├── image/
│   └── annos/
└── validation/
    ├── image/
    └── annos/
```

После установки ML-зависимостей обучение запускается явно:

```powershell
python -m clothing_search.segmentation.train --config configs/train.yaml
```

DeepFashion2 содержит категории `top`, `bottom`, `dress` и `outerwear`, но не
содержит разметку обуви, сумок и аксессуаров. Поэтому будущие метрики U-Net на
этом датасете будут рассчитываться только для представленных классов.

## Рабочий поиск

Текущий pipeline использует:

- `mattmdjaga/segformer_b2_clothes` для готовой сегментации;
- `patrickjohncyh/fashion-clip` для 512-мерных эмбеддингов;
- Qdrant в persistent local mode, без Docker и отдельного сервера.

Установка поисковых зависимостей выполняется только внутри `venv`:

```powershell
python -m pip install -e ".[search]"
```

Каталог имеет следующую структуру:

```text
data/catalog/
├── images/
│   ├── sku-001.jpg
│   └── sku-002.jpg
└── metadata.json
```

Пример `metadata.json`:

```json
[
  {
    "item_id": "sku-001",
    "category": "top",
    "brand": "Example",
    "color": "blue",
    "price": 3990,
    "image_url": "https://example.test/sku-001.jpg"
  }
]
```

Построение локального индекса:

```powershell
python scripts/build_index.py --catalog data/catalog
```

Индекс сохраняется в `data/qdrant`, который исключён из Git. При первом
использовании Transformers загрузит веса моделей в пользовательский кэш
Hugging Face; сами веса также не добавляются в репозиторий.

## FastAPI и web-интерфейс

Для запуска API и простой страницы поиска установите runtime-зависимости в
локальное окружение:

```powershell
python -m pip install -e ".[dev,search,api]"
```

Перед поиском каталог должен быть проиндексирован:

```powershell
python scripts/build_index.py --catalog data/catalog
```

Запуск приложения:

```powershell
python -m clothing_search.api
```

После запуска страница доступна по адресу `http://127.0.0.1:8000/`.
Основные endpoints:

- `GET /health` — проверка доступности сервиса;
- `POST /search` — multipart-загрузка изображения, `category` и `top_k`;
- `POST /catalog/add` — добавление изображения товара в локальный каталог и
  индекс Qdrant.

## Оценка качества поиска

На текущем этапе оценивается рабочий pipeline с предобученным SegFormer.
U-Net не обучается и не участвует в сравнении, пока не будет получен checkpoint.

Формат manifest для запросов:

```json
[
  {
    "query_id": "query-001",
    "image_path": "data/evaluation/query-001.jpg",
    "category": "top",
    "relevant_item_ids": ["sku-001", "sku-014"]
  }
]
```

Запуск оценки:

```powershell
python scripts/evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --config configs/app.yaml `
  --top-k 10 `
  --output results/segformer-report.json `
  --pretty
```

Отчёт содержит `Precision@K`, `Recall@K`, `mAP@K`, latency summary и
метрики по каждому query. Папки `data/` и `results/` исключены из Git.
