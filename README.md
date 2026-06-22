# ClothLens

Система поиска визуально схожей одежды на основе сегментации и семантических
эмбеддингов.

Подробное человеко-читаемое объяснение архитектуры, пайплайна, индексации и
demo-данных находится в [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Возможности

ClothLens - это готовое локальное приложение для визуального поиска похожей
одежды. Проект объединяет сегментацию одежды, эмбеддинги FashionCLIP, локальный
векторный поиск Qdrant и web-интерфейс на FastAPI.

Поддерживаются два режима сегментации:

- **SegFormer** - предобученный сегментатор для всех категорий интерфейса.
- **Гибридный режим** - U-Net используется для `top`, `bottom`, `dress`,
  `outerwear`, а SegFormer автоматически остаётся для `shoes`, `bag`,
  `accessories`.
  Если U-Net в гибридном режиме не находит выбранный класс или выделяет для
  него слишком маленькую область, приложение автоматически откатывается на
  SegFormer и показывает это в статусе поиска.

Обученная U-Net использует ResNet34 encoder и checkpoint
`models/unet_best.ckpt`. На validation subset DeepFashion2 получено
`val_miou = 0.7342`.

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

## U-Net и DeepFashion2

Архитектура U-Net использует ImageNet-предобученный ResNet34 и восемь выходных
классов приложения. Обучение выполняется на DeepFashion2 subset для четырёх
классов: `top`, `bottom`, `dress`, `outerwear`.

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

Подготовка переносимой подвыборки:

```powershell
python scripts\prepare_deepfashion2_subset.py `
  --source data\raw\DeepFashion2 `
  --output data\raw\DeepFashion2_subset `
  --train-size 30000 `
  --validation-size 5000 `
  --archive data\raw\DeepFashion2_subset.zip `
  --clean
```

Обучение на RTX 5070:

```powershell
.\scripts\setup_training_env.ps1 -Profile rtx5070
venv\Scripts\python.exe -m clothing_search.segmentation.train --config configs/train_rtx5070.yaml
```

После обучения скопируйте лучший checkpoint в путь приложения:

```powershell
Copy-Item data\unet_training_result\unet_rtx5070_best.ckpt models\unet_best.ckpt
```

DeepFashion2 не содержит разметку обуви, сумок и аксессуаров. Поэтому U-Net
применяется только к четырём обученным классам, а гибридный режим корректно
переключается на SegFormer для остальных категорий.

Дополнительно в гибридном режиме есть защитный fallback: если U-Net путает
класс и выбранная категория занимает подозрительно маленькую часть маски,
поиск повторно использует SegFormer. Это защищает embedding-поиск от ситуации,
когда в вектор попадает маленький случайный фрагмент вместо всей вещи.

## Поиск похожей одежды

Pipeline использует:

- `mattmdjaga/segformer_b2_clothes` как универсальный сегментатор;
- обученную U-Net для четырёх классов DeepFashion2;
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
В форме поиска можно выбрать режим сегментации: SegFormer-only или гибридный
режим U-Net + SegFormer.
Основные endpoints:

- `GET /health` — проверка доступности сервиса;
- `POST /search` — multipart-загрузка изображения, `category` и `top_k`;
- `POST /catalog/add` — добавление изображения товара в локальный каталог и
  индекс Qdrant.

## Оценка качества

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

Оценка SegFormer-only:

```powershell
python scripts/evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --config configs/app.yaml `
  --top-k 10 `
  --output results/segformer-report.json `
  --pretty
```

Оценка U-Net backend:

```powershell
python scripts/evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --config configs/app_unet.yaml `
  --top-k 10 `
  --output results/unet-report.json `
  --pretty
```

Сравнение отчётов:

```powershell
python scripts\compare_search_reports.py `
  --baseline baselines\segformer_demo_report.json `
  --current results\unet-report.json `
  --baseline-name SegFormer `
  --current-name U-Net `
  --output results\segformer-vs-unet.md
```

Отчёт содержит `Precision@K`, `Recall@K`, `mAP@K`, latency summary и
метрики по каждому query. Папки `data/` и `results/` исключены из Git.
