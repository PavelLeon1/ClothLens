# ClothLens: понятное объяснение проекта

Этот файл написан как личная шпаргалка по проекту. Его можно читать сверху
вниз, чтобы понять, что уже сделано, как всё связано и что говорить при
демонстрации.

## 1. Что делает проект

ClothLens - это система поиска похожей одежды по изображению.

Пользователь загружает фото, выбирает категорию одежды, например `top` или
`dress`, а система:

1. Находит нужную область одежды на изображении.
2. Вырезает эту область.
3. Превращает вырезанную картинку в embedding-вектор.
4. Ищет похожие embedding-векторы в локальной базе Qdrant.
5. Возвращает похожие товары из каталога.

Тема практики звучит как система поиска схожей одежды на основе U-Net и
методов эмбеддинга. В проекте U-Net подготовлен как академический backend для
будущего обучения, но рабочая демонстрационная версия сейчас использует
предобученный SegFormer. Это осознанное решение: сначала сделано рабочее ПО, а
U-Net можно обучить и сравнить позже.

## 2. Главная идея пайплайна

Упрощённая схема:

```text
Фото пользователя
      |
      v
SegFormer segmentation
      |
      v
Маска категории одежды
      |
      v
Crop нужной вещи из исходного изображения
      |
      v
FashionCLIP embedding
      |
      v
Qdrant vector search
      |
      v
Похожие товары из каталога
```

Самые важные понятия:

- **Сегментация** - модель определяет, какие пиксели относятся к одежде.
- **Маска** - двумерная карта, где каждому пикселю назначен класс: `top`,
  `bottom`, `dress`, `shoes`, `bag` и так далее.
- **Crop** - вырезанный прямоугольник с нужной вещью.
- **Embedding** - числовой вектор, который описывает смысл и внешний вид
  картинки.
- **Vector search** - поиск ближайших векторов. Если два изображения похожи,
  их embedding-векторы обычно находятся рядом.

## 3. Что значит "проиндексировать каталог"

Каталог изначально состоит из обычных файлов:

```text
data/catalog/
├── images/
│   ├── demo-top-blue-shirt.jpg
│   └── ...
└── metadata.json
```

Для человека этого достаточно: мы видим картинки и описание. Но для быстрого
поиска системе нужно заранее подготовить catalog index.

Индексация означает:

1. Открыть каждую картинку товара.
2. Пропустить её через FashionCLIP.
3. Получить embedding-вектор размерности 512.
4. Сохранить этот вектор в Qdrant.
5. Прикрепить к вектору metadata: `item_id`, `category`, `brand`, `color`,
   `name`.

После индексации Qdrant хранит не сами картинки, а их числовые представления.
Когда приходит query, система считает embedding query-картинки и спрашивает у
Qdrant: "какие векторы ближе всего к этому?". Так поиск работает быстрее и
правильнее, чем ручной перебор файлов.

Команда индексации:

```powershell
python scripts/build_index.py --catalog data/catalog
```

Это нужно, когда каталог уже лежит на диске как папка `data/catalog/images` и
файл `metadata.json`. Для обычного пользователя есть более простой путь:
запустить web-приложение и загрузить изображения через форму
`Add Images To Catalog`. В этом режиме система сама:

1. сохраняет каждую картинку в `data/catalog/images`;
2. создаёт безопасный `item_id` из имени файла или заданного префикса;
3. дописывает запись в `metadata.json`;
4. считает embedding;
5. сразу добавляет embedding в Qdrant.

То есть после загрузки через UI отдельная команда `build_index.py` не нужна:
новые товары уже находятся в индексе.

После командной индексации или добавления товаров через UI появляется и
обновляется локальное хранилище:

```text
data/qdrant/
```

Эта папка не добавляется в Git, потому что это generated data.

## 4. Какие данные сейчас подготовлены

Сейчас в проекте создан demo-набор без скачивания из интернета:

```text
data/catalog/images/
data/catalog/metadata.json
data/evaluation/queries/
data/evaluation/manifest.json
results/demo-segformer-report.json
```

Каталог содержит 12 синтетических изображений:

- `top`: синяя и красная рубашка;
- `bottom`: джинсы и чёрные брюки;
- `dress`: зелёное и фиолетовое платье;
- `shoes`: белая и чёрная обувь;
- `bag`: коричневая и красная сумка;
- `accessories`: жёлтая шляпа и оранжевый шарф.

Эти картинки нужны не для научной оценки качества модели, а для демонстрации
механики проекта: структура каталога, metadata, embeddings, Qdrant index,
поиск, отчёт.

Почему synthetic demo полезен:

- работает без интернета;
- воспроизводится одной командой;
- не зависит от авторских прав на фото;
- показывает, как устроен pipeline.

Почему synthetic demo ограничен:

- SegFormer обучался на реальных изображениях людей и одежды;
- нарисованные предметы он может распознавать хуже;
- для настоящего отчёта лучше заменить demo-картинки реальными фото.

Генерация demo-данных:

```powershell
python scripts/create_demo_data.py --root data
```

## 5. Как устроены основные модули

### Конфигурация

Файл:

```text
configs/app.yaml
```

Он задаёт:

- какой сегментатор использовать;
- какую модель embedding брать;
- где лежит Qdrant;
- сколько результатов возвращать по умолчанию.

Код загрузки:

```text
src/clothing_search/config.py
```

### Сегментация

Базовый интерфейс:

```text
src/clothing_search/segmentation/base.py
```

Рабочий backend:

```text
src/clothing_search/segmentation/segformer.py
```

Подготовленный U-Net:

```text
src/clothing_search/segmentation/unet.py
```

Важный момент: SegFormer возвращает свои классы, например `Upper-clothes`,
`Pants`, `Bag`. Код сворачивает их в классы приложения:

```text
background, top, bottom, dress, outerwear, shoes, bag, accessories
```

### Crop

Файл:

```text
src/clothing_search/segmentation/crop.py
```

Он берёт маску и категорию, находит bounding box нужной области и вырезает её
из исходного изображения. Это важно: embedding считается не по всей фотографии,
а по конкретной вещи.

### Embeddings

Файл:

```text
src/clothing_search/embeddings/encoder.py
```

Используется `patrickjohncyh/fashion-clip`. На выходе получается
512-мерный нормализованный вектор.

Нормализация нужна, чтобы cosine similarity работал предсказуемо.

### Search storage

Файл:

```text
src/clothing_search/search/qdrant_store.py
```

Qdrant хранит embedding-векторы и metadata. Для каждого товара используется
стабильный UUID, полученный из `item_id`. Поэтому повторная индексация товара
обновляет существующую точку, а не создаёт хаос из дублей.

### Pipeline

Файл:

```text
src/clothing_search/pipeline.py
```

`SearchPipeline.search()` связывает всё вместе:

1. `segmenter.segment(image)`
2. `crop_category(...)`
3. `encoder.encode(crop.image)`
4. `store.search(vector, category, top_k)`
5. `SearchResponse`

### API и web UI

Файлы:

```text
src/clothing_search/api/app.py
src/clothing_search/api/templates/index.html
src/clothing_search/api/static/app.js
src/clothing_search/api/static/styles.css
```

Запуск:

```powershell
python -m clothing_search.api
```

После запуска:

```text
http://127.0.0.1:8000/
```

Endpoints:

- `GET /health` - проверить, что сервис жив;
- `POST /search` - найти похожую одежду по изображению;
- `POST /catalog/add` - добавить товар в каталог и сразу проиндексировать.
- `GET /catalog-images/{item_id}.jpg` - отдать локально сохранённую картинку
  товара для карточек результата.

На странице есть две основные формы:

1. `Search Similar Clothes` - загрузить query-фото, выбрать категорию и найти
   похожие товары.
2. `Add Images To Catalog` - загрузить один или несколько товаров в каталог без
   консоли. Для пачки файлов выбирается одна категория, а `brand`, `color`,
   `price` и `ID prefix` можно оставить пустыми. Если `ID prefix` пустой, UI
   создаёт `item_id` автоматически из имени файла и timestamp.

### Evaluation

Файлы:

```text
src/clothing_search/evaluation.py
scripts/evaluate_search.py
```

Evaluation manifest:

```json
[
  {
    "query_id": "query-top-blue-shirt",
    "image_path": "queries/query-top-blue-shirt.jpg",
    "category": "top",
    "relevant_item_ids": [
      "demo-top-blue-shirt",
      "demo-top-red-shirt"
    ]
  }
]
```

Команда:

```powershell
python scripts/evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --top-k 2 `
  --output results/demo-segformer-report.json `
  --pretty
```

Метрики:

- `Precision@K` - какая доля top-K результатов релевантна;
- `Recall@K` - какую долю всех релевантных товаров нашли;
- `AP@K` - насколько высоко стоят релевантные результаты;
- `mAP@K` - средний AP по всем query;
- latency - сколько секунд занимает обработка query.

## 6. Как запустить проект с нуля

Установить зависимости в venv:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,search,api]"
```

Создать demo-данные:

```powershell
python scripts/create_demo_data.py --root data
```

Построить индекс:

```powershell
python scripts/build_index.py --catalog data/catalog
```

Запустить web-приложение:

```powershell
python -m clothing_search.api
```

Открыть:

```text
http://127.0.0.1:8000/
```

После запуска можно добавлять новые изображения в каталог прямо на странице в
блоке `Add Images To Catalog`. Они сохраняются в `data/catalog/images`,
получают запись в `metadata.json` и сразу индексируются в Qdrant. Перезапускать
`build_index.py` для таких добавлений не нужно.

Прогнать evaluation:

```powershell
python scripts/evaluate_search.py `
  --manifest data/evaluation/manifest.json `
  --top-k 2 `
  --output results/demo-segformer-report.json `
  --pretty
```

Запустить тесты:

```powershell
python -m pytest --cov=clothing_search --cov-report=term-missing
```

## 7. Как заменить demo-данные на реальные

Есть два способа.

Первый способ, самый удобный для демонстрации: запустить приложение командой
`python -m clothing_search.api`, открыть `http://127.0.0.1:8000/` и загрузить
фотографии через форму `Add Images To Catalog`. Пользователь выбирает файлы,
категорию и при желании общие поля `brand`, `color`, `price`. Всё остальное
делается автоматически.

Второй способ нужен, если каталог готовится заранее как набор файлов. Тогда
нужно сохранить ту же структуру:

```text
data/catalog/
├── images/
│   ├── sku-001.jpg
│   ├── sku-002.jpg
│   └── ...
└── metadata.json
```

Пример metadata:

```json
[
  {
    "item_id": "sku-001",
    "category": "top",
    "brand": "Example",
    "color": "blue",
    "name": "Blue cotton shirt",
    "image_url": ""
  }
]
```

Правила:

- имя файла должно совпадать с `item_id`, например `sku-001.jpg`;
- категория должна быть одной из поддержанных;
- после замены каталога нужно снова выполнить `scripts/build_index.py`;
- если меняется evaluation-набор, нужно обновить `data/evaluation/manifest.json`.

Разница между способами простая: UI подходит для постепенного добавления
товаров и сам обновляет индекс, а `scripts/build_index.py` подходит для полной
пересборки каталога из уже подготовленной папки.

## 8. Что говорить про U-Net

U-Net в проекте не забыт. Сейчас сделано:

- подготовлена архитектура U-Net;
- подготовлена загрузка DeepFashion2;
- подготовлены transforms, metrics и training entrypoint;
- приложение построено так, чтобы сегментатор был заменяемым компонентом.

Почему сейчас используется SegFormer:

- он уже предобучен;
- быстро даёт рабочий segmentation backend;
- позволяет сначала показать полностью работающее приложение;
- позже его можно сравнить с обученным U-Net.

Корректная формулировка для защиты:

> В рамках практики реализована система поиска схожей одежды с модульной
> архитектурой сегментации. U-Net подготовлен как обучаемый backend, а текущая
> рабочая версия использует предобученный SegFormer для демонстрации полного
> прикладного пайплайна. После обучения U-Net обе модели можно сравнить по
> одинаковым retrieval-метрикам.

## 9. Что уже проверено

На последнем полноценном этапе:

- все тесты проходили;
- покрытие было выше 80%;
- Ruff проходил;
- зависимости были согласованы через `pip check`;
- данные, индекс, отчёты и веса моделей были исключены из Git.

После добавления demo-данных также проверено:

- demo-каталог создаётся скриптом;
- индекс Qdrant построен на 12 товарах;
- прямой vector search по синей рубашке возвращает синюю и красную рубашки;
- evaluation-report создаётся в `results/demo-segformer-report.json`;
- synthetic query `shoes` помечается ошибкой сегментации, что нормально для
  нарисованных данных и честно отражено в отчёте.

## 10. Короткое резюме

Проект состоит из четырёх слоёв:

1. **ML segmentation** - найти одежду на фото.
2. **Embedding** - превратить crop вещи в вектор.
3. **Vector search** - найти похожие векторы в Qdrant.
4. **API/UI/evaluation** - дать пользователю интерфейс и получить метрики.

Самое важное для понимания:

- индекс - это заранее посчитанные embedding-векторы каталога;
- Qdrant ищет похожие вещи по близости векторов;
- SegFormer сейчас делает рабочую сегментацию;
- U-Net подготовлен, но требует отдельного обучения;
- demo-данные нужны для воспроизводимой проверки, реальные фото можно заменить
  без изменения архитектуры.
