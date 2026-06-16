const searchForm = document.querySelector('#search-form');
const statusBox = document.querySelector('#status');
const catalogForm = document.querySelector('#catalog-form');
const catalogStatus = document.querySelector('#catalog-status');
const catalogLog = document.querySelector('#catalog-log');
const cropImage = document.querySelector('#crop-image');
const maskImage = document.querySelector('#mask-image');
const cropPreviewCard = document.querySelector('#crop-preview-card');
const maskPreviewCard = document.querySelector('#mask-preview-card');
const results = document.querySelector('#results');

const CATEGORY_LABELS = {
  top: 'верх',
  bottom: 'низ',
  dress: 'платье',
  outerwear: 'верхняя одежда',
  shoes: 'обувь',
  bag: 'сумка',
  accessories: 'аксессуары',
};

function setStatus(message) {
  statusBox.textContent = message;
}

function categoryLabel(value) {
  return CATEGORY_LABELS[value] || value || 'неизвестно';
}

function friendlyErrorMessage(message) {
  if (!message) {
    return 'Произошла ошибка.';
  }
  if (message.includes('Uploaded file is not a valid image')) {
    return 'Загруженный файл не является изображением.';
  }
  if (message.includes('Unsupported clothing category')) {
    return 'Выбрана неподдерживаемая категория одежды.';
  }
  if (message.includes('was not found')) {
    return 'Одежда выбранной категории не найдена на изображении.';
  }
  if (message.includes('item_id contains unsupported characters')) {
    return 'ID товара содержит неподдерживаемые символы.';
  }
  return message;
}

function renderResults(items) {
  results.innerHTML = '';
  for (const item of items) {
    const card = document.createElement('article');
    card.className = 'result-card';
    const metadata = item.metadata || {};

    if (metadata.image_url) {
      const image = document.createElement('img');
      image.src = metadata.image_url;
      image.alt = `Товар каталога ${item.item_id}`;
      card.appendChild(image);
    }

    const title = document.createElement('h3');
    title.textContent = item.item_id;
    card.appendChild(title);

    const score = document.createElement('p');
    score.textContent = `Сходство: ${Number(item.score).toFixed(3)}`;
    card.appendChild(score);

    const category = document.createElement('p');
    category.textContent = `Категория: ${categoryLabel(metadata.category)}`;
    card.appendChild(category);

    const brand = document.createElement('p');
    brand.textContent = `Бренд: ${metadata.brand || '-'}`;
    card.appendChild(brand);

    results.appendChild(card);
  }
}

function clearPreview() {
  cropImage.removeAttribute('src');
  maskImage.removeAttribute('src');
  cropPreviewCard.classList.remove('has-image');
  maskPreviewCard.classList.remove('has-image');
}

function showPreview(cropImageData, maskImageData) {
  cropImage.src = cropImageData;
  maskImage.src = maskImageData;
  cropPreviewCard.classList.add('has-image');
  maskPreviewCard.classList.add('has-image');
}

function setCatalogStatus(message) {
  catalogStatus.textContent = message;
}

function slugify(value) {
  const normalized = value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/\.[^/.]+$/, '')
    .replace(/[^a-z0-9_.-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
  return normalized || 'item';
}

function buildItemId(file, index, total, prefix, timestamp) {
  const base = slugify(prefix || file.name);
  if (prefix) {
    return total > 1 ? `${base}-${index + 1}` : base;
  }
  return `${base}-${timestamp}-${index + 1}`;
}

function appendCatalogLog(message, isError = false) {
  const item = document.createElement('li');
  item.textContent = message;
  item.className = isError ? 'error' : 'success';
  catalogLog.appendChild(item);
}

searchForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(searchForm);
  setStatus('Идёт поиск...');
  results.innerHTML = '';
  clearPreview();

  try {
    const response = await fetch('/search', {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Поиск не выполнен.');
    }

    showPreview(payload.crop_image, payload.mask_image);
    renderResults(payload.results);
    setStatus(`Найдено похожих товаров: ${payload.results.length}.`);
  } catch (error) {
    setStatus(friendlyErrorMessage(error.message));
  }
});

catalogForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = Array.from(catalogForm.elements.catalog_files.files);
  const category = catalogForm.elements.category.value;
  const brand = catalogForm.elements.brand.value.trim();
  const color = catalogForm.elements.color.value.trim();
  const price = catalogForm.elements.price.value.trim();
  const prefix = catalogForm.elements.item_id_prefix.value.trim();
  const submitButton = catalogForm.querySelector('button[type="submit"]');

  if (!files.length) {
    setCatalogStatus('Выберите хотя бы одно изображение каталога.');
    return;
  }

  catalogLog.innerHTML = '';
  submitButton.disabled = true;
  const timestamp = Date.now();
  let indexedCount = 0;

  try {
    for (const [index, file] of files.entries()) {
      const itemId = buildItemId(file, index, files.length, prefix, timestamp);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('item_id', itemId);
      formData.append('category', category);
      if (brand) {
        formData.append('brand', brand);
      }
      if (color) {
        formData.append('color', color);
      }
      if (price) {
        formData.append('price', price);
      }

      setCatalogStatus(`Индексация ${index + 1}/${files.length}: ${file.name}`);
      const response = await fetch('/catalog/add', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          payload.detail || `Не удалось проиндексировать файл ${file.name}`,
        );
      }

      indexedCount += 1;
      appendCatalogLog(
        `Проиндексировано: ${payload.item_id} (${categoryLabel(payload.category)})`,
      );
    }
    setCatalogStatus(`Проиндексировано изображений: ${indexedCount}.`);
    catalogForm.reset();
  } catch (error) {
    appendCatalogLog(friendlyErrorMessage(error.message), true);
    setCatalogStatus(
      `Остановлено после ${indexedCount}/${files.length} изображений.`,
    );
  } finally {
    submitButton.disabled = false;
  }
});
