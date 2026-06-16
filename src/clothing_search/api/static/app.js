const searchForm = document.querySelector('#search-form');
const statusBox = document.querySelector('#status');
const catalogForm = document.querySelector('#catalog-form');
const catalogStatus = document.querySelector('#catalog-status');
const catalogLog = document.querySelector('#catalog-log');
const cropImage = document.querySelector('#crop-image');
const maskImage = document.querySelector('#mask-image');
const results = document.querySelector('#results');

function setStatus(message) {
  statusBox.textContent = message;
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
      image.alt = `Catalog item ${item.item_id}`;
      card.appendChild(image);
    }

    const title = document.createElement('h3');
    title.textContent = item.item_id;
    card.appendChild(title);

    const score = document.createElement('p');
    score.textContent = `Score: ${Number(item.score).toFixed(3)}`;
    card.appendChild(score);

    const category = document.createElement('p');
    category.textContent = `Category: ${metadata.category || 'unknown'}`;
    card.appendChild(category);

    const brand = document.createElement('p');
    brand.textContent = `Brand: ${metadata.brand || '-'}`;
    card.appendChild(brand);

    results.appendChild(card);
  }
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
  setStatus('Searching...');
  results.innerHTML = '';

  try {
    const response = await fetch('/search', {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Search failed');
    }

    cropImage.src = payload.crop_image;
    maskImage.src = payload.mask_image;
    renderResults(payload.results);
    setStatus(`Found ${payload.results.length} similar items.`);
  } catch (error) {
    setStatus(error.message);
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
    setCatalogStatus('Choose at least one catalog photo.');
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

      setCatalogStatus(`Indexing ${index + 1}/${files.length}: ${file.name}`);
      const response = await fetch('/catalog/add', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Failed to index ${file.name}`);
      }

      indexedCount += 1;
      appendCatalogLog(`Indexed ${payload.item_id} as ${payload.category}`);
    }
    setCatalogStatus(`Indexed ${indexedCount} catalog image(s).`);
    catalogForm.reset();
  } catch (error) {
    appendCatalogLog(error.message, true);
    setCatalogStatus(`Stopped after ${indexedCount}/${files.length} image(s).`);
  } finally {
    submitButton.disabled = false;
  }
});
