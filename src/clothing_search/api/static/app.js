const form = document.querySelector('#search-form');
const statusBox = document.querySelector('#status');
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
    card.innerHTML = `
      <h3>${item.item_id}</h3>
      <p>Score: ${Number(item.score).toFixed(3)}</p>
      <p>Category: ${metadata.category || 'unknown'}</p>
      <p>Brand: ${metadata.brand || '-'}</p>
    `;
    results.appendChild(card);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
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
