const emailInput = document.getElementById('emailInput');
const charCount = document.getElementById('charCount');
const predictButton = document.getElementById('submitBtn');
const analyzeButton = document.getElementById('analyzeBtn');
const clearButton = document.getElementById('clearBtn');
const resetButton = document.getElementById('resetBtn');
const predictionValue = document.getElementById('predictionValue');
const confidenceValue = document.getElementById('confidenceValue');
const modelValue = document.getElementById('modelValue');
const processingValue = document.getElementById('processingValue');
const resultCard = document.getElementById('resultCard');
const explanationList = document.getElementById('explanationList');

function updateCharCount() {
  charCount.textContent = `${emailInput.value.length} / 2000`;
}

function setResultState(type, label, confidence, modelName, processing) {
  resultCard.className = `result-card ${type}`;
  predictionValue.textContent = label;
  confidenceValue.textContent = `${confidence}%`;
  modelValue.textContent = modelName;
  processingValue.textContent = processing;
}

function renderExplanation(list) {
  explanationList.innerHTML = '';

  if (!list || list.length === 0) {
    explanationList.innerHTML = '<li>Explanation appears after analysis.</li>';
    return;
  }

  list.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    explanationList.appendChild(li);
  });
}

async function analyzeEmail() {
  const text = emailInput.value.trim();

  if (!text) {
    alert('Please enter an email message before analyzing.');
    return;
  }

  const start = performance.now();
  setResultState('neutral', 'Processing...', '0', 'Loading...', '...');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email_text: text })
    });

    const payload = await response.json();
    const duration = ((performance.now() - start) / 1000).toFixed(2);

    if (!response.ok) {
      throw new Error(payload.error || 'Prediction request failed.');
    }

    const prediction = payload.prediction.toUpperCase();
    const confidence = Number(payload.confidence || 0).toFixed(2);
    const model = payload.model || 'Model';
    const type = prediction === 'SPAM' ? 'spam' : 'ham';

    setResultState(type, prediction, confidence, model, `${duration}s`);
    renderExplanation(payload.explanation || [
      'Text classification uses TF-IDF signals and probability-based ranking.',
      'This is a model estimate, not a guarantee of factual intent.'
    ]);
  } catch (error) {
    setResultState('neutral', 'Error', '0', 'Unavailable', '—');
    renderExplanation([error.message]);
  }
}

emailInput.addEventListener('input', updateCharCount);
predictButton.addEventListener('click', analyzeEmail);
analyzeButton.addEventListener('click', analyzeEmail);
clearButton.addEventListener('click', () => {
  emailInput.value = '';
  updateCharCount();
});
resetButton.addEventListener('click', () => {
  emailInput.value = '';
  updateCharCount();
  setResultState('neutral', '—', '0', '—', '—');
  renderExplanation([]);
});

document.querySelectorAll('.chip').forEach((button) => {
  button.addEventListener('click', () => {
    emailInput.value = button.dataset.example;
    updateCharCount();
  });
});

updateCharCount();
