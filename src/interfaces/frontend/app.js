/* =============================================================
   AI LEARNING ASSISTANT — APP.JS
   Nintendo 2001 Chrome Edition
   ============================================================= */
'use strict';

const API_BASE = '';

/* ─── Global State ─── */
let isLoading = false;
let selectedFilenames = [];     // Array of selected document filenames
let activeView = 'chat';        // 'chat' | 'summary' | 'quiz' | 'flashcards'

/* ─── DOM Accessors ─── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
  fileInput:       () => $('#fileInput'),
  uploadBtn:       () => $('#uploadBtn'),
  uploadZone:      () => $('#uploadZone'),
  docList:         () => $('#docList'),
  docCount:        () => $('#docCount'),
  emptyDocs:       () => $('#emptyDocs'),
  clearSelBtn:     () => $('#clearSelectionBtn'),
  selectAllBtn:    () => $('#selectAllBtn'),
  tabBar:          () => $('#tabBar'),
  chatMessages:    () => $('#chatMessages'),
  chatInput:       () => $('#chatInput'),
  sendBtn:         () => $('#sendBtn'),
  chatEmpty:       () => $('#chatEmpty'),
  summaryContent:  () => $('#summaryContent'),
  quizContent:     () => $('#quizContent'),
  flashcardsContent: () => $('#flashcardsContent'),
  genSummaryBtn:   () => $('#genSummaryBtn'),
  genQuizBtn:      () => $('#genQuizBtn'),
  genFlashcardsBtn: () => $('#genFlashcardsBtn'),
  statusIndicator: () => $('#statusIndicator'),
};

/* ─── API Helper ─── */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || JSON.stringify(err));
  }
  return res.json();
}

function escapeHTML(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* ─── Filters Builder ─── */
function getFilters() {
  if (selectedFilenames.length === 0) return null;
  if (selectedFilenames.length === 1) {
    return { filename: selectedFilenames[0] };
  }
  return { filenames: selectedFilenames };
}

function getDocumentParam() {
  // For endpoints that take "document" as single filename
  if (selectedFilenames.length === 1) return selectedFilenames[0];
  return null;
}

/* ═══════════════════════════════════════════
   SIDEBAR — Document Multi-Select
   ═══════════════════════════════════════════ */

function updateSelectionUI() {
  const clearBtn = DOM.clearSelBtn();
  const selAllBtn = DOM.selectAllBtn();
  if (selectedFilenames.length > 0) {
    clearBtn.classList.add('visible');
  } else {
    clearBtn.classList.remove('visible');
  }
}

function handleCheckboxChange(e) {
  const cb = e.target;
  const fname = cb.dataset.filename;
  if (cb.checked) {
    if (!selectedFilenames.includes(fname)) selectedFilenames.push(fname);
  } else {
    selectedFilenames = selectedFilenames.filter(f => f !== fname);
  }
  updateSelectionUI();
}

async function loadDocuments() {
  try {
    const data = await apiFetch('/documents');
    const docs = data.documents || data || [];
    const list = DOM.docList();
    const emptyEl = DOM.emptyDocs();
    const countEl = DOM.docCount();

    list.innerHTML = '';

    if (docs.length === 0) {
      list.innerHTML = '<div class="doc-list-empty" id="emptyDocs">No documents uploaded yet.<br/>Upload a PDF to start.</div>';
      countEl.textContent = '';
      return;
    }

    countEl.textContent = `${docs.length} file${docs.length > 1 ? 's' : ''}`;

    docs.forEach(doc => {
      const fname = typeof doc === 'string' ? doc : (doc.filename || doc.name);
      const item = document.createElement('label');
      item.className = 'doc-item has-check';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'doc-checkbox';
      cb.dataset.filename = fname;
      cb.checked = selectedFilenames.includes(fname);
      cb.addEventListener('change', handleCheckboxChange);

      const nameSpan = document.createElement('span');
      nameSpan.className = 'doc-name';
      nameSpan.textContent = fname;

      item.appendChild(cb);
      item.appendChild(nameSpan);
      list.appendChild(item);
    });

    updateSelectionUI();
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function handleClearSelection() {
  selectedFilenames = [];
  $$('.doc-checkbox').forEach(cb => cb.checked = false);
  updateSelectionUI();
}

function handleSelectAll() {
  selectedFilenames = [];
  $$('.doc-checkbox').forEach(cb => {
    cb.checked = true;
    selectedFilenames.push(cb.dataset.filename);
  });
  updateSelectionUI();
}

/* ═══════════════════════════════════════════
   UPLOAD
   ═══════════════════════════════════════════ */

async function handleUpload() {
  const file = DOM.fileInput().files[0];
  if (!file || isLoading) return;

  setLoadingState(true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    DOM.fileInput().value = '';
    await loadDocuments();
  } catch (err) {
    alert('Upload failed: ' + err.message);
  } finally {
    setLoadingState(false);
  }
}

/* ═══════════════════════════════════════════
   TAB ROUTING (View Switching)
   ═══════════════════════════════════════════ */

function switchView(viewName) {
  activeView = viewName;

  // Update tab buttons
  $$('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });

  // Show/hide panels (preserves state — just display toggle)
  $$('.view-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `view-${viewName}`);
  });
}

/* ═══════════════════════════════════════════
   LOADING STATE
   ═══════════════════════════════════════════ */

function setLoadingState(loading) {
  isLoading = loading;
  DOM.sendBtn().disabled = loading;
  DOM.genSummaryBtn().disabled = loading;
  DOM.genQuizBtn().disabled = loading;
  DOM.genFlashcardsBtn().disabled = loading;
  DOM.uploadBtn().disabled = loading;
}

function requireSelection(featureName) {
  if (selectedFilenames.length === 0) {
    return `<div class="error-message">Select one or more documents from the sidebar before generating ${featureName}.</div>`;
  }
  return null;
}

/* ═══════════════════════════════════════════
   CHAT / Q&A
   ═══════════════════════════════════════════ */

function appendChatBubble(type, content) {
  // Remove empty state
  const emptyEl = DOM.chatEmpty();
  if (emptyEl) emptyEl.remove();

  const log = DOM.chatMessages();
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${type}`;

  const label = document.createElement('span');
  label.className = 'bubble-label';
  label.textContent = type === 'user' ? 'You' : 'AI';

  bubble.appendChild(label);

  const textEl = document.createElement('div');
  textEl.innerHTML = content;
  bubble.appendChild(textEl);

  log.appendChild(bubble);
  log.scrollTop = log.scrollHeight;
}

async function handleAsk() {
  const input = DOM.chatInput();
  const question = input.value.trim();
  if (!question || isLoading) return;

  appendChatBubble('user', escapeHTML(question));
  input.value = '';
  setLoadingState(true);

  try {
    const data = await apiFetch('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, filters: getFilters() }),
    });

    let answerHTML = escapeHTML(data.answer);
    answerHTML = answerHTML.replace(/\[S(\d+)\]/g, '<span class="citation-tag">S$1</span>');

    if (data.citations && data.citations.length > 0) {
      answerHTML += '<div class="citations-section"><span class="citations-label">Sources</span>';
      data.citations.forEach(c => {
        answerHTML += `<span class="citation-tag">${escapeHTML(c.marker)} ${escapeHTML(c.filename)} p.${c.page}</span>`;
      });
      answerHTML += '</div>';
    }

    appendChatBubble('ai', answerHTML);
  } catch (err) {
    appendChatBubble('ai', `<span style="color:var(--error)">${escapeHTML(err.message)}</span>`);
  } finally {
    setLoadingState(false);
  }
}

/* ═══════════════════════════════════════════
   SUMMARY
   ═══════════════════════════════════════════ */

async function handleSummary() {
  if (isLoading) return;
  const err = requireSelection('a summary');
  if (err) { DOM.summaryContent().innerHTML = err; return; }

  setLoadingState(true);
  DOM.summaryContent().innerHTML = '<div class="loading-indicator">⏳ Generating Summary…</div>';

  try {
    const payload = {
      document: getDocumentParam(),
      filters: getFilters(),
    };
    const data = await apiFetch('/summarize', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    let html = `<div class="content-card">
      <div class="card-header">📝 Summary — ${escapeHTML((data.scope || '').toUpperCase())}${data.target ? ': ' + escapeHTML(data.target) : ''}</div>
      <p>${escapeHTML(data.summary)}</p>`;

    if (data.key_points && data.key_points.length > 0) {
      html += '<ul>';
      data.key_points.forEach(kp => { html += `<li>${escapeHTML(kp)}</li>`; });
      html += '</ul>';
    }

    if (data.citations && data.citations.length > 0) {
      html += '<div class="citations-section"><span class="citations-label">Sources</span>';
      data.citations.forEach(c => {
        html += `<span class="citation-tag">${escapeHTML(c.marker)} ${escapeHTML(c.filename)} p.${c.page}</span>`;
      });
      html += '</div>';
    }

    html += '</div>';
    DOM.summaryContent().innerHTML = html;
  } catch (err) {
    DOM.summaryContent().innerHTML = `<div class="error-message">Summary Failed: ${escapeHTML(err.message)}</div>`;
  } finally {
    setLoadingState(false);
  }
}

/* ═══════════════════════════════════════════
   QUIZ
   ═══════════════════════════════════════════ */

async function handleQuiz() {
  if (isLoading) return;
  const err = requireSelection('a quiz');
  if (err) { DOM.quizContent().innerHTML = err; return; }

  setLoadingState(true);
  DOM.quizContent().innerHTML = '<div class="loading-indicator">⏳ Generating Quiz…</div>';

  try {
    const payload = {
      document: getDocumentParam(),
      filters: getFilters(),
    };
    const data = await apiFetch('/quiz', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const items = data.items || data.quiz || [];
    let html = '';

    items.forEach((item, qi) => {
      const letters = ['A', 'B', 'C', 'D'];
      html += `<div class="quiz-item" data-qi="${qi}" data-correct="${item.correct_index}">`;
      html += `<div class="quiz-meta">`;
      if (item.difficulty) html += `<span>${escapeHTML(item.difficulty)}</span>`;
      if (item.topic) html += `<span>${escapeHTML(item.topic)}</span>`;
      if (item.source_markers && item.source_markers.length) {
        html += `<span>${item.source_markers.join(', ')}</span>`;
      }
      html += `</div>`;
      html += `<div class="quiz-question">Q${qi + 1}. ${escapeHTML(item.question)}</div>`;
      html += '<div class="quiz-options">';
      (item.options || []).forEach((opt, oi) => {
        html += `<div class="quiz-option" data-qi="${qi}" data-oi="${oi}">
          <span class="opt-letter">${letters[oi]})</span>
          <span>${escapeHTML(opt)}</span>
        </div>`;
      });
      html += '</div>';
      html += `<div class="quiz-explanation" id="exp-${qi}">
        <div class="exp-label">Explanation</div>
        <p>${escapeHTML(item.explanation || '')}</p>
      </div>`;
      html += '</div>';
    });

    if (data.citations && data.citations.length > 0) {
      html += '<div class="content-card"><div class="card-header">Sources</div>';
      data.citations.forEach(c => {
        html += `<span class="citation-tag">${escapeHTML(c.marker)} ${escapeHTML(c.filename)} p.${c.page}</span>`;
      });
      html += '</div>';
    }

    DOM.quizContent().innerHTML = html;
    attachQuizListeners();
  } catch (err) {
    DOM.quizContent().innerHTML = `<div class="error-message">Quiz Failed: ${escapeHTML(err.message)}</div>`;
  } finally {
    setLoadingState(false);
  }
}

function attachQuizListeners() {
  $$('.quiz-option').forEach(opt => {
    opt.addEventListener('click', function () {
      const qi = this.dataset.qi;
      const oi = parseInt(this.dataset.oi);
      const quizItem = $(`.quiz-item[data-qi="${qi}"]`);
      const correctIdx = parseInt(quizItem.dataset.correct);

      // Prevent re-answering
      if (quizItem.classList.contains('answered')) return;
      quizItem.classList.add('answered');

      // Mark selection
      this.classList.add('selected');

      // Show correct/incorrect
      const allOpts = quizItem.querySelectorAll('.quiz-option');
      allOpts.forEach((o, i) => {
        if (i === correctIdx) o.classList.add('correct');
        else if (i === oi && oi !== correctIdx) o.classList.add('incorrect');
        o.style.pointerEvents = 'none';
      });

      // Show explanation
      const exp = $(`#exp-${qi}`);
      if (exp) exp.classList.add('visible');
    });
  });
}

/* ═══════════════════════════════════════════
   FLASHCARDS
   ═══════════════════════════════════════════ */

async function handleFlashcards() {
  if (isLoading) return;
  const err = requireSelection('flashcards');
  if (err) { DOM.flashcardsContent().innerHTML = err; return; }

  setLoadingState(true);
  DOM.flashcardsContent().innerHTML = '<div class="loading-indicator">⏳ Generating Flashcards…</div>';

  try {
    const payload = {
      document: getDocumentParam(),
      filters: getFilters(),
    };
    const data = await apiFetch('/flashcards', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const cards = data.cards || [];
    let html = '<p class="flashcard-instruction">Click a card to flip it and reveal the answer.</p>';
    html += '<div class="flashcard-grid">';

    cards.forEach((card, ci) => {
      html += `<div class="flashcard-wrapper">
        <div class="flashcard" onclick="this.classList.toggle('flipped')">
          <div class="flashcard-face flashcard-front">
            <span class="face-label">Question</span>
            <div class="face-text">${escapeHTML(card.front)}</div>
            ${card.topic ? `<span class="face-topic">${escapeHTML(card.topic)}</span>` : ''}
          </div>
          <div class="flashcard-face flashcard-back">
            <span class="face-label">Answer</span>
            <div class="face-text">${escapeHTML(card.back)}</div>
            ${card.hint ? `<div class="face-hint">💡 ${escapeHTML(card.hint)}</div>` : ''}
            ${card.source_markers && card.source_markers.length ? `<span class="face-topic">${card.source_markers.join(', ')}</span>` : ''}
          </div>
        </div>
      </div>`;
    });

    html += '</div>';

    if (data.citations && data.citations.length > 0) {
      html += '<div class="content-card" style="margin-top:var(--sp-lg)"><div class="card-header">Sources</div>';
      data.citations.forEach(c => {
        html += `<span class="citation-tag">${escapeHTML(c.marker)} ${escapeHTML(c.filename)} p.${c.page}</span>`;
      });
      html += '</div>';
    }

    DOM.flashcardsContent().innerHTML = html;
  } catch (err) {
    DOM.flashcardsContent().innerHTML = `<div class="error-message">Flashcards Failed: ${escapeHTML(err.message)}</div>`;
  } finally {
    setLoadingState(false);
  }
}

/* ═══════════════════════════════════════════
   HEALTH CHECK
   ═══════════════════════════════════════════ */

async function checkHealth() {
  try {
    await apiFetch('/health');
    DOM.statusIndicator().className = 'nav-status online';
    DOM.statusIndicator().textContent = '● Online';
  } catch {
    DOM.statusIndicator().className = 'nav-status offline';
    DOM.statusIndicator().textContent = '● Offline';
  }
}

/* ═══════════════════════════════════════════
   DRAG & DROP
   ═══════════════════════════════════════════ */

function setupDragDrop() {
  const zone = DOM.uploadZone();
  ['dragenter', 'dragover'].forEach(evt => {
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(evt => {
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove('dragover'); });
  });
  zone.addEventListener('drop', e => {
    if (e.dataTransfer.files.length) {
      DOM.fileInput().files = e.dataTransfer.files;
    }
  });
}

/* ═══════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Load docs
  loadDocuments();
  checkHealth();
  setInterval(checkHealth, 30000);

  // Upload
  DOM.uploadBtn().addEventListener('click', handleUpload);
  setupDragDrop();

  // Selection buttons
  DOM.clearSelBtn().addEventListener('click', handleClearSelection);
  DOM.selectAllBtn().addEventListener('click', handleSelectAll);

  // Tab routing
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  // Chat
  DOM.sendBtn().addEventListener('click', handleAsk);
  DOM.chatInput().addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  });

  // Generate buttons
  DOM.genSummaryBtn().addEventListener('click', handleSummary);
  DOM.genQuizBtn().addEventListener('click', handleQuiz);
  DOM.genFlashcardsBtn().addEventListener('click', handleFlashcards);
});
