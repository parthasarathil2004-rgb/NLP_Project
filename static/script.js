/**
 * NLP Studio — script.js
 * Handles: live preview, pipeline submit, output rendering, downloads
 */

/* ── DOM refs ─────────────────────────────────────────────── */
const inputText      = document.getElementById('inputText');
const charCount      = document.getElementById('charCount');
const livePreview    = document.getElementById('livePreview');
const outputArea     = document.getElementById('outputArea');
const runBtn         = document.getElementById('runBtn');
const clearBtn       = document.getElementById('clearBtn');
const selectAll      = document.getElementById('selectAll');
const themeToggle    = document.getElementById('themeToggle');
const fileUpload     = document.getElementById('fileUpload');
const loaderOverlay  = document.getElementById('loaderOverlay');
const dlText         = document.getElementById('dlText');
const dlTfidf        = document.getElementById('dlTfidf');
const dlBow          = document.getElementById('dlBow');

/* ── Helpers ──────────────────────────────────────────────── */
/** Collect checkbox options into a plain object */
function getOptions() {
  const opts = {};
  document.querySelectorAll('.toggle-row input[type=checkbox]').forEach(cb => {
    opts[cb.name] = cb.checked;
  });
  return opts;
}

/** Simple live-preview — just apply client-side cleaning (no NLP tasks) */
function liveClean(text) {
  const opts = getOptions();
  let t = text;
  if (opts.lowercase)          t = t.toLowerCase();
  if (opts.remove_html)        t = t.replace(/<.*?>/g, '');
  if (opts.remove_urls)        t = t.replace(/https?:\/\/\S+|www\.\S+/g, '');
  if (opts.remove_emojis)      t = t.replace(/[^\x00-\x7F]+/g, '');
  if (opts.remove_punctuation) t = t.replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '');
  if (opts.remove_special)     t = t.replace(/[^a-zA-Z0-9\s]/g, '');
  if (opts.remove_whitespace)  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

/* ── Live preview listener ────────────────────────────────── */
let debounceTimer;
inputText.addEventListener('input', () => {
  const text = inputText.value;
  charCount.textContent = text.length;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const preview = liveClean(text);
    livePreview.textContent = preview || 'Your cleaned text appears here as you type…';
  }, 150);
});

// Also update preview when toggles change
document.querySelectorAll('.toggle-row input').forEach(cb => {
  cb.addEventListener('change', () => {
    const preview = liveClean(inputText.value);
    livePreview.textContent = preview || 'Your cleaned text appears here as you type…';
  });
});

/* ── Theme toggle ─────────────────────────────────────────── */
themeToggle.addEventListener('click', () => {
  const html = document.documentElement;
  const isDark = html.dataset.theme === 'dark';
  html.dataset.theme = isDark ? 'light' : 'dark';
  themeToggle.querySelector('.theme-icon').textContent = isDark ? '●' : '◑';
});

/* ── File upload ──────────────────────────────────────────── */
fileUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    inputText.value = ev.target.result;
    charCount.textContent = ev.target.result.length;
    livePreview.textContent = liveClean(ev.target.result);
  };
  reader.readAsText(file);
});

/* ── Select All ───────────────────────────────────────────── */
let allSelected = false;
selectAll.addEventListener('click', () => {
  allSelected = !allSelected;
  document.querySelectorAll('.toggle-row input[type=checkbox]').forEach(cb => {
    cb.checked = allSelected;
  });
  selectAll.textContent = allSelected ? 'Clear All' : 'Select All';
  livePreview.textContent = liveClean(inputText.value) || 'Your cleaned text appears here as you type…';
});

/* ── Clear ────────────────────────────────────────────────── */
clearBtn.addEventListener('click', () => {
  inputText.value = '';
  charCount.textContent = '0';
  livePreview.textContent = 'Your cleaned text appears here as you type…';
  renderEmpty();
  disableDownloads();
});

/* ── Run Pipeline ─────────────────────────────────────────── */
runBtn.addEventListener('click', async () => {
  const text = inputText.value.trim();
  if (!text) { showToast('Please enter some text first.'); return; }

  loaderOverlay.hidden = false;
  runBtn.disabled = true;

  try {
    const res = await fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, options: getOptions() })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data);
    enableDownloads(data);
  } catch (err) {
    showToast('Error: ' + err.message, true);
  } finally {
    loaderOverlay.hidden = true;
    runBtn.disabled = false;
  }
});

/* ── Downloads ────────────────────────────────────────────── */
let lastCleanedText = '';

dlText.addEventListener('click', async () => {
  if (!lastCleanedText) return;
  const res = await fetch('/download/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: lastCleanedText })
  });
  const blob = await res.blob();
  triggerDownload(blob, 'cleaned_text.txt');
});

dlTfidf.addEventListener('click', () => downloadModel('tfidf_model'));
dlBow.addEventListener('click',   () => downloadModel('bow_model'));

async function downloadModel(name) {
  const res = await fetch(`/download/model/${name}`);
  if (!res.ok) { showToast('Model not trained yet. Run TF-IDF / BOW first.', true); return; }
  const blob = await res.blob();
  triggerDownload(blob, `${name}.pkl`);
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function enableDownloads(data) {
  dlText.disabled  = false;
  dlTfidf.disabled = !data.models_saved?.tfidf;
  dlBow.disabled   = !data.models_saved?.bow;
}
function disableDownloads() {
  dlText.disabled = dlTfidf.disabled = dlBow.disabled = true;
}

/* ═══════════════════════════════════════════════════════════
   RENDER RESULTS
   ═══════════════════════════════════════════════════════════ */
function renderResults(data) {
  lastCleanedText = data.cleaned;
  outputArea.innerHTML = '';

  /* Cleaned text */
  addBlock('🧹 Cleaned Text', `<p class="result-cleaned">${escHtml(data.cleaned)}</p>`);

  const nlp = data.nlp;

  /* Word tokens */
  if (nlp.word_tokens) {
    addBlock('🔤 Word Tokens <span class="rtag">' + nlp.word_tokens.length + '</span>',
      tokenList(nlp.word_tokens, ''));
  }

  /* Sentence tokens */
  if (nlp.sentence_tokens) {
    addBlock('📄 Sentence Tokens <span class="rtag">' + nlp.sentence_tokens.length + '</span>',
      tokenList(nlp.sentence_tokens, 'sent'));
  }

  /* Stopwords removed */
  if (nlp.stopwords_removed) {
    addBlock('🚫 After Stopwords Removal <span class="rtag">' + nlp.stopwords_removed.length + '</span>',
      tokenList(nlp.stopwords_removed, ''));
  }

  /* Stemming */
  if (nlp.stemmed) {
    addBlock('🌱 Stemmed Tokens', tokenList(nlp.stemmed, 'stem'));
  }

  /* Lemmatization */
  if (nlp.lemmatized) {
    addBlock('📚 Lemmatized Tokens', tokenList(nlp.lemmatized, 'lem'));
  }

  /* POS Tagging */
  if (nlp.pos_tags) {
    const rows = nlp.pos_tags.map(([word, tag]) =>
      `<tr><td>${escHtml(word)}</td><td><span class="pos-tag">${tag}</span></td><td style="color:var(--text2);font-size:11px">${posDesc(tag)}</td></tr>`
    ).join('');
    addBlock('🏷️ POS Tags', `
      <div class="matrix-scroll">
        <table class="pos-table">
          <tr><th>Word</th><th>Tag</th><th>Description</th></tr>
          ${rows}
        </table>
      </div>`);
  }

  /* NER */
  if (nlp.ner_entities) {
    if (nlp.ner_entities.length === 0) {
      addBlock('🔍 Named Entities', '<p style="color:var(--text2);font-size:12px;font-family:var(--font-mono)">No named entities found.</p>');
    } else {
      const chips = nlp.ner_entities.map(e =>
        `<div class="ner-chip">${escHtml(e.entity)} <span class="ner-type">${e.label}</span></div>`
      ).join('');
      addBlock('🔍 Named Entities', `<div class="ner-list">${chips}</div>`);
    }
  }

  /* TF-IDF */
  if (nlp.tfidf) {
    addBlock('📊 TF-IDF Matrix <span class="rtag">' + nlp.tfidf.features.length + ' features</span>',
      renderMatrix(nlp.tfidf.features, nlp.tfidf.matrix));
  }

  /* BOW */
  if (nlp.bow) {
    addBlock('📦 Bag of Words <span class="rtag">' + nlp.bow.features.length + ' features</span>',
      renderMatrix(nlp.bow.features, nlp.bow.matrix));
  }
}

/* ── Sub-renderers ── */
function addBlock(title, bodyHtml) {
  const div = document.createElement('div');
  div.className = 'result-block';
  div.innerHTML = `
    <div class="result-title">${title}</div>
    <div class="result-body">${bodyHtml}</div>`;
  outputArea.appendChild(div);
}

function tokenList(tokens, cls) {
  return `<div class="token-list">
    ${tokens.map(t => `<span class="token ${cls}">${escHtml(String(t))}</span>`).join('')}
  </div>`;
}

function renderMatrix(features, rows) {
  const featureCols = features.map(f => `<th>${escHtml(f)}</th>`).join('');
  const dataRows = rows.map((row, i) => {
    const cells = features.map(f => `<td>${row[f] ?? 0}</td>`).join('');
    return `<tr><td>Doc ${i+1}</td>${cells}</tr>`;
  }).join('');
  return `<div class="matrix-scroll">
    <table class="matrix-table">
      <tr><th>Doc</th>${featureCols}</tr>
      ${dataRows}
    </table>
  </div>`;
}

/* POS tag descriptions */
const POS_DESC = {
  NN:'Noun', NNS:'Noun (pl)', NNP:'Proper Noun', NNPS:'Proper Noun (pl)',
  VB:'Verb', VBD:'Verb (past)', VBG:'Verb (-ing)', VBN:'Verb (past part)', VBP:'Verb (pres)', VBZ:'Verb (3rd)',
  JJ:'Adjective', JJR:'Adj (comp)', JJS:'Adj (super)',
  RB:'Adverb', RBR:'Adv (comp)', RBS:'Adv (super)',
  PRP:'Pronoun', 'PRP$':'Poss Pronoun', WP:'Wh-Pronoun',
  DT:'Determiner', IN:'Preposition', CC:'Conjunction',
  CD:'Cardinal', MD:'Modal', TO:'to', EX:'Existential',
  '.':'Punct', ',':'Punct', ':':'Punct',
};
function posDesc(tag) { return POS_DESC[tag] || tag; }

/* Empty state */
function renderEmpty() {
  outputArea.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">◎</div>
      <p>Run the pipeline to see results here.</p>
    </div>`;
}

/* HTML escape */
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* Toast notifications */
function showToast(msg, isError = false) {
  const t = document.createElement('div');
  t.style.cssText = `
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:${isError ? 'var(--accent3)' : 'var(--accent)'};
    color:#000; font-family:var(--font-mono); font-size:13px;
    padding:10px 20px; border-radius:8px; z-index:9999;
    animation:slideIn .3s ease; font-weight:700;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}