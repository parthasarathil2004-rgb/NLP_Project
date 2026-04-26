"""
NLP Text Cleaning Web App - Flask Backend
Run: python app.py
"""

import os
import re
import string
import pickle
import nltk
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ── Download required NLTK data ──────────────────────────────────────────────
NLTK_RESOURCES = [
    'punkt', 'punkt_tab', 'stopwords', 'wordnet',
    'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'omw-1.4'
]

def download_nltk_resources():
    for r in NLTK_RESOURCES:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ── Lazy-import NLTK modules with graceful fallbacks ─────────────────────────
def _try_import_nltk():
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords, wordnet as wn
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk import pos_tag, ne_chunk
        # Quick validation
        word_tokenize("test sentence")
        sw = set(stopwords.words('english'))
        return word_tokenize, sent_tokenize, sw, wn, pos_tag, ne_chunk, PorterStemmer, WordNetLemmatizer
    except Exception:
        return None

_nltk = _try_import_nltk()

# Regex-based fallbacks
def _regex_word_tok(text):
    return re.findall(r"[a-zA-Z0-9']+", text)

def _regex_sent_tok(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p] or [text]

FALLBACK_STOPWORDS = set(
    "a an the is are was were be been being have has had do does did will would could "
    "should may might must shall can i me my myself we our you your he she it its they "
    "them their this that these those and or but if in on at to for of with by from up "
    "about into through".split()
)

if _nltk:
    word_tokenize, sent_tokenize, STOP_WORDS, wordnet, pos_tag, ne_chunk, PorterStemmer, WordNetLemmatizer = _nltk
    stemmer    = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    NLTK_OK    = True
else:
    word_tokenize = _regex_word_tok
    sent_tokenize = _regex_sent_tok
    STOP_WORDS    = FALLBACK_STOPWORDS
    wordnet = pos_tag = ne_chunk = None
    stemmer = lemmatizer = None
    NLTK_OK = False
    print("WARNING: NLTK corpus data unavailable — using regex fallbacks.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
os.makedirs('models', exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_wordnet_pos(tag):
    if not wordnet:
        return None
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text, options):
    if options.get('lowercase'):
        text = text.lower()
    if options.get('remove_html'):
        text = re.sub(r'<.*?>', '', text)
    if options.get('remove_urls'):
        text = re.sub(r'http\S+|www\S+', '', text)
    if options.get('remove_emojis'):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
    if options.get('remove_punctuation'):
        text = text.translate(str.maketrans('', '', string.punctuation))
    if options.get('remove_special'):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if options.get('remove_whitespace'):
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── NLP pipeline ──────────────────────────────────────────────────────────────
def run_nlp(text, options):
    results = {}
    words = word_tokenize(text)

    if options.get('tokenization'):
        results['word_tokens']     = words
        results['sentence_tokens'] = sent_tokenize(text)

    if options.get('stopwords'):
        filtered = [w for w in words if w.lower() not in STOP_WORDS]
        results['stopwords_removed'] = filtered

    if options.get('stemming'):
        if stemmer:
            results['stemmed'] = [stemmer.stem(w) for w in words]
        else:
            # Very simple suffix-stripping fallback
            def simple_stem(w):
                for s in ('ing','tion','ed','ly','er','est','ness'):
                    if w.endswith(s) and len(w) - len(s) > 2:
                        return w[:-len(s)]
                return w
            results['stemmed'] = [simple_stem(w) for w in words]

    if options.get('lemmatization'):
        if lemmatizer and pos_tag and wordnet:
            tagged = pos_tag(words)
            results['lemmatized'] = [
                lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged
            ]
        else:
            results['lemmatized'] = words  # identity fallback

    if options.get('pos_tagging'):
        if pos_tag:
            results['pos_tags'] = pos_tag(words)
        else:
            results['pos_tags'] = [(w, 'NN') for w in words]  # fallback: all nouns

    if options.get('ner'):
        if ne_chunk and pos_tag:
            tagged  = pos_tag(words)
            chunked = ne_chunk(tagged)
            entities = []
            for subtree in chunked:
                if hasattr(subtree, 'label'):
                    entity = ' '.join([w for w, _ in subtree.leaves()])
                    entities.append({'entity': entity, 'label': subtree.label()})
            results['ner_entities'] = entities
        else:
            results['ner_entities'] = []

    if options.get('tfidf'):
        sents = sent_tokenize(text)
        if len(sents) < 2:
            sents = [text, text + ' nlp processing']
        tfidf_model = TfidfVectorizer()
        X = tfidf_model.fit_transform(sents)
        with open('models/tfidf_model.pkl', 'wb') as f:
            pickle.dump(tfidf_model, f)
        df = pd.DataFrame(X.toarray(), columns=tfidf_model.get_feature_names_out())
        results['tfidf'] = {
            'features': tfidf_model.get_feature_names_out().tolist(),
            'matrix':   df.round(4).to_dict(orient='records')
        }

    if options.get('bow'):
        sents = sent_tokenize(text)
        if len(sents) < 2:
            sents = [text, text + ' nlp processing']
        bow_model = CountVectorizer()
        X = bow_model.fit_transform(sents)
        with open('models/bow_model.pkl', 'wb') as f:
            pickle.dump(bow_model, f)
        df = pd.DataFrame(X.toarray(), columns=bow_model.get_feature_names_out())
        results['bow'] = {
            'features': bow_model.get_feature_names_out().tolist(),
            'matrix':   df.to_dict(orient='records')
        }

    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data    = request.get_json()
        text    = data.get('text', '')
        options = data.get('options', {})

        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400

        cleaned     = clean_text(text, options)
        nlp_results = run_nlp(cleaned, options)

        return jsonify({
            'original': text,
            'cleaned':  cleaned,
            'nlp':      nlp_results,
            'models_saved': {
                'tfidf': os.path.exists('models/tfidf_model.pkl'),
                'bow':   os.path.exists('models/bow_model.pkl')
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/model/<model_name>')
def download_model(model_name):
    allowed = {'tfidf_model', 'bow_model'}
    if model_name not in allowed:
        return jsonify({'error': 'Invalid model'}), 400
    path = f'models/{model_name}.pkl'
    if not os.path.exists(path):
        return jsonify({'error': 'Model not trained yet'}), 404
    return send_file(path, as_attachment=True)

@app.route('/download/text', methods=['POST'])
def download_text():
    data = request.get_json()
    text = data.get('text', '')
    from io import BytesIO
    buf = BytesIO(text.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name='cleaned_text.txt',
                     mimetype='text/plain')

if __name__ == '__main__':
    print()
    print("=" * 50)
    print("  NLP Studio — Text Processing Lab")
    print("  Running at http://127.0.0.1:5000")
    if not NLTK_OK:
        print("  NOTE: Install NLTK data for full NLP features")
        print("  >>> python -c \"import nltk; nltk.download('all')\"")
    print("=" * 50)
    print()
    app.run(debug=True, use_reloader=False)