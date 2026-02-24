import os
import tempfile
from flask import Flask, request, render_template_string, jsonify
from create_dada_poem import extract_text, detect_language, load_spacy_pipeline, extract_nouns_verbs_spacy, extract_nouns_verbs_heuristic, assemble_poem

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>Dadaistisches Gedicht</title></head>
    <body>
        <h1>Dadaistisches Gedicht Tool</h1>
        <p>Laden Sie ein Bildschirmfoto hoch und erhalten Sie ein dadaistisches Gedicht mit sechs Zeilen.</p>
        <form method="post" action="/generate" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="Gedicht generieren">
        </form>
    </body>
    </html>
    """)

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'Keine Datei hochgeladen.'}), 400
    file = request.files['image']
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        file.save(tmp.name)
        text = extract_text(tmp.name)
    os.unlink(tmp.name)
    if not text.strip():
        return jsonify({'error': 'Kein Text im Bild gefunden.'}), 400
    # Determine language and load model
    lang = detect_language(text)
    nlp = load_spacy_pipeline(lang)
    if nlp:
        words = extract_nouns_verbs_spacy(text, nlp)
    else:
        words = extract_nouns_verbs_heuristic(text)
    if not words:
        poem_lines = ["Keine Substantive oder Verben gefunden"] * 6
    else:
        poem_lines = assemble_poem(words, lines=6)
    # Render poem in HTML; each line separate
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>Dadaistisches Gedicht</title></head>
    <body>
        <h1>Ihr Gedicht</h1>
        <pre>{{ poem }}</pre>
        <a href="/">Noch einmal</a>
    </body>
    </html>
    """, poem="\n".join(poem_lines))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
