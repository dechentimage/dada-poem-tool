#!/usr/bin/env python3
"""
create_dada_poem.py
~~~~~~~~~~~~~~~~~~~~

This script implements a simple tool to generate a six‑line dadaistic poem from
the visible text contained in a screenshot of a web page.  It expects the
image to be a raster file (PNG, JPEG, etc.) and uses the open‑source OCR
engine Tesseract via the `pytesseract` library to extract text from the image.

Once the text has been extracted, the script employs spaCy to perform
part‑of‑speech (POS) tagging and filters the tokens to retain only nouns and
verbs.  The pool of candidate words is then shuffled and arranged into
exactly six lines to form a poem.  Each line contains a random number of
words (between two and five) drawn from the pool.  If the pool of nouns
and verbs is exhausted before six lines can be created, the words are
re‑shuffled and reused as necessary.  The resulting poem is printed to
standard output.

Requirements
------------
This script depends on a few external packages and data files:

* **Tesseract OCR** must be installed and available on the system `PATH`.  On
  most Linux distributions you can install it via your package manager
  (e.g. `sudo apt‑get install tesseract‑ocr`).  On macOS you can use
  Homebrew (`brew install tesseract`).  Windows users can download the
  installer from the Tesseract project page.
* **pytesseract** is a Python wrapper around Tesseract and can be installed
  from PyPI: `pip install pytesseract`.
* **Pillow** (PIL) is used to open image files: `pip install pillow`.
* **spaCy** provides the NLP pipeline.  You need to install spaCy itself
  (`pip install spacy`) and at least one language model appropriate for your
  screenshot.  For German text use `python -m spacy download de_core_news_sm`,
  and for English use `python -m spacy download en_core_web_sm`.  If no
  language model is available, the script falls back to a very simple
  heuristic that treats words starting with an uppercase letter as nouns
  (useful in German) and words ending with common German verb endings (e.g.
  "en", "st", "t") as verbs.

Usage
-----
Run the script from the command line, passing the path to the image file
containing your web page screenshot:

    python3 create_dada_poem.py /path/to/screenshot.png

The script will output a six‑line poem using only nouns and verbs from the
extracted text.  If you wish to specify the language explicitly (so the
correct spaCy model is loaded), you can use the `--lang` option:

    python3 create_dada_poem.py --lang de /path/to/screenshot.png
    python3 create_dada_poem.py --lang en /path/to/screenshot.png

If no language is specified, the script will attempt to detect whether
German ("de") or English ("en") is more appropriate by looking at the
presence of German umlauts in the text and whether spaCy models are
available for those languages.  If both attempts fail, it will use the
heuristic fallback.

Notes
-----
* This tool is designed to generate poetic output rather than accurate
  grammatical sentences.  The poem may be surprising or nonsensical – this
  playful quality is intentional and fits the Dada aesthetic.
* The script works best when the uploaded screenshot contains clear and
  legible text.  Images with noisy backgrounds or small fonts may yield
  poor OCR quality.
* If you encounter errors related to missing spaCy models, please install
  the appropriate model for your language as described above.

"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Iterable, List, Optional

try:
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "Pillow (PIL) is required for image handling.\n"
        "Install it with: pip install pillow"
    ) from e

try:
    import pytesseract
except ImportError as e:
    raise ImportError(
        "pytesseract is required to interface with the Tesseract OCR engine.\n"
        "Install it with: pip install pytesseract\n"
        "Tesseract itself must also be installed on your system."
    ) from e

try:
    import spacy
    from spacy.language import Language
except ImportError:
    # If spaCy isn't installed, we'll handle it gracefully later.
    spacy = None  # type: ignore
    Language = None  # type: ignore


def extract_text(image_path: str) -> str:
    """Perform OCR on the provided image and return the extracted text.

    Parameters
    ----------
    image_path: str
        Path to the image file containing the screenshot.

    Returns
    -------
    str
        A string of extracted text.  Newlines from the OCR output are
        preserved to aid in later tokenization.
    """
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as exc:
        raise RuntimeError(f"Unable to open or process image '{image_path}': {exc}")
    return text


def load_spacy_pipeline(lang: str) -> Optional[Language]:
    """Attempt to load a spaCy language pipeline for the given language code.

    This helper returns `None` if spaCy is not installed or the requested
    model cannot be loaded.

    Parameters
    ----------
    lang: str
        A two‑letter ISO language code (e.g. "de" for German, "en" for English).

    Returns
    -------
    Language | None
        The loaded spaCy language object, or `None` if not available.
    """
    if spacy is None:
        return None
    model_names = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
    }
    model_name = model_names.get(lang)
    if model_name is None:
        return None
    try:
        nlp = spacy.load(model_name)
        return nlp
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Heuristically guess whether the extracted text is German or English.

    The presence of characters with umlauts or the sharp s (ß) triggers a
    German guess; otherwise English is assumed.  This function does not
    guarantee accuracy but serves as a fallback when the user does not specify
    the language explicitly.

    Parameters
    ----------
    text: str
        The text extracted from the image.

    Returns
    -------
    str
        The guessed language code ("de" or "en").  Defaults to "en" if no
        German character is detected.
    """
    german_chars = set("äöüÄÖÜß")
    if any(ch in german_chars for ch in text):
        return "de"
    return "en"


def extract_nouns_verbs_spacy(text: str, nlp: Language) -> List[str]:
    """Use spaCy to extract nouns and verbs from the text.

    Parameters
    ----------
    text: str
        The OCR‑extracted text.
    nlp: Language
        A loaded spaCy language pipeline.

    Returns
    -------
    List[str]
        A list of tokens (strings) that are classified as nouns or verbs.
    """
    doc = nlp(text)
    words: List[str] = []
    for token in doc:
        # Only consider alphabetic tokens (ignore numbers, punctuation).
        if not token.is_alpha:
            continue
        # spaCy's coarse POS tags: 'NOUN' (common noun), 'PROPN' (proper noun),
        # 'VERB' (full verbs) and 'AUX' (auxiliary verbs).  Include all four.
        if token.pos_ in {"NOUN", "PROPN", "VERB", "AUX"}:
            words.append(token.text)
    return words


def extract_nouns_verbs_heuristic(text: str) -> List[str]:
    """Fallback heuristic to extract noun‑ and verb‑like words from text.

    When no spaCy model is available, this function attempts a rough
    approximation.  For German, any word starting with an uppercase letter is
    treated as a noun, and words ending with certain typical verb endings
    ("en", "st", "t", "end") are treated as verbs.  For English, nouns and
    verbs are harder to distinguish without a model; we therefore include
    alphabetic words longer than two characters, which tends to bias toward
    meaningful content words.

    Parameters
    ----------
    text: str
        The OCR‑extracted text.

    Returns
    -------
    List[str]
        A list of candidate words.
    """
    words: List[str] = []
    tokens = re.findall(r"\b\w+\b", text)
    for token in tokens:
        # Skip tokens that are purely numeric or single characters.
        if len(token) <= 1 or token.isdigit():
            continue
        # German nouns start with uppercase letters.
        if token[0].isupper():
            words.append(token)
            continue
        # Typical German verb endings.
        german_verb_endings = ("en", "st", "t", "end")
        lower = token.lower()
        if lower.endswith(german_verb_endings):
            words.append(token)
            continue
        # For English, include longer alphabetic words as a rough approximation.
        if token.isalpha() and len(token) > 3:
            words.append(token)
    return words


def assemble_poem(words: Iterable[str], lines: int = 6) -> List[str]:
    """Shuffle and arrange the given words into a fixed number of lines.

    Parameters
    ----------
    words: Iterable[str]
        The candidate pool of words (nouns and verbs).
    lines: int
        The number of lines the poem should contain.  Defaults to six.

    Returns
    -------
    List[str]
        A list of strings, each representing one line of the poem.
    """
    words_list = list(words)
    random.shuffle(words_list)
    if not words_list:
        return ["(keine Wörter gefunden)"] * lines
    result: List[str] = []
    # Determine an approximate number of words per line.  Aim for 2–5 words.
    segment_size = max(2, min(5, len(words_list) // lines or 2))
    idx = 0
    for _ in range(lines):
        if idx >= len(words_list):
            # Reshuffle and reset if we run out of words.
            random.shuffle(words_list)
            idx = 0
        end_idx = idx + segment_size
        segment = words_list[idx:end_idx]
        idx = end_idx
        result.append(" ".join(segment))
    return result


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a Dadaist poem from a screenshot containing text.")
    parser.add_argument("image", help="Path to the screenshot image (PNG, JPEG, etc.)")
    parser.add_argument("--lang", choices=["de", "en"], default=None,
                        help="Language code (de or en). If omitted, a best guess is made.")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"Error: file '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Perform OCR.
    text = extract_text(args.image)
    if not text.strip():
        print("No text detected in the image.")
        sys.exit(0)

    # Determine language and load spaCy model.
    lang = args.lang or detect_language(text)
    nlp = load_spacy_pipeline(lang)

    # Extract nouns and verbs using spaCy if possible; otherwise use heuristics.
    if nlp is not None:
        words = extract_nouns_verbs_spacy(text, nlp)
    else:
        words = extract_nouns_verbs_heuristic(text)

    if not words:
        print("Keine Substantive oder Verben gefunden (no nouns or verbs found).")
        sys.exit(0)

    poem_lines = assemble_poem(words, lines=6)
    print()
    print("\n".join(poem_lines))


if __name__ == "__main__":
    main()
