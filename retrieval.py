from pathlib import Path
from textwrap import wrap

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_passages(folder):
    passages = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            paragraph = " ".join(paragraph.split())
            for chunk in wrap(paragraph, width=1200, break_long_words=False):
                passages.append({"id": f"S{len(passages) + 1}", "file": str(path), "text": chunk})
    return passages


def retrieve(passages, query, limit=12):
    corpus = [p["text"] for p in passages]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(corpus + [query])
    scores = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    ranked = scores.argsort()[::-1][:limit]
    return [passages[i] for i in ranked]
