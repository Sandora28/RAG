import re, os, pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

PDF_PATH = os.getenv("MATSNE_PDF", "matsne.pdf")
INDEX_DIR = "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text(pdf_path):
    """ამოიღებს მთელ ტექსტს PDF-დან"""
    reader = PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def split_into_articles(full_text):
    """გაყოფს ტექსტს მუხლებად"""
    matches = list(re.finditer(r'(?m)^\s*მუხლი\s*(\d+)\b', full_text))
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        num = m.group(1)
        articles.append({"article": num, "text": block})
    return articles

def build_embeddings(articles):
    """ქმნის embedding-ებს ყველა მუხლისთვის"""
    model = SentenceTransformer(MODEL_NAME)
    texts = [a["text"] for a in articles]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def save_index(embeddings, articles):
    """ინახავს FAISS ინდექსს და მუხლების მეტამონაცემებს"""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "articles.pkl"), "wb") as f:
        pickle.dump(articles, f)
    print(" ინდექსი და მუხლები შენახულია:", INDEX_DIR)

def main():
    print("ვკითხულობ PDF ფაილს:", PDF_PATH)
    full_text = extract_text(PDF_PATH)
    articles = split_into_articles(full_text)
    print(f"🔍 ნაპოვნია {len(articles)} მუხლი.")
    if len(articles) == 0:
        raise SystemExit(" მუხლები ვერ მოიძებნა. გადაამოწმე PDF ან regex.")
    print(" ვქმნი embedding-ებს...")
    embeddings = build_embeddings(articles)
    save_index(embeddings, articles)
    with open(os.path.join(INDEX_DIR, "full_text.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

if __name__ == "__main__":
    main()
