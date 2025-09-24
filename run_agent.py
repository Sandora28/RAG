import os, pickle, faiss, requests
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")

PROMPT_TEMPLATE = """
შენ ხარ იურიდიული ასისტენტი, რომელიც პასუხობს მხოლოდ საქართველოს სამოქალაქო კოდექსის (Matsne — №134) მიხედვით.

ქვემოთ მოცემულია შესაბამისი მუხლები:

{context}

შეკითხვა: {question}

 დაწერე პასუხი ქართულად, გასაგებად და მოკლედ, ისე რომ ახსნა იყოს, თუ რა განსაზღვრავს ან როგორია სამართლებრივი ნორმა.  
 საჭიროების შემთხვევაში შეგიძლია მოკლე ციტირებაც, მაგრამ ძირითადი ტექსტი უნდა იყოს შენს სიტყვებში ახსნილი.  
 მიუთითე წყარო ფორმატით [მუხლი <N>].  
 თუ შესაბამისი რეგულაცია ვერ მოიძებნა, მიუწერე: "კოდექსში შესაბამისი რეგულაცია არ არის."
"""


def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "articles.pkl"), "rb") as f:
        articles = pickle.load(f)
    return index, articles

def embed_query(query, model):
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.array(emb).astype("float32")

def search(index, qvec, k=4):
    D, I = index.search(qvec, k)
    return I[0], D[0]

def format_context(indices, articles):
    chunks = []
    for idx in indices:
        art = articles[idx]
        chunks.append(f"მუხლი {art['article']}\n{art['text']}")
    return "\n\n---\n\n".join(chunks)

def call_ollama(prompt, model_name):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512}
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", data)

def main():
    print("📂 ვტვირთავ ინდექსს...")
    index, articles = load_index()
    model = SentenceTransformer(MODEL_NAME)
    print("ინდექსი ჩატვირთულია. შეგიძლია დასვა იურიდიული კითხვები ქართულად.")
    print("გამოსასვლელად აკრიფე 'exit'.")

    while True:
        q = input("\nშეკითხვა > ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        qvec = embed_query(q, model)
        ids, scores = search(index, qvec, k=6)

        context = format_context(ids, articles)

        prompt = PROMPT_TEMPLATE.format(context=context, question=q)
        try:
            answer = call_ollama(prompt, OLLAMA_MODEL)
        except Exception as e:
            print("Ollama error:", e)
            continue

        print("\n--- პასუხი ---\n")
        print(answer)
        print("\n--- წყარო მუხლები ---")
        for i, idx in enumerate(ids):
            print(f"{i+1}. მუხლი {articles[idx]['article']} (score={scores[i]:.4f})")

if __name__ == "__main__":
    main()
