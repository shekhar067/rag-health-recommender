import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
import logging
import argparse
from typing import List, Tuple
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# 1. DATA LOADING
# -------------------------------
def preprocess_text(text: str) -> str:
    """Clean MIMIC-III note text."""
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)  # Remove de-identified placeholders
    text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
    return text

def load_mimic_notes(mimic_csv_path: str, max_notes: int = 10000) -> Tuple[List[str], List[str]]:
    """Load and preprocess MIMIC-III notes."""
    logging.info("Loading MIMIC-III notes...")
    try:
        df = pd.read_csv(mimic_csv_path)
        df = df[df['CATEGORY'] == 'Discharge summary'].head(max_notes)
        notes = [preprocess_text(note) for note in df['TEXT'].fillna('').tolist()]
        titles = df['ROW_ID'].astype(str).tolist()
        logging.info(f"Loaded {len(notes)} notes.")
        return notes, titles
    except Exception as e:
        logging.error(f"Failed to load MIMIC-III notes: {e}")
        raise

# -------------------------------
# 2. EMBEDDING MODEL (BioBERT)
# -------------------------------
logging.info("Loading BioBERT sentence transformer...")
EMBED_MODEL = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
logging.info("BioBERT loaded.")

# -------------------------------
# 3. INDEXING WITH FAISS
# -------------------------------
def encode_in_batches(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts in batches to manage memory."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings.append(EMBED_MODEL.encode(batch, show_progress_bar=True))
    return np.vstack(embeddings)

def build_faiss_index(embeddings: np.ndarray, save_path: str = None) -> faiss.Index:
    """Build and optionally save HNSW index."""
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 40
        index.add(embeddings)
        if save_path:
            faiss.write_index(index, save_path)
            logging.info(f"Saved FAISS index to {save_path}")
        return index
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise

# -------------------------------
# 4. LLM FOR GENERATION
# -------------------------------
logging.info("Loading FLAN-T5 generator...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
GENERATOR = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=-1,
    max_length=128
)
logging.info("Generator loaded.")

# -------------------------------
# 5. RAG PIPELINE
# -------------------------------
def retrieve(query: str, top_k: int, faiss_index: faiss.Index, docs: List[str], titles: List[str]) -> Tuple[List[str], List[str]]:
    """Retrieve documents with error handling."""
    if not query.strip():
        logging.error("Empty query provided.")
        return [], []
    try:
        query_emb = EMBED_MODEL.encode([query])
        distances, idxs = faiss_index.search(np.array(query_emb), top_k)
        retrieved_docs = [docs[i] for i in idxs[0] if i < len(docs)]
        retrieved_titles = [titles[i] for i in idxs[0] if i < len(titles)]
        logging.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        return retrieved_docs, retrieved_titles
    except Exception as e:
        logging.error(f"Retrieval failed: {e}")
        return [], []

def build_prompt(user_query: str, docs: List[str], titles: List[str]) -> str:
    context = "\n".join([f"[{title}]: {doc}" for title, doc in zip(titles, docs)])
    prompt = (
        f"You are a medical assistant. Based on the following medical articles:\n"
        f"{context}\n\n"
        f"User question: {user_query}\n"
        f"Give a clear, safe, evidence-based health recommendation using only the above information."
    )
    return prompt

def generate_answer(prompt: str) -> str:
    try:
        answer = GENERATOR(prompt)[0]['generated_text']
        return answer
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return f"Error: {str(e)}"

def rag_health_recommend(user_query: str, top_k: int, faiss_index: faiss.Index, docs: List[str], titles: List[str], save_to: str = None) -> dict:
    """Run RAG pipeline for a single query."""
    try:
        logging.info(f"Processing user query: {user_query}")
        retrieved_docs, retrieved_titles = retrieve(user_query, top_k, faiss_index, docs, titles)
        if not retrieved_docs:
            return {"query": user_query, "answer": "No relevant documents found.", "citations": []}
        prompt = build_prompt(user_query, retrieved_docs, retrieved_titles)
        answer = generate_answer(prompt)
        result = {
            "query": user_query,
            "answer": answer,
            "citations": retrieved_titles,
            "retrieved_docs": retrieved_docs
        }
        if save_to:
            with open(save_to, "a") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            logging.info(f"Appended output to {save_to}")
        return result
    except Exception as e:
        logging.error(f"RAG pipeline failed: {e}")
        return {"query": user_query, "answer": f"Error: {str(e)}", "citations": []}

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Health Recommender")
    parser.add_argument("--mimic_path", default="data/mimic-iii/NOTEEVENTS.csv", help="Path to MIMIC-III notes")
    parser.add_argument("--max_notes", type=int, default=10000, help="Max notes to load")
    parser.add_argument("--top_k", type=int, default=2, help="Number of documents to retrieve")
    parser.add_argument("--index_path", default="outputs/faiss_index.bin", help="Path to save/load FAISS index")
    parser.add_argument("--output", default="outputs/rag_output.json", help="Output file for results")
    parser.add_argument("--query_file", default=None, help="File with queries (one per line)")
    return parser.parse_args()

# -------------------------------
# 6. MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    PUBMED_DOCS, DOC_TITLES = load_mimic_notes(args.mimic_path, args.max_notes)
    
    # Build or load FAISS index
    if os.path.exists(args.index_path):
        logging.info(f"Loading existing FAISS index from {args.index_path}")
        FAISS_INDEX = faiss.read_index(args.index_path)
    else:
        logging.info("Indexing MIMIC-III notes...")
        DOC_EMBEDDINGS = encode_in_batches(PUBMED_DOCS, batch_size=32)
        FAISS_INDEX = build_faiss_index(DOC_EMBEDDINGS, save_path=args.index_path)
    
    # Process queries
    if args.query_file:
        with open(args.query_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
        results = []
        for query in queries:
            result = rag_health_recommend(query, args.top_k, FAISS_INDEX, PUBMED_DOCS, DOC_TITLES, args.output)
            results.append(result)
            print(f"Question: {result['query']}\nAnswer: {result['answer']}\nCitations:\n" + "\n".join([f"- {c}" for c in result['citations']]))
    else:
        example_query = "How should high blood pressure be treated?"
        result = rag_health_recommend(example_query, args.top_k, FAISS_INDEX, PUBMED_DOCS, DOC_TITLES, args.output)
        print(f"Question: {result['query']}\nAnswer: {result['answer']}\nCitations:\n" + "\n".join([f"- {c}" for c in result['citations']]))