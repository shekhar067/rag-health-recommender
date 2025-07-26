"""
Retrieval-Augmented Generation (RAG) for Personalized Health Recommendations
Uses BioBERT for retrieval, FAISS for semantic search, and FLAN-T5 for answer generation.
Author: <Your Name>
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# 1. DATASET (can load from file)
# -------------------------------
PUBMED_DOCS = [
    "Hypertension is managed by lifestyle modification and medications such as ACE inhibitors.",
    "Metformin remains a first-line drug for type 2 diabetes.",
    "COVID-19 vaccines, including mRNA vaccines, greatly reduce severe infection risk.",
    "Inhaled corticosteroids are mainstay for asthma management.",
    "Statins help lower cholesterol and prevent cardiovascular disease."
]
DOC_TITLES = [
    "Hypertension Management",
    "Type 2 Diabetes Therapy",
    "COVID-19 Vaccines",
    "Asthma Treatment",
    "Statin Use in CVD Prevention"
]

# -------------------------------
# 2. EMBEDDING MODEL (BioBERT)
# -------------------------------
logging.info("Loading BioBERT sentence transformer...")
EMBED_MODEL = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
logging.info("BioBERT loaded.")

# -------------------------------
# 3. INDEXING WITH FAISS
# -------------------------------
logging.info("Indexing PubMed documents...")
DOC_EMBEDDINGS = EMBED_MODEL.encode(PUBMED_DOCS)
DIM = DOC_EMBEDDINGS.shape[1]
FAISS_INDEX = faiss.IndexFlatL2(DIM)
FAISS_INDEX.add(np.array(DOC_EMBEDDINGS))
logging.info("Indexing complete.")

# -------------------------------
# 4. LLM FOR GENERATION (PyTorch only)
# -------------------------------
logging.info("Loading FLAN-T5 generator...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
GENERATOR = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",  # PyTorch only!
    device=-1,       # -1=CPU, 0=GPU if available
    max_length=128
)
logging.info("Generator loaded.")

# -------------------------------
# 5. RAG PIPELINE
# -------------------------------
def retrieve(query: str, top_k: int = 2) -> Tuple[List[str], List[str]]:
    query_emb = EMBED_MODEL.encode([query])
    _, idxs = FAISS_INDEX.search(np.array(query_emb), top_k)
    docs = [PUBMED_DOCS[i] for i in idxs[0]]
    titles = [DOC_TITLES[i] for i in idxs[0]]
    return docs, titles

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
    answer = GENERATOR(prompt)[0]['generated_text']
    return answer

def rag_health_recommend(user_query: str, top_k: int = 2, save_to: str = None) -> str:
    logging.info(f"Processing user query: {user_query}")
    docs, titles = retrieve(user_query, top_k=top_k)
    prompt = build_prompt(user_query, docs, titles)
    answer = generate_answer(prompt)
    citations = "\nCitations:\n" + "\n".join([f"- {t}" for t in titles])
    result = f"Question: {user_query}\nAnswer: {answer}{citations}"
    print(result)
    # Optionally save to a file
    if save_to:
        with open(save_to, "w") as f:
            json.dump({
                "user_query": user_query,
                "retrieved_docs": docs,
                "doc_titles": titles,
                "prompt": prompt,
                "answer": answer,
                "citations": titles
            }, f, indent=2)
        logging.info(f"Saved output to {save_to}")
    return result

# -------------------------------
# 6. EXAMPLE USAGE
# -------------------------------
if __name__ == "__main__":
    example_query = "How should high blood pressure be treated?"
    rag_health_recommend(example_query, top_k=2, save_to="output.json")

    # Uncomment for interactive use:
    # while True:
    #     user_query = input("Enter a health question (or 'exit'): ")
    #     if user_query.lower() == "exit":
    #         break
    #     rag_health_recommend(user_query)
