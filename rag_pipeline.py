import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
import logging
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# 1. DATA LOADING & PREP
# -------------------------------
def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)  # Remove de-identified placeholders
    text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
    return text

def load_mimic_notes(mimic_csv_path: str, max_notes: int = 10000) -> Tuple[List[str], List[str]]:
    """Load and preprocess MIMIC-III discharge summaries."""
    logging.info(f"Loading up to {max_notes} notes from {mimic_csv_path}...")
    df = pd.read_csv(mimic_csv_path)
    df = df[df['CATEGORY'] == 'Discharge summary'].head(max_notes)
    notes = [preprocess_text(note) for note in df['TEXT'].fillna('').tolist()]
    titles = [f"Note_ID_{row_id}" for row_id in df['ROW_ID'].tolist()]
    logging.info(f"Loaded {len(notes)} discharge summaries.")
    return notes, titles

# -------------------------------
# 2. MODEL LOADING
# -------------------------------
logging.info("Loading BioBERT sentence transformer...")
EMBED_MODEL = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

logging.info("Loading FLAN-T5 generator...")
GENERATOR_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-large")
GENERATOR_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
GENERATOR = pipeline(
    "text2text-generation",
    model=GENERATOR_MODEL,
    tokenizer=GENERATOR_TOKENIZER,
    device=-1, # Use -1 for CPU, 0 for GPU
    max_length=150
)

# -------------------------------
# 3. INDEXING
# -------------------------------
def build_or_load_faiss_index(docs: List[str], index_path: str) -> faiss.Index:
    """Build a new FAISS index or load an existing one."""
    if os.path.exists(index_path):
        logging.info(f"Loading existing FAISS index from {index_path}")
        return faiss.read_index(index_path)
    
    logging.info("No existing index found. Building new FAISS index...")
    embeddings = EMBED_MODEL.encode(docs, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    
    logging.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    return index

# -------------------------------
# 4. RAG PIPELINE CORE
# -------------------------------
def retrieve(query: str, top_k: int, faiss_index: faiss.Index) -> List[int]:
    """Retrieve top_k document indices from FAISS."""
    query_emb = EMBED_MODEL.encode([query])
    _, idxs = faiss_index.search(np.array(query_emb).astype('float32'), top_k)
    return idxs[0]

def build_personalized_prompt(user_query: str, patient_context: str, docs: List[str]) -> str:
    """Builds a prompt that includes patient context for personalization."""
    context_str = "\n\n".join(docs)
    prompt = (
        f"You are a clinical assistant AI. Your task is to provide an evidence-based recommendation for a specific patient.\n\n"
        f"PATIENT PROFILE: {patient_context}\n\n"
        f"Use the following retrieved medical notes as your primary source of information:\n"
        f"--- RETRIEVED CONTEXT START ---\n"
        f"{context_str}\n"
        f"--- RETRIEVED CONTEXT END ---\n\n"
        f"Based ONLY on the provided context and the patient profile, answer the following question. "
        f"If the context is insufficient to answer, state that you cannot provide a recommendation based on the information available.\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )
    return prompt

def rag_health_recommend(
    user_query: str,
    patient_context: str,
    top_k: int,
    faiss_index: faiss.Index,
    all_docs: List[str],
    all_titles: List[str]
) -> Dict:
    """Runs the full personalized RAG pipeline with a safety check."""
    logging.info(f"Processing query for patient: {patient_context}")
    
    # 1. Retrieve
    retrieved_indices = retrieve(user_query, top_k, faiss_index)
    
    # 2. Safety Check & Context Gathering
    if len(retrieved_indices) == 0:
        logging.warning("Safety Check: No documents were retrieved. Aborting generation.")
        return {
            "query": user_query,
            "patient_context": patient_context,
            "answer": "I cannot provide a recommendation as no relevant information was found.",
            "citations": []
        }
        
    retrieved_docs = [all_docs[i] for i in retrieved_indices]
    retrieved_titles = [all_titles[i] for i in retrieved_indices]
    
    # 3. Augment (Build Prompt)
    prompt = build_personalized_prompt(user_query, patient_context, retrieved_docs)
    
    # 4. Generate
    answer = GENERATOR(prompt)[0]['generated_text']
    
    return {
        "query": user_query,
        "patient_context": patient_context,
        "answer": answer,
        "citations": retrieved_titles
    }
