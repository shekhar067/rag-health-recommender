import os
import logging
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import List, Tuple, Dict
import torch

# --- CORRECTED MODEL NAME HERE ---
MODEL_MAP = {
    "retrievers": {
        "biobert": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" # Corrected name
    },
    "generators": {
        "flan-t5": "google/flan-t5-large",
        "medalpaca": "medalpaca/medalpaca-7b"
    }
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- The rest of the file remains exactly the same ---

def load_models(retriever_name: str, generator_name: str) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    retriever_model_id = MODEL_MAP["retrievers"].get(retriever_name)
    logging.info(f"Loading retriever: {retriever_model_id}")
    embed_model = SentenceTransformer(retriever_model_id, device=device)

    generator_model_id = MODEL_MAP["generators"].get(generator_name)
    logging.info(f"Loading generator: {generator_model_id}")

    if "flan-t5" in generator_name:
        tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_id, torch_dtype=torch.float16, device_map="auto")
        task = "text2text-generation"
    elif "medalpaca" in generator_name:
        tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
        model = AutoModelForCausalLM.from_pretrained(generator_model_id, torch_dtype=torch.float16, device_map="auto")
        task = "text-generation"
    else:
        raise ValueError(f"Generator pipeline not configured for: {generator_name}")

    generator_pipeline = pipeline(task, model=model, tokenizer=tokenizer, max_new_tokens=150)
    return {"retriever": embed_model, "generator": generator_pipeline}

def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_mimic_notes(mimic_csv_path: str, max_notes: int = 1000) -> Tuple[List[str], List[str]]:
    logging.info(f"Loading up to {max_notes} notes from {mimic_csv_path}...")
    df = pd.read_csv(mimic_csv_path)
    df_filtered = df[df['CATEGORY'] == 'Discharge summary'].head(max_notes)
    notes = [preprocess_text(note) for note in df_filtered['TEXT'].fillna('').tolist()]
    titles = [f"Note_ID_{row_id}" for row_id in df_filtered['ROW_ID'].tolist()]
    logging.info(f"Loaded {len(notes)} discharge summaries.")
    return notes, titles

def build_or_load_faiss_index(docs: List[str], index_path: str, retriever_model: SentenceTransformer) -> faiss.Index:
    if os.path.exists(index_path):
        logging.info(f"Loading existing FAISS index from {index_path}")
        return faiss.read_index(index_path)
    
    logging.info(f"Building new FAISS index at {index_path}...")
    embeddings = retriever_model.encode(docs, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, index_path)
    logging.info(f"Saved FAISS index to {index_path}")
    return index

def build_personalized_prompt(user_query: str, patient_context: str, docs: List[str]) -> str:
    context_str = "\n\n".join([doc[:500] for doc in docs])  # Reduced from 1000
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

def rag_health_recommend(user_query: str, patient_context: str, top_k: int, models: Dict, faiss_index: faiss.Index, all_docs: List[str], all_titles: List[str]) -> Dict:
    retriever = models['retriever']
    generator = models['generator']

    query_emb = retriever.encode([user_query])
    _, idxs = faiss_index.search(np.array(query_emb).astype('float32'), top_k)
    retrieved_indices = idxs[0]

    if len(retrieved_indices) == 0 or -1 in retrieved_indices:
        return {"query": user_query, "patient_context": patient_context, "answer": "I cannot provide a recommendation as no relevant information was found.", "citations": []}
        
    retrieved_docs = [all_docs[i] for i in retrieved_indices if i < len(all_docs)]
    retrieved_titles = [all_titles[i] for i in retrieved_indices if i < len(all_titles)]
    
    prompt = build_personalized_prompt(user_query, patient_context, retrieved_docs)
    
    generated_obj = generator(prompt)[0]
    answer = generated_obj.get('generated_text', '')
    
    if "Answer:" in answer:
        answer = answer.split("Answer:")[1].strip()

    return {"query": user_query, "patient_context": patient_context, "answer": answer, "citations": retrieved_titles}
