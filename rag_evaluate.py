import json
import os
import logging
import argparse
from typing import List, Dict
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Assuming rag_pipeline.py and its models are in the same path
from rag_pipeline import rag_health_recommend, load_mimic_notes, build_or_load_faiss_index
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- BASELINE LLM (Loaded once for efficiency) ---
BASE_LLM_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-large")
BASE_LLM_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
BASE_GENERATOR = pipeline("text2text-generation", model=BASE_LLM_MODEL, tokenizer=BASE_LLM_TOKENIZER, device=-1, max_length=150)

def plain_llm_generate(user_query: str, patient_context: str) -> str:
    """Generates a response from the baseline LLM using patient context."""
    prompt = (
        f"PATIENT PROFILE: {patient_context}\n\n"
        f"Question: {user_query}\n\n"
        f"Provide a clear, evidence-based health recommendation.\n"
        f"Answer:"
    )
    return BASE_GENERATOR(prompt)[0]['generated_text']

# --- METRIC COMPUTATION ---
def compute_metrics(pred: str, gold: str) -> Dict:
    """Computes BLEU, ROUGE-L, and checks for exact match."""
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)
    
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    
    return {"bleu": bleu, "rougeL": rouge_l}

# --- EVALUATION CORE ---
def evaluate_system(
    eval_dataset: List[Dict],
    system_func,
    faiss_components: Dict = None
) -> pd.DataFrame:
    """Evaluates a given system (RAG or baseline) on the dataset."""
    results = []
    for item in eval_dataset:
        query, context, gold = item["query"], item["patient_context"], item["gold"]
        
        if faiss_components: # RAG system
            pred_dict = system_func(query, context, faiss_components['top_k'], faiss_components['index'], faiss_components['docs'], faiss_components['titles'])
            pred = pred_dict['answer']
        else: # Baseline LLM
            pred = system_func(query, context)
            
        metrics = compute_metrics(pred, gold)
        results.append({"query": query, "context": context, "gold": gold, "prediction": pred, **metrics})
        
    df = pd.DataFrame(results)
    
    # Calculate BERTScore for the whole set
    preds, golds = df["prediction"].tolist(), df["gold"].tolist()
    _, _, f1 = bert_score(preds, golds, lang="en", verbose=True)
    df['bertscore_f1'] = f1.tolist()
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Personalized RAG Evaluation")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo/NOTEEVENTS.csv", help="Path to MIMIC-III notes for RAG")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for index and results")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents for RAG")
    args = parser.parse_args()

    # Load evaluation data
    with open(args.eval_dataset, "r") as f:
        eval_data = json.load(f)

    # --- RAG System Evaluation ---
    logging.info("Setting up RAG components for evaluation...")
    docs, titles = load_mimic_notes(args.mimic_path)
    index_path = os.path.join(args.output_dir, "faiss_index_biobert.bin")
    faiss_index = build_or_load_faiss_index(docs, index_path)
    rag_components = {"top_k": args.top_k, "index": faiss_index, "docs": docs, "titles": titles}
    
    logging.info("Evaluating Personalized RAG System...")
    rag_results_df = evaluate_system(eval_data, rag_health_recommend, rag_components)
    
    # --- Baseline LLM Evaluation ---
    logging.info("Evaluating Baseline LLM...")
    base_results_df = evaluate_system(eval_data, plain_llm_generate)

    # --- Log Mean Scores and Save Results ---
    logging.info("\n--- RAG Mean Scores ---")
    logging.info(rag_results_df[['bleu', 'rougeL', 'bertscore_f1']].mean())
    
    logging.info("\n--- Baseline LLM Mean Scores ---")
    logging.info(base_results_df[['bleu', 'rougeL', 'bertscore_f1']].mean())

    rag_results_df.to_csv(os.path.join(args.output_dir, "rag_evaluation_results.csv"), index=False)
    base_results_df.to_csv(os.path.join(args.output_dir, "baseline_evaluation_results.csv"), index=False)
    
    logging.info(f"✅ Evaluation complete. Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()
