Of course. Here is the complete, updated `rag_evaluate.py` script.

The main changes in this version are:

  * It now accepts `--retriever` and `--generator` arguments so you can test each of your four configurations.
  * It saves the output CSV files with unique names (e.g., `rag_evaluation_biobert_flan-t5.csv`) so that your results for different experiments don't overwrite each other.
  * It uses the modular `load_models` function from the updated `rag_pipeline.py`.

This is the version you should use for your comparative experiments.

-----

### \#\# Updated `rag_evaluate.py` File

```python
import os
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict
import torch

# Assuming rag_pipeline.py and its models are in the same path
from rag_pipeline import (
    load_models,
    load_mimic_notes,
    build_or_load_faiss_index,
    rag_health_recommend,
    build_personalized_prompt # Import this for the baseline
)
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- METRIC COMPUTATION ---
def compute_metrics(pred: str, gold: str) -> Dict:
    """Computes BLEU and ROUGE-L scores."""
    smoothie = SmoothingFunction().method4
    # Ensure pred and gold are strings and handle empty predictions
    pred = str(pred) if pred is not None else ""
    gold = str(gold) if gold is not None else ""
    
    pred_tokens = pred.split()
    gold_tokens = [gold.split()] # List of reference sentences
    
    bleu = sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothie)
    
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    
    return {"bleu": bleu, "rougeL": rouge_l}


def main():
    # --- MODIFIED: Added retriever and generator arguments ---
    parser = argparse.ArgumentParser(description="Personalized RAG Evaluation Framework")
    parser.add_argument("--retriever", type=str, required=True, choices=["biobert", "pubmedbert"], help="Retriever model to evaluate")
    parser.add_argument("--generator", type=str, required=True, choices=["flan-t5", "medalpaca"], help="Generator model to evaluate")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo-data/NOTEEVENTS.csv", help="Path to MIMIC-III notes for RAG")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for index and results")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents for RAG")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"🔬 Starting Evaluation for Retriever: {args.retriever}, Generator: {args.generator}")

    # --- MODIFIED: Dynamically load the specified models ---
    models = load_models(args.retriever, args.generator)
    docs, titles = load_mimic_notes(args.mimic_path)
    
    # Use a unique index path for each retriever
    index_path = os.path.join(args.output_dir, f"faiss_index_{args.retriever}.bin")
    faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])
    
    with open(args.eval_dataset, "r") as f:
        eval_data = json.load(f)

    # --- RAG System Evaluation ---
    logging.info("Evaluating Personalized RAG System...")
    rag_results = []
    for item in eval_data:
        query, context, gold = item["query"], item["patient_context"], item["gold"]
        
        pred_dict = rag_health_recommend(
            user_query=query,
            patient_context=context,
            top_k=args.top_k,
            models=models,
            faiss_index=faiss_index,
            all_docs=docs,
            all_titles=titles
        )
        pred = pred_dict['answer']
        metrics = compute_metrics(pred, gold)
        rag_results.append({"query": query, "context": context, "gold": gold, "prediction": pred, **metrics})
    
    rag_results_df = pd.DataFrame(rag_results)
    
    # Add BERTScore
    preds, golds = rag_results_df["prediction"].tolist(), rag_results_df["gold"].tolist()
    _, _, f1 = bert_score(preds, golds, lang="en", verbose=True, device=DEVICE)
    rag_results_df['bertscore_f1'] = f1.tolist()

    # --- MODIFIED: Save to a unique filename ---
    output_filename = f"rag_evaluation_{args.retriever}_{args.generator}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    rag_results_df.to_csv(output_path, index=False)
    
    logging.info(f"✅ RAG evaluation complete. Results saved to '{output_path}'")
    logging.info("\n--- RAG Mean Scores ---")
    logging.info(rag_results_df[['bleu', 'rougeL', 'bertscore_f1']].mean())


if __name__ == "__main__":
    main()
```
