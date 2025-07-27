import os
import json
import logging
import argparse
import pandas as pd
import torch
import pickle
import warnings
from rag_pipeline import load_models, load_mimic_notes, build_or_load_faiss_index, rag_health_recommend
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from transformers import pipeline as baseline_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from typing import List

# Suppress verbose warnings for a cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Download required NLTK resources quietly
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "master_orchestrator.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def text_f1_score(gold_tokens, pred_tokens):
    gold_set, pred_set = set(gold_tokens), set(pred_tokens)
    overlap = gold_set & pred_set
    if not gold_set or not pred_set: return 0.0
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(gold_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_metrics(pred: str, gold: str, retrieved_docs: List[str]) -> dict:
    smoothie = SmoothingFunction().method4
    pred, gold = str(pred), str(gold)
    pred_tokens, gold_tokens = pred.split(), gold.split()
    
    # --- FIXED: gold_tokens is wrapped in a list for meteor and bleu ---
    gold_tokens_list = [gold_tokens]
    
    bleu = sentence_bleu(gold_tokens_list, pred_tokens, smoothing_function=smoothie)
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    meteor = meteor_score(gold_tokens_list, pred_tokens)
    f1 = text_f1_score(gold_tokens, pred_tokens)

    # RAG-Specific Metrics
    pred_ngrams = set(nltk.ngrams(pred_tokens, 3))
    doc_text = " ".join(retrieved_docs) if retrieved_docs else ""
    doc_ngrams = set(nltk.ngrams(doc_text.split(), 3))
    novelty = len(pred_ngrams - doc_ngrams) / len(pred_ngrams) if pred_ngrams else 0.0
    faithfulness = len(set(pred_tokens) & set(doc_text.split())) / len(set(pred_tokens)) if pred_tokens else 0.0

    return { "bleu": bleu, "rougeL": rouge_l, "meteor": meteor, "f1": f1, "novelty": novelty, "faithfulness": faithfulness }

def run_experiment(config: tuple, args: argparse.Namespace, docs: list, titles: list, eval_data: list, is_baseline=False):
    if is_baseline:
        retriever_name, generator_name = "baseline", "flan-t5"
        logging.info(f"\n{'='*20} 🚀 Starting Baseline LLM Experiment {'='*20}")
    else:
        retriever_name, generator_name = config
        logging.info(f"\n{'='*20} 🚀 Starting RAG Experiment: {retriever_name.upper()} + {generator_name.upper()} {'='*20}")

    try:
        if not is_baseline:
            models = load_models(retriever_name, generator_name)
            index_path = os.path.join(args.output_dir, f"faiss_index_{retriever_name}.bin")
            faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])
        else:
            device_num = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float16, device_map="auto")
            generator = baseline_pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

        eval_results = []
        for item in eval_data:
            if is_baseline:
                prompt = f'PATIENT PROFILE: {item["patient_context"]}\n\nQuestion: {item["query"]}\n\nProvide a clear, evidence-based health recommendation.\nAnswer:'
                pred = generator(prompt)[0]['generated_text']
                retrieved_for_metrics = []
            else:
                result = rag_health_recommend(item["query"], item["patient_context"], args.top_k, models, faiss_index, docs, titles)
                pred = result['answer']
                retrieved_for_metrics = result.get('retrieved_docs', [])
            
            metrics = compute_metrics(pred, item['gold'], retrieved_for_metrics)
            eval_results.append({"query": item["query"], "prediction": pred, "gold": item['gold'], **metrics})
        
        eval_df = pd.DataFrame(eval_results)
        preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, bert_f1 = bert_score(preds, golds, lang="en", verbose=False, device=device)
        eval_df['bertscore_f1'] = bert_f1.tolist()

        filename = "baseline_evaluation_results.csv" if is_baseline else f"rag_evaluation_{retriever_name}_{generator_name}.csv"
        output_path = os.path.join(args.output_dir, filename)
        eval_df.to_csv(output_path, index=False)
        logging.info(f"✅ Evaluation complete. Results saved to '{output_path}'")

    except Exception as e:
        logging.error(f"FATAL ERROR during {'baseline' if is_baseline else config} experiment: {e}", exc_info=True)

def run_final_comparison(configurations: list, output_dir: str):
    # This should be a separate script for modularity, but can be called here.
    # We will assume a separate comparison script exists and is called after this script finishes.
    logging.info("All experiments finished. You can now run the comparison script.")

def main():
    parser = argparse.ArgumentParser(description="Master RAG Experiment Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/NOTEEVENTS.csv", help="Path to notes CSV")
    parser.add_argument("--max_notes", type=int, default=10, help="Maximum notes to process")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Evaluation dataset")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    rag_experiments_to_run = [
        ('biobert', 'flan-t5'),
        ('pubmedbert', 'flan-t5'),
        # ('biobert', 'medalpaca'),
        # ('pubmedbert', 'medalpaca'),
    ]

    try:
        docs, titles = load_mimic_notes(args.mimic_path, args.max_notes)
        with open(args.eval_dataset, "r") as f:
            eval_data = json.load(f)

        if not docs or not eval_data:
            logging.error("No documents or evaluation data loaded. Aborting.")
            return
        
        run_experiment(None, args, docs, titles, eval_data, is_baseline=True)
        for config in rag_experiments_to_run:
            run_experiment(config, args, docs, titles, eval_data, is_baseline=False)
        
        # After all experiments, you would run your comparison script.
        # For now, we'll just log that it's done.
        run_final_comparison(rag_experiments_to_run, args.output_dir)

    except Exception as e:
        logging.error("Fatal error in main execution.", exc_info=True)
    
    logging.info("\nMaster orchestrator completed.")

if __name__ == "__main__":
    main()
