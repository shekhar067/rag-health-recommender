
import os
import json
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse
import pandas as pd
import torch
import pickle
from rag_pipeline import load_models, load_mimic_notes, build_or_load_faiss_index, rag_health_recommend
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from transformers import pipeline as baseline_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from google.cloud import aiplatform
import nltk
from typing import List


nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def text_f1_score(gold_tokens, pred_tokens):
    """Compute F1 score based on token set overlap."""
    gold_set, pred_set = set(gold_tokens), set(pred_tokens)
    overlap = gold_set & pred_set
    if not gold_set or not pred_set:
        return 0.0
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(gold_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_metrics(pred: str, gold: str, retrieved_docs: List[str]) -> dict:
    """Compute an enhanced set of evaluation metrics."""
    smoothie = SmoothingFunction().method4
    pred, gold = str(pred), str(gold)
    pred_tokens = pred.split() if isinstance(pred, str) else ' '.join(map(str, pred)).split() if isinstance(pred, (list, tuple)) else []
    gold_tokens = gold.split() if isinstance(gold, str) else ' '.join(map(str, gold)).split() if isinstance(gold, (list, tuple)) else []
    logging.info(f"Token debug: gold_tokens={gold_tokens}, pred_tokens={pred_tokens}")
    # Standard Metrics
    bleu = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothie)
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    meteor = meteor_score([gold_tokens] if gold_tokens else [[]], [pred_tokens] if pred_tokens else [[]])
    f1 = text_f1_score(gold_tokens, pred_tokens)
    em = 1.0 if pred.lower() == gold.lower() else 0.0
    clinical_accuracy = 1.0 if "heart failure" in pred.lower() and any(term in pred.lower() for term in ["low-sodium", "furosemide"]) else 0.0
    
    # RAG-Specific & Other Metrics
    pred_ngrams = set(nltk.ngrams(pred_tokens, 3))
    doc_ngrams = set(nltk.ngrams(" ".join(retrieved_docs).split(), 3)) if retrieved_docs else set()
    novelty = len(pred_ngrams - doc_ngrams) / len(pred_ngrams) if pred_ngrams else 0.0
    coverage = len(set(gold_tokens) & set(pred_tokens)) / len(set(gold_tokens)) if gold_tokens else 0.0
    answer_length = len(pred_tokens)
    retrieved_text = " ".join(retrieved_docs) if retrieved_docs else ""
    faithfulness = len(set(pred_tokens) & set(retrieved_text.split())) / len(set(pred_tokens)) if pred_tokens else 0.0

    return {
        "bleu": bleu, "rougeL": rouge_l, "meteor": meteor, "f1": f1, "em": em,
        "clinical_accuracy": clinical_accuracy, "novelty": novelty, "coverage": coverage,
        "answer_length": answer_length, "faithfulness": faithfulness
    }

def setup_logging(output_dir: str):
    """Sets up a clean logger for each run."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "master_orchestrator.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def run_experiment(config: tuple, args: argparse.Namespace, docs: list, titles: list, eval_data: list, is_baseline=False):
    """Runs a single full experiment (either RAG or baseline) and saves results."""
    
    if is_baseline:
        retriever_name, generator_name = "baseline", "flan-t5"
        logging.info(f"\n{'='*20} 🚀 Starting Baseline LLM Experiment {'='*20}")
    else:
        retriever_name, generator_name = config
        logging.info(f"\n{'='*20} 🚀 Starting RAG Experiment: {retriever_name.upper()} + {generator_name.upper()} {'='*20}")

    try:
        # Load models only if it's a RAG experiment
        if not is_baseline:
            models = load_models(retriever_name, generator_name)
            index_path = os.path.join(args.output_dir, f"faiss_index_{retriever_name}.bin")
            faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])
        else:
            device_num = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
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
                retrieved_for_metrics = result.get('retrieved_docs', docs[:args.top_k])  # Fallback to top_k docs

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
    """Execute final comparison and visualization."""
    os.system(f"python run_final_comparison.py --output_dir {output_dir}")

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Master RAG Experiment Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo-data/NOTEEVENTS.csv", help="Path to notes CSV")
    parser.add_argument("--max_notes", type=int, default=10, help="Maximum notes to process")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eval_dataset", type=str, default="data/evaluation_dataset.json", help="Evaluation dataset")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    # Cache eval_data for efficiency
    cache_path = os.path.join(args.output_dir, "eval_data_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            eval_data = pickle.load(f)
        logging.info(f"Loaded eval data from cache: {cache_path}")
    else:
        with open(args.eval_dataset, "r") as f:
            eval_data = json.load(f)
        with open(cache_path, 'wb') as f:
            pickle.dump(eval_data, f)
        logging.info(f"Cached eval data to: {cache_path}")

    rag_experiments_to_run = [
        ('biobert', 'flan-t5'),
        ('pubmedbert', 'flan-t5'),
        ('biobert', 'medalpaca'),
        ('pubmedbert', 'medalpaca'),
    ]

    try:
        docs, titles = load_mimic_notes(args.mimic_path, args.max_notes)
        if not docs or not eval_data:
            logging.error("No documents or evaluation data loaded. Aborting.")
            return
        
        # Run Baseline
        run_experiment(None, args, docs, titles, eval_data, is_baseline=True)

        # Run RAG experiments
        for config in rag_experiments_to_run:
            run_experiment(config, args, docs, titles, eval_data, is_baseline=False)
        
        # Run final comparison
        run_final_comparison(rag_experiments_to_run, args.output_dir)

    except Exception as e:
        logging.error("Fatal error in main execution.", exc_info=True)
    
    logging.info("\nMaster orchestrator completed.")

if __name__ == "__main__":
    main()
