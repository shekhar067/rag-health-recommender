import os
import json
import logging
import argparse
import pandas as pd
import torch
import pickle
from rag_pipeline import load_models, load_mimic_notes, build_or_load_faiss_index, rag_health_recommend
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import pipeline as baseline_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from google.cloud import aiplatform
import nltk

nltk.download('punkt')

def setup_logging(output_dir: str):
    """Set up logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "master_orchestrator.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def compute_metrics(pred: str, gold: str, retrieved_docs: List[str]) -> dict:
    """Compute all evaluation metrics."""
    smoothie = SmoothingFunction().method4
    pred, gold = str(pred), str(gold)
    pred_tokens, gold_tokens = pred.split(), [gold.split()]
    bleu = sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothie)
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    meteor = meteor_score(gold_tokens, pred_tokens)
    f1 = f1_score(gold_tokens, pred_tokens)
    em = 1.0 if pred.lower() == gold.lower() else 0.0
    clinical_accuracy = 1.0 if "heart failure" in pred.lower() and any(term in pred.lower() for term in ["low-sodium", "furosemide"]) else 0.0

    # Novelty: Proportion of unique 3-grams not in retrieved docs
    pred_ngrams = set(nltk.ngrams(pred.split(), 3))
    doc_ngrams = set(nltk.ngrams(" ".join(retrieved_docs).split(), 3))
    novelty = len(pred_ngrams - doc_ngrams) / len(pred_ngrams) if pred_ngrams else 0.0

    # Coverage: Proportion of gold tokens in pred
    coverage = len(set(gold.split()) & set(pred.split())) / len(set(gold.split())) if gold.split() else 0.0

    # Answer Length
    answer_length = len(pred.split())

    # Faithfulness (Simple Proxy): Proportion of pred tokens in retrieved docs
    retrieved_text = " ".join(retrieved_docs)
    faithfulness = len(set(pred.split()) & set(retrieved_text.split())) / len(set(pred.split())) if pred.split() else 0.0

    return {
        "bleu": bleu, "rougeL": rouge_l, "meteor": meteor, "f1": f1, "em": em,
        "clinical_accuracy": clinical_accuracy, "novelty": novelty, "coverage": coverage,
        "answer_length": answer_length, "faithfulness": faithfulness
    }

def run_rag_experiment(config: tuple, args: argparse.Namespace, docs: list, titles: list, eval_data: list):
    """Run RAG experiment."""
    retriever_name, generator_name = config
    logging.info(f"\n{'='*20} 🚀 Starting RAG Experiment: {retriever_name.upper()} + {generator_name.upper()} {'='*20}")
    
    try:
        models = load_models(retriever_name, generator_name)
        index_path = os.path.join(args.output_dir, f"faiss_index_{retriever_name}.bin")
        faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])
        
        eval_results = []
        for item in eval_data:
            result = rag_health_recommend(item["query"], item["patient_context"], args.top_k, models, faiss_index, docs, titles)
            metrics = compute_metrics(result['answer'], item['gold'], result.get('retrieved_docs', []))  # Use retrieved_docs if available
            eval_results.append({"query": item["query"], "context": item["patient_context"], "gold": item['gold'], "prediction": result['answer'], **metrics})
            logging.info(f"Query: {item['query'][:50]}..., ROUGE-L: {metrics['rougeL']:.4f}, Novelty: {metrics['novelty']:.4f}")

        if not eval_results:
            logging.warning("Evaluation loop produced no results.")
            return

        eval_df = pd.DataFrame(eval_results)
        preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _, _, f1 = bert_score(preds, golds, lang="en", verbose=False, device=device)
        eval_df['bertscore_f1'] = f1.tolist()

        output_path = os.path.join(args.output_dir, f"rag_evaluation_{retriever_name}_{generator_name}.csv")
        eval_df.to_csv(output_path, index=False)
        logging.info(f"✅ RAG evaluation complete. Results saved to '{output_path}'")
        logging.info(f"Mean Scores: {eval_df[['bleu', 'rougeL', 'bertscore_f1', 'meteor', 'f1', 'em', 'clinical_accuracy', 'novelty', 'coverage', 'answer_length', 'faithfulness']].mean()}")
    except Exception as e:
        logging.error(f"FATAL ERROR during RAG experiment {config}: {e}", exc_info=True)

def run_baseline_experiment(args: argparse.Namespace, eval_data: list):
    """Run baseline LLM (FLAN-T5)."""
    logging.info(f"\n{'='*20} 🚀 Starting Baseline LLM Experiment {'='*20}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float16, device_map="auto")
        generator = baseline_pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
        
        eval_results = []
        for item in eval_data:
            prompt = f'PATIENT PROFILE: {item["patient_context"]}\n\nQuestion: {item["query"]}\n\nProvide a clear, evidence-based health recommendation.\nAnswer:'
            pred = generator(prompt)[0]['generated_text']
            metrics = compute_metrics(pred, item['gold'], [])  # No retrieved docs for baseline
            eval_results.append({"query": item["query"], "context": item["patient_context"], "gold": item['gold'], "prediction": pred, **metrics})
            logging.info(f"Query: {item['query'][:50]}..., ROUGE-L: {metrics['rougeL']:.4f}, Novelty: {metrics['novelty']:.4f}")

        if not eval_results:
            logging.warning("Baseline evaluation loop produced no results.")
            return

        eval_df = pd.DataFrame(eval_results)
        preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _, _, f1 = bert_score(preds, golds, lang="en", verbose=False, device=device)
        eval_df['bertscore_f1'] = f1.tolist()

        output_path = os.path.join(args.output_dir, "baseline_evaluation_results.csv")
        eval_df.to_csv(output_path, index=False)
        logging.info(f"✅ Baseline evaluation complete. Results saved to '{output_path}'")
        logging.info(f"Mean Scores: {eval_df[['bleu', 'rougeL', 'bertscore_f1', 'meteor', 'f1', 'em', 'clinical_accuracy', 'novelty', 'coverage', 'answer_length', 'faithfulness']].mean()}")
    except Exception as e:
        logging.error(f"FATAL ERROR during baseline experiment: {e}", exc_info=True)

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Master RAG Experiment Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo-data/NOTEEVENTS.csv", help="Path to notes CSV")
    parser.add_argument("--max_notes", type=int, default=10, help="Maximum notes to process")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eval_dataset", type=str, default="data/evaluation_dataset.json", help="Evaluation dataset")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    cache_path = os.path.join(args.output_dir, "eval_data_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            eval_data = pickle.load(f)
        logging.info(f"Loaded eval data from cache: {cache_path}")
    else:
        try:
            with open(args.eval_dataset, "r") as f:
                eval_data = json.load(f)
            with open(cache_path, 'wb') as f:
                pickle.dump(eval_data, f)
            logging.info(f"Cached eval data to: {cache_path}")
        except FileNotFoundError:
            logging.error(f"Evaluation dataset not found: {args.eval_dataset}")
            return

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
        
        run_baseline_experiment(args, eval_data)
        for config in rag_experiments_to_run:
            run_rag_experiment(config, args, docs, titles, eval_data)
        os.system(f"python run_final_comparison_enhanced_updated.py --output_dir {args.output_dir}")

    except Exception as e:
        logging.error("Fatal error in main execution.", exc_info=True)
    
    logging.info("\nMaster orchestrator completed.")

if __name__ == "__main__":
    main()
