import os
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import all necessary functions from your pipeline
from rag_pipeline import (
    load_models,
    load_mimic_notes,
    build_or_load_faiss_index,
    rag_health_recommend
)
# Import metric calculations
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

# --- Logging and Setup ---
def setup_logging(output_dir: str):
    """Sets up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "master_orchestrator.log")
    # Clear previous log file
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# --- Metric Calculation (copied from rag_evaluate.py for convenience) ---
def compute_metrics(pred: str, gold: str) -> dict:
    """Computes BLEU and ROUGE-L scores."""
    smoothie = SmoothingFunction().method4
    pred = str(pred) if pred is not None else ""
    gold = str(gold) if gold is not None else ""
    pred_tokens = pred.split()
    gold_tokens = [gold.split()]
    
    bleu = sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothie)
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    return {"bleu": bleu, "rougeL": rouge_l}

# --- Main Experiment Logic ---
def run_single_experiment(config: tuple, args: argparse.Namespace, docs: list, titles: list, eval_data: list):
    """Runs one full experiment: RAG generation + evaluation."""
    retriever_name, generator_name = config
    logging.info(f"\n{'='*20} 🚀 Starting Experiment: {retriever_name.upper()} + {generator_name.upper()} {'='*20}")

    # 1. Load models for this configuration
    models = load_models(retriever_name, generator_name)

    # 2. Build or load the specific index for the retriever
    index_path = os.path.join(args.output_dir, f"faiss_index_{retriever_name}.bin")
    faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])

    # 3. Run RAG pipeline for all queries
    rag_results = []
    for item in eval_data:
        query, context = item["query"], item["patient_context"]
        result = rag_health_recommend(query, context, args.top_k, models, faiss_index, docs, titles)
        # Add the gold answer for easy evaluation
        result['gold'] = item['gold']
        rag_results.append(result)

    # 4. Evaluate the results and save to CSV
    eval_results = []
    for result in rag_results:
        metrics = compute_metrics(result['answer'], result['gold'])
        eval_results.append({
            "query": result['query'],
            "context": result['patient_context'],
            "gold": result['gold'],
            "prediction": result['answer'],
            **metrics
        })
    
    eval_df = pd.DataFrame(eval_results)
    preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score(preds, golds, lang="en", verbose=False, device=device)
    eval_df['bertscore_f1'] = f1.tolist()

    output_filename = f"rag_evaluation_{retriever_name}_{generator_name}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    eval_df.to_csv(output_path, index=False)
    logging.info(f"✅ Evaluation complete for {retriever_name}+{generator_name}. Results saved to '{output_path}'")

# --- Final Comparison Logic (copied from compare_results.py) ---
def run_final_comparison(configurations: list, output_dir: str):
    """Loads all CSVs and generates a final comparison chart."""
    logging.info(f"\n{'='*20} 📊 Generating Final Comparison {'='*20}")
    
    results_data = {}
    for config_tuple in configurations:
        config_name = f"{config_tuple[0]}_{config_tuple[1]}"
        file_path = os.path.join(output_dir, f'rag_evaluation_{config_name}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            results_data[config_name.replace('_', ' + ').title()] = df[['bleu', 'rougeL', 'bertscore_f1']].mean()
        else:
            logging.warning(f"Result file not found, skipping: {file_path}")

    if not results_data:
        logging.error("No result files found to compare. Please run experiments first.")
        return

    summary_df = pd.DataFrame(results_data).T
    print("\n--- Performance Summary Table ---")
    print(summary_df.to_string(formatters={'bleu':'{:.3f}'.format, 'rougeL':'{:.3f}'.format, 'bertscore_f1':'{:.3f}'.format}))
    
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    configs_labels = list(summary_df.index)
    x = np.arange(len(metrics))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, config_name in enumerate(configs_labels):
        scores = summary_df.loc[config_name].values
        offset = width * (i - (len(configs_labels) - 1) / 2)
        rects = ax.bar(x + offset, scores, width, label=config_name)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    ax.set_ylabel('Average Score', fontsize=14)
    ax.set_title('Overall Model Performance Comparison', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(title='Configurations', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_ylim(0, 1)

    fig.tight_layout()
    chart_path = os.path.join(output_dir, 'final_comparison_chart.png')
    plt.savefig(chart_path)
    logging.info(f"\nFinal comparison chart saved to '{chart_path}'")
    plt.show()

def main():
    """Main function to orchestrate all experiments and comparisons."""
    parser = argparse.ArgumentParser(description="Master RAG Experiment Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo-data/NOTEEVENTS.csv", help="Path to MIMIC-III notes CSV")
    parser.add_argument("--max_notes", type=int, default=1000, help="Maximum number of notes to process")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save all outputs")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Path to the personalized query dataset")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    # --- DEFINE ALL EXPERIMENTS TO RUN ---
    experiments_to_run = [
        ('biobert', 'flan-t5'),
        ('pubmedbert', 'flan-t5'),
        # ('biobert', 'medalpaca'),      # Uncomment these as you are ready to test them
        # ('pubmedbert', 'medalpaca'),
    ]

    # --- PRE-LOAD DATA ---
    docs, titles = load_mimic_notes(args.mimic_path, args.max_notes)
    with open(args.eval_dataset, "r") as f:
        eval_data = json.load(f)

    # --- RUN ALL EXPERIMENTS SEQUENTIALLY ---
    for config in experiments_to_run:
        run_single_experiment(config, args, docs, titles, eval_data)

    # --- RUN FINAL COMPARISON ---
    run_final_comparison(experiments_to_run, args.output_dir)

    logging.info("\nMaster orchestrator has completed all tasks.")

if __name__ == "__main__":
    main()
