Certainly. Here is the complete, final version of the master `orchestrator.py` script.

This single script is designed to automate your entire research workflow. When you run it, it will:

1.  Evaluate the baseline FLAN-T5 model.
2.  Run all your defined RAG experiments (e.g., BioBERT + FLAN-T5, PubMedBERT + FLAN-T5).
3.  Generate the individual CSV result files for each.
4.  Create a final summary table and a comparison chart with all results.

### **Final `orchestrator.py` File**

```python
import os
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import all necessary functions from your pipeline and metric libraries
from rag_pipeline import (
    load_models,
    load_mimic_notes,
    build_or_load_faiss_index,
    rag_health_recommend
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import pipeline as baseline_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. SCRIPT SETUP ---

def setup_logging(output_dir: str):
    """Sets up a clean logger for each run."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "master_orchestrator.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def compute_metrics(pred: str, gold: str) -> dict:
    """Calculates BLEU and ROUGE-L scores for a prediction-gold pair."""
    smoothie = SmoothingFunction().method4
    pred, gold = str(pred), str(gold)
    pred_tokens, gold_tokens = pred.split(), [gold.split()]
    
    bleu = sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothie)
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge_l_scorer.score(gold, pred)['rougeL'].fmeasure
    return {"bleu": bleu, "rougeL": rouge_l}

# --- 2. CORE EXPERIMENT FUNCTIONS ---

def run_rag_experiment(config: tuple, args: argparse.Namespace, docs: list, titles: list, eval_data: list):
    """Runs one full RAG experiment: generation + evaluation."""
    retriever_name, generator_name = config
    logging.info(f"\n{'='*20} 🚀 Starting RAG Experiment: {retriever_name.upper()} + {generator_name.upper()} {'='*20}")
    
    models = load_models(retriever_name, generator_name)
    index_path = os.path.join(args.output_dir, f"faiss_index_{retriever_name}.bin")
    faiss_index = build_or_load_faiss_index(docs, index_path, models['retriever'])
    
    eval_results = []
    for item in eval_data:
        result = rag_health_recommend(item["query"], item["patient_context"], args.top_k, models, faiss_index, docs, titles)
        metrics = compute_metrics(result['answer'], item['gold'])
        eval_results.append({"query": item["query"], "context": item["patient_context"], "gold": item['gold'], "prediction": result['answer'], **metrics})
    
    eval_df = pd.DataFrame(eval_results)
    preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score(preds, golds, lang="en", verbose=False, device=device)
    eval_df['bertscore_f1'] = f1.tolist()

    output_path = os.path.join(args.output_dir, f"rag_evaluation_{retriever_name}_{generator_name}.csv")
    eval_df.to_csv(output_path, index=False)
    logging.info(f"✅ RAG evaluation complete. Results saved to '{output_path}'")

def run_baseline_experiment(args: argparse.Namespace, eval_data: list):
    """Runs the Baseline LLM evaluation."""
    logging.info(f"\n{'='*20} 🚀 Starting Baseline LLM Experiment {'='*20}")
    
    try:
        device_num = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float16, device_map="auto")
        generator = baseline_pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_num, max_new_tokens=150)
        
        eval_results = []
        for item in eval_data:
            prompt = f'PATIENT PROFILE: {item["patient_context"]}\n\nQuestion: {item["query"]}\n\nProvide a clear, evidence-based health recommendation.\nAnswer:'
            pred = generator(prompt)[0]['generated_text']
            metrics = compute_metrics(pred, item['gold'])
            eval_results.append({"query": item["query"], "context": item["patient_context"], "gold": item['gold'], "prediction": pred, **metrics})

        eval_df = pd.DataFrame(eval_results)
        preds, golds = eval_df["prediction"].tolist(), eval_df["gold"].tolist()
        _, _, f1 = bert_score(preds, golds, lang="en", verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
        eval_df['bertscore_f1'] = f1.tolist()

        output_path = os.path.join(args.output_dir, "baseline_evaluation_results.csv")
        eval_df.to_csv(output_path, index=False)
        logging.info(f"✅ Baseline evaluation complete. Results saved to '{output_path}'")
    except Exception as e:
        logging.error(f"FATAL ERROR during baseline experiment: {e}", exc_info=True)


def run_final_comparison(configurations: list, output_dir: str):
    """Loads all CSVs and generates a final comparison chart."""
    logging.info(f"\n{'='*20} 📊 Generating Final Comparison {'='*20}")

    systems_to_compare = { "Baseline LLM": os.path.join(output_dir, 'baseline_evaluation_results.csv') }
    for r_name, g_name in configurations:
        system_name = f"{r_name.title()} + {g_name.title()}"
        systems_to_compare[system_name] = os.path.join(output_dir, f'rag_evaluation_{r_name}_{g_name}.csv')

    results_data = {}
    for system_name, file_path in systems_to_compare.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            results_data[system_name] = df[['bleu', 'rougeL', 'bertscore_f1']].mean()
        else:
            logging.warning(f"Result file not found for comparison, skipping: {file_path}")

    if not results_data:
        logging.error("No result files were found to compare. Aborting chart generation.")
        return

    summary_df = pd.DataFrame(results_data).T
    print("\n--- Final Performance Summary ---")
    print(summary_df.to_string(formatters={'bleu':'{:.3f}'.format, 'rougeL':'{:.3f}'.format, 'bertscore_f1':'{:.3f}'.format}))
    
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    system_labels = list(summary_df.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(system_labels)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, system_name in enumerate(system_labels):
        scores = summary_df.loc[system_name].values
        offset = width * (i - (len(system_labels) - 1) / 2)
        rects = ax.bar(x + offset, scores, width, label=system_name)
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

# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """Defines experiments, parses arguments, and runs the entire workflow."""
    parser = argparse.ArgumentParser(description="Master RAG Experiment Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo-data/NOTEEVENTS.csv", help="Path to notes CSV")
    parser.add_argument("--max_notes", type=int, default=10, help="Maximum notes to process")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for all outputs")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Path to evaluation dataset")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    # --- DEFINE ALL EXPERIMENTS TO RUN ---
    rag_experiments_to_run = [
        ('biobert', 'flan-t5'),
        ('pubmedbert', 'flan-t5'),
         ('biobert', 'medalpaca'),  # Keep these commented until you are ready to test
         ('pubmedbert', 'medalpaca'),
    ]

    try:
        docs, titles = load_mimic_notes(args.mimic_path, args.max_notes)
        with open(args.eval_dataset, "r") as f:
            eval_data = json.load(f)

        if not docs or not eval_data:
            logging.error("No documents or evaluation data loaded. Aborting experiments.")
            return
        
        # --- RUN ALL DEFINED EXPERIMENTS SEQUENTIALLY ---
        
        # 1. Run Baseline First
        run_baseline_experiment(args, eval_data)
        
        # 2. Run all RAG experiments
        for config in rag_experiments_to_run:
            run_rag_experiment(config, args, docs, titles, eval_data)

        # 3. After all experiments are done, run the final comparison
        run_final_comparison(rag_experiments_to_run, args.output_dir)

    except Exception as e:
        logging.error("A fatal error occurred in the main execution block.", exc_info=True)
    
    logging.info("\nMaster orchestrator has completed all tasks.")


if __name__ == "__main__":
    main()
```
