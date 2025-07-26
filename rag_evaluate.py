import numpy as np
import pandas as pd
import re
import argparse
import json
import logging
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from scipy.stats import ttest_rel
from rag_pipeline import rag_health_recommend, PUBMED_DOCS, DOC_TITLES, FAISS_INDEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# 1. EVALUATION DATA
# -------------------------------
def preprocess_text(text: str) -> str:
    """Clean MIMIC-III note text."""
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_mimic_samples(mimic_csv_path: str, diagnoses_csv_path: str = None, num_samples: int = 50) -> List[Dict]:
    """Create query-gold pairs from MIMIC-III notes."""
    logging.info("Creating evaluation samples...")
    try:
        df_notes = pd.read_csv(mimic_csv_path)
        if 'CATEGORY' in df.columns:
            df_notes = df_notes[df_notes['CATEGORY'] == 'Discharge summary'].head(num_samples)
        else:
            df_notes = df_notes.head(num_samples)
        
        samples = []
        df_diag = pd.read_csv(diagnoses_csv_path) if diagnoses_csv_path else None
        for _, row in df_notes.iterrows():
            condition = "unknown condition"
            if df_diag is not None:
                diag_row = df_diag[df_diag['HADM_ID'] == row['HADM_ID']]
                if not diag_row.empty:
                    condition = diag_row['ICD9_CODE'].iloc[0]
            query = f"What is the treatment for {condition}?"
            gold = extract_treatment(row['TEXT'])
            samples.append({"query": query, "gold": gold})
        logging.info(f"Created {len(samples)} evaluation samples.")
        return samples
    except Exception as e:
        logging.error(f"Failed to create samples: {e}")
        raise
def extract_treatment(note: str) -> str:
    """Extract treatment from note (simplified, needs medical expertise)."""
    note = preprocess_text(note)
    match = re.search(r'(Treatment|Plan|Medication):\s*(.*?)(?:\n|$)', note, re.I)
    return match.group(2).strip() if match else note[:100]

# -------------------------------
# 2. BASELINE LLM
# -------------------------------
def plain_llm_generate(user_query: str):
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=-1, max_length=128)
        prompt = f"User question: {user_query}\nGive a clear, safe, evidence-based health recommendation."
        return generator(prompt)[0]['generated_text']
    except Exception as e:
        logging.error(f"Baseline LLM failed: {e}")
        return f"Error: {str(e)}"

# -------------------------------
# 3. EVALUATION METRICS
# -------------------------------
def exact_match(pred: str, gold: str) -> bool:
    return gold.lower() in pred.lower()

def compute_bleu(pred: str, gold: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)

def compute_rouge(pred: str, gold: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)['rougeL'].fmeasure

def evaluate(samples: List[Dict], pipeline_func, system_name: str = "RAG", top_k: int = 2, faiss_index=None, docs=None, titles=None) -> List[Dict]:
    results = []
    preds = []
    golds = []
    logging.info(f"\n----- {system_name} Results -----")
    for s in samples:
        pred = pipeline_func(s["query"], top_k, faiss_index, docs, titles) if system_name == "RAG" else plain_llm_generate(s["query"])
        pred = pred['answer'] if isinstance(pred, dict) else pred
        preds.append(pred)
        golds.append(s["gold"])
        bleu = compute_bleu(pred, s["gold"])
        rouge = compute_rouge(pred, s["gold"])
        match = exact_match(pred, s["gold"])
        results.append({
            "system": system_name,
            "query": s["query"],
            "gold": s["gold"],
            "prediction": pred,
            "exact_match": match,
            "bleu": bleu,
            "rougeL": rouge
        })
        logging.info(f"\nQ: {s['query']}\nGOLD: {s['gold']}\nPRED: {pred}\nBLEU: {bleu:.2f} | ROUGE-L: {rouge:.2f} | Exact Match: {match}")
    
    # BERTScore
    try:
        P, R, F1 = bert_score(preds, golds, lang="en", verbose=True)
        logging.info(f"{system_name} BERTScore-F1 avg: {F1.mean():.3f}")
        for idx, r in enumerate(results):
            r['bertscore_f1'] = float(F1[idx])
    except Exception as e:
        logging.error(f"BERTScore failed: {e}")
    return results

def compute_statistical_significance(rag_results: List[Dict], base_results: List[Dict], metric: str) -> Dict:
    """Compute paired t-test for RAG vs. baseline."""
    try:
        rag_vals = [r[metric] if metric != "exact_match" else int(r[metric]) for r in rag_results]
        base_vals = [r[metric] if metric != "exact_match" else int(r[metric]) for r in base_results]
        t_stat, p_value = ttest_rel(rag_vals, base_vals)
        logging.info(f"{metric} t-test: t={t_stat:.2f}, p={p_value:.3f}")
        return {"t_stat": t_stat, "p_value": p_value}
    except Exception as e:
        logging.error(f"Statistical test failed: {e}")
        return {"t_stat": 0, "p_value": 1}

def create_chart(rag_results: List[Dict], base_results: List[Dict], output_file: str) -> None:
    """Generate Chart.js configuration for metrics."""
    from statistics import mean
    metrics = ["exact_match", "bleu", "rougeL", "bertscore_f1"]
    chart_data = {
        "type": "bar",
        "data": {
            "labels": ["RAG", "Plain LLM"],
            "datasets": [
                {
                    "label": metric.replace("_", " ").title(),
                    "data": [
                        mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in rag_results]),
                        mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in base_results])
                    ],
                    "backgroundColor": ["#4CAF50" if i == 0 else "#2196F3" for i in range(2)]
                } for metric in metrics
            ]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": true,
                    "max": 1,
                    "title": {"display": true, "text": "Metric Value"}
                },
                "x": {
                    "title": {"display": true, "text": "System"}
                }
            },
            "plugins": {
                "title": {
                    "display": true,
                    "text": "RAG vs. Plain LLM Performance"
                }
            }
        }
    }
    with open(output_file, "w") as f:
        json.dump(chart_data, f, indent=2)
    logging.info(f"Saved Chart.js configuration to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--mimic_path", default="data/mimic-iii/NOTEEVENTS.csv", help="Path to MIMIC-III notes")
    parser.add_argument("--diagnoses_path", default="data/mimic-iii/DIAGNOSES_ICD.csv", help="Path to MIMIC-III diagnoses")
    parser.add_argument("--output", default="outputs/eval_results.json", help="Output file for evaluation results")
    parser.add_argument("--chart_output", default="outputs/chart_config.json", help="Output file for Chart.js configuration")
    return parser.parse_args()

# -------------------------------
# 5. MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    
    # Create evaluation samples
    SAMPLES = create_mimic_samples(args.mimic_path, args.diagnoses_path, num_samples=50)
    
    # Run evaluations
    rag_results = evaluate(SAMPLES, rag_health_recommend, system_name="RAG", top_k=2, faiss_index=FAISS_INDEX, docs=PUBMED_DOCS, titles=DOC_TITLES)
    base_results = evaluate(SAMPLES, plain_llm_generate, system_name="PlainLLM")
    
    # Statistical significance
    compute_statistical_significance(rag_results, base_results, "bertscore_f1")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump({"rag_results": rag_results, "base_results": base_results}, f, indent=2)
    logging.info(f"Saved evaluation results to {args.output}")
    
    # Generate visualization
    create_chart(rag_results, base_results, args.chart_output)
    
    logging.info("\n🎯 Evaluation complete! Results and chart configuration saved.")