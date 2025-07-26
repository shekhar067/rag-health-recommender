"""
Batch Evaluation & Visualization for RAG Health Recommender
- Exact-match, BLEU, ROUGE, BERTScore
- Per-sample fields for clarity, faithfulness, novelty (manual)
- Compares RAG vs plain LLM (no retrieval)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# NLG metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from rag_pipeline import rag_health_recommend

# ---------- 1. Sample Questions (add more fields as needed) ----------
SAMPLES = [
    {
        "query": "How should high blood pressure be treated?",
        "gold": "Lifestyle modification and medications such as ACE inhibitors.",
        "clarity": None,          # fill manually 1-5
        "faithfulness": None,     # fill manually 1-5
        "novelty": None           # fill manually 1-5
    },
    {
        "query": "What is first-line therapy for type 2 diabetes?",
        "gold": "Metformin remains a first-line drug for type 2 diabetes.",
        "clarity": None,
        "faithfulness": None,
        "novelty": None
    }
    # Add more samples here...
]

# ---------- 2. Optionally: Simple Plain LLM Baseline ----------
def plain_llm_generate(user_query: str):
    """Call only the generator with no context (baseline for comparison)."""
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=-1, max_length=128)
    prompt = f"User question: {user_query}\nGive a clear, safe, evidence-based health recommendation."
    return generator(prompt)[0]['generated_text']

# ---------- 3. Evaluation ----------
def exact_match(pred: str, gold: str) -> bool:
    return gold.lower() in pred.lower()

def compute_bleu(pred, gold):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)

def compute_rouge(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)['rougeL'].fmeasure

def evaluate(samples: List[Dict], pipeline_func, system_name="RAG"):
    results = []
    preds = []
    golds = []
    for s in samples:
        pred = pipeline_func(s["query"])
        preds.append(pred)
        golds.append(s["gold"])
        bleu = compute_bleu(pred, s["gold"])
        rouge = compute_rouge(pred, s["gold"])
        result = {
            "system": system_name,
            "query": s["query"],
            "gold": s["gold"],
            "prediction": pred,
            "exact_match": exact_match(pred, s["gold"]),
            "bleu": bleu,
            "rougeL": rouge,
            "clarity": s.get("clarity", None),
            "faithfulness": s.get("faithfulness", None),
            "novelty": s.get("novelty", None)
        }
        results.append(result)
        print(f"Q: {s['query']}\nGOLD: {s['gold']}\nPRED: {pred}\nBLEU: {bleu:.2f}  ROUGE-L: {rouge:.2f}\n")
    # BERTScore (average for whole set)
    P, R, F1 = bert_score(preds, golds, lang="en", verbose=True)
    print(f"{system_name} BERTScore-F1 avg: {F1.mean():.3f}")
    for idx, r in enumerate(results):
        r['bertscore_f1'] = float(F1[idx])
    with open(f"evaluation_results_{system_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

# ---------- 4. Visualization ----------
def visualize_metrics(rag_results, base_results):
    # For each metric: plot bar chart RAG vs Base
    metrics = ["exact_match", "bleu", "rougeL", "bertscore_f1"]
    titles = {
        "exact_match": "Accuracy",
        "bleu": "BLEU",
        "rougeL": "ROUGE-L",
        "bertscore_f1": "BERTScore-F1"
    }
    for metric in metrics:
        rag_val = np.mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in rag_results])
        base_val = np.mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in base_results])
        plt.figure(figsize=(4,3))
        plt.bar(["RAG", "Plain LLM"], [rag_val, base_val], color=["green", "gray"])
        plt.title(f"{titles[metric]}: RAG vs Plain LLM")
        plt.ylim(0,1)
        plt.ylabel(titles[metric])
        plt.show()

if __name__ == "__main__":
    print("=== Evaluating RAG pipeline ===")
    rag_results = evaluate(SAMPLES, rag_health_recommend, system_name="RAG")
    print("=== Evaluating plain LLM baseline ===")
    base_results = evaluate(SAMPLES, plain_llm_generate, system_name="PlainLLM")
    visualize_metrics(rag_results, base_results)
    print("Evaluation complete! Check JSON files for full details.")
